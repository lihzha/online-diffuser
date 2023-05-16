import os
import time
import copy
from os.path import join
from collections import deque
import sys
import torch.nn as nn
import torch
import numpy as np
import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.models.EBM import EBMDiffusionModel
import gymnasium as gym
import panda_gym
import cv2
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # dataset: str = 'maze2d-large-v1'
    dataset: str = 'PandaReach-v3'
    config: str = 'config.maze2d'

class gt_density_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,64),
            nn.Sigmoid(),
            nn.Linear(64,128),
            nn.Sigmoid(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        prob = self.net(x)
        # prob = torch.clamp(prob / torch.median(prob), max=1)
        return prob

def cycle(dl):
    while True:
        for data in dl:
            yield data

def _make_dir(args_path, dirname=None):
    time_now = time.gmtime()
    _time = str(time_now[1]) + '_' + str(time_now[2])
    # _time = '3_31'
    if dirname == None:
        folder = args_path + _time
    else:
        folder = args_path + _time + '_' + dirname
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def unify_args(args_train, args_d, args_v, args):
    args_list = ['device', 'n_diffusion_steps', 'horizon', 'max_path_length']
    for k in args_list:
        v = args_train.__getattribute__(k)
        args_v.__setattr__(k, v)
        args_d .__setattr__(k, v) 
        args.__setattr__(k, v)


class online_trainer:

    def __init__(self, policy, args_train, trainer_d, trainer_v):
        
        self.policy = policy 

        self.args_train = args_train

        self.value_trainer = trainer_v
        self.diffusion_trainer = trainer_d

        self.diffusion_trainer.guide = self.policy.guide
        self.diffusion_trainer.sample_kwargs = self.policy.sample_kwargs

        self.init_args()
        # self.refresh_buffer()
        self.buffer = copy.deepcopy(self.diffusion_trainer.dataset.fields)

        self.env = self.diffusion_trainer.dataset.env
        self.total_reward = 0
        self.score = []

        # if not args_train.ebm:
        #     state_dist = np.loadtxt('/home/lihan/diffuser-maze2d/useful/gt_density_model/final_gt.txt')
        #     self.neg_state_dist = torch.sigmoid(torch.tensor(-np.log(state_dist))).numpy()
        #     self.state_dist_flat = self.neg_state_dist.flatten()
        
        self.ep_num = deque([0,0],maxlen=2)
        self.sample_num = deque([0,0],maxlen=2)
        # specific to skew-fit
        # obs_dim = trainer_v.ema_model.transition_dim - trainer_v.ema_model.cond_dim
        # self.vae = LinearVAE(obs_dim)
        # self.q_goal = q_goal(model=self.vae, dataset=self.value_trainer.dataset.fields)

    def init_args(self):
        """Override the arguments in preloaded trainer."""
        
        args_list = ['discount', 'max_n_episodes', 'max_path_length', 'termination_penalty', \
                    'batch_size', 'horizon', 'normed', 'use_padding', 'normalizer', 'buffer_size']
        for k in args_list:
            v = self.args_train.__getattribute__(k)
            self.__setattr__(k, v)

        # refresh dataset params
        
        args_list = ['horizon', 'max_n_episodes', 'use_padding', 'max_path_length']
        for dataset in [self.diffusion_trainer.dataset]:
            self.override_dataset_args(dataset, self, args_list)

    def refresh_buffer(self):
        """Clean the old datasets from the field."""

        self.diffusion_trainer.dataset.fields = ReplayBuffer(self.max_n_episodes, self.max_path_length, \
                                            self.termination_penalty)
        if not self.args_train.reinit_buffer:
            self.buffer = ReplayBuffer(self.max_n_episodes, self.max_path_length, \
                                            self.termination_penalty)
        else:
            obs = np.load('/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/4_14_ebm_scratch_changing_pexplore_scale0.01/buffer_vis.npy')[-500:]
            self.buffer = ReplayBuffer(self.max_n_episodes, self.max_path_length, 
                                            self.termination_penalty)
            self.buffer.reinit(obs)


    def train(self):
        """Online training scenerio."""

        total_reward = []
        total_score = []
        open_loop = False

        for it in range(self.args_train.iterations):

        # --------------------------- Initialization ----------------------------#
            self.total_reward = 0
            episode = {}
            if self.args_train.predict_action:
                actions, next_obs, obs, rew, terminals = [], [], [], [], []
            else:
                next_obs, obs, rew, terminals = [], [], [], []
            obs_info = self.env.reset()
            self.observation = obs_info[0]['observation']
            # self.args_train.set_t = min(0.5/3000 * it,0.5)
            # p_explore = self.args_train.p_explore
        # --------------------------- Setting target ----------------------------#
            # set_t = np.random.binomial(1,self.args_train.set_t)
            target = obs_info[0]['desired_goal']
            # if self.args_train.conditional:
            #     print('Resetting target')
            #     if set_t:
            #         try:
            #             target = self.sample_target()
            #             self.env.set_target(target)
            #         except:
            #             self.env.set_target()
            #         target = self.env._target
            #         cond = {
            #             self.horizon - 1: np.array([*target, 0, 0])
            #         }
            #     else:
            #         self.env.set_target()
            #         target = self.env._target
            cond = {
                self.horizon - 1: np.array([*target, 0, 0, 0]),
            }      

            # ------------------------- For renderering --------------------------#
            self.rollout = [np.concatenate((target,(0.,0.))),self.observation.copy()]

            for t in range(self.max_path_length):
                
                # state = self.env.state_vector().copy()
                
                if open_loop:
                    if t == 0:
                        cond[0] = self.observation
                        if self.args_train.predict_action:
                            action, samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=p_explore)
                        else:
                            samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=p_explore)
                        sequence = samples.observations[0]

                    # if t < len(sequence) - 1:
                    #     next_waypoint = sequence[t+1]
                    # else:
                    #     next_waypoint = sequence[-1].copy()
                    #     next_waypoint[2:] = 0
                    # action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                else:
                    if t % self.horizon == 0:
                        cond[0] = self.observation
                        # state = self.env.state_vector().copy()
                        if self.args_train.predict_action:
                            action, samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=0)
                        else:
                            samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=0)
                    # sequence = samples.observations[0]
                    # try:
                    #     next_waypoint = sequence[t-t//self.horizon*self.horizon+1].copy()
                    # except:
                    #     next_waypoint = sequence[t-t//self.horizon*self.horizon].copy()
                    # action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_observation = next_observation['observation']
                # cv2.imwrite('trial_rendering.png',self.env.render())
                if np.linalg.norm((next_observation - self.observation)[:2]) < 1e-3 and it!=0:
                    break

                self.total_reward += reward
                # score = self.env.get_normalized_score(self.total_reward)
                self.rollout.append(next_observation.copy())
                if 'maze2d' in self.args_train.dataset:
                    xy = next_observation[:2]
                    goal = self.env.unwrapped._target
                    print(
                        f'it: {it} | maze | pos: {xy} | goal: {goal}'
                    )
                else:
                    xy = next_observation[:3]
                    print(
                        f'it: {it} | panda | pos: {xy} | goal: {target}'
                    )

                if self.args_train.predict_action:
                    actions.append(action)               
                next_obs.append(next_observation)
                obs.append(self.observation)
                rew.append(reward)
                terminals.append(terminated)

                if terminated:
                    break

                self.observation = next_observation
                print(t)
            

            if len(obs) >= self.horizon:
                if self.args_train.predict_action:
                    episode = self.format_episode(actions,next_obs,obs,rew,terminals)
                else:
                    episode = self.format_episode(None,next_obs,obs,rew,terminals)
                self.add_to_fields(episode)
            
            # if len(obs) > 0:
            #     if self.args_train.predict_action:
            #         episode = self.format_episode(actions,next_obs,obs,rew,terminals)
            #     else:
            #         episode = self.format_episode(None,next_obs,obs,rew,terminals)
            #     self.add_to_fields(episode)

            if it == 0:
                dataset = self.diffusion_trainer.dataset
                self.initialize_normalizer(dataset)

            print("----------------",it,"round finished------------------")
            print("total_reward", self.total_reward)
            total_reward.append(self.total_reward)
            print('Average total reward is:', sum(total_reward)/len(total_reward))
            print('Non-zero rewards is:', len(np.nonzero(total_reward)[0])/len(total_reward)*100, "%")
            # total_score.append(score) 

                
            if it % self.args_train.train_freq == self.args_train.train_freq - 1:
                num_trainsteps = self.process_dataset(it)
                # self.save_buffer()
                self.diffusion_trainer.train(num_trainsteps, p_explore=0, online=True)
                # self.test(it)
                # for i in range(100):
                #     o = 10*i+2
                #     self.test(o)
                # sys.exit(print(self.score,sum(self.score)/len(self.score)))

        print(total_reward)
        print(total_score)

    def test(self, it):

        open_loop = False
        self.total_reward = 0
        self.observation = self.env.reset()
        self.env.set_target((7,9))
        target = self.env._target
        cond = {
                self.horizon - 1: np.array([*target, 0, 0]),
            }
        self.rollout = [np.concatenate((target,(0.,0.))),self.observation.copy()]
        
        for t in range(self.max_path_length):
            
            state = self.env.state_vector().copy()
            if open_loop:
                if t == 0:
                    cond[0] = self.observation
                    if self.args_train.predict_action:
                        action, samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=0)
                    else:
                        samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=0)
                    sequence = samples.observations[0]

                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            else:
                if t % self.horizon == 0:
                    cond[0] = self.observation
                    state = self.env.state_vector().copy()
                    if self.args_train.predict_action:
                        action, samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=0)
                    else:
                        samples = self.policy(cond, it, batch_size=self.args_train.batch_size, verbose=self.args_train.verbose, p_explore=0)
                sequence = samples.observations[0]
                try:
                    next_waypoint = sequence[t-t//self.horizon*self.horizon+1].copy()
                except:
                    next_waypoint = sequence[t-t//self.horizon*self.horizon].copy()
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, reward, terminal, _ = self.env.step(action)
            self.total_reward += reward
            
            if 'maze2d' in self.args_train.dataset:
                xy = next_observation[:2]
                goal = self.env.unwrapped._target
                print(
                    f'maze | pos: {xy} | goal: {goal}'
                )
            self.rollout.append(next_observation.copy())
        
            if t % 299 == 0:
                fullpath = join(self.args_train.savepath, '{}_{}_test.png'.format(it,t//self.horizon))
                renderer.composite(fullpath, samples.observations, ncol=1)

            if t % self.args_train.vis_freq == 0 or terminal:

                # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                renderer.composite(join(self.args_train.savepath, '{}_test_rollout.png'.format(it)), np.array(self.rollout)[None], ncol=1)

                # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)
            if terminal:
                break

            self.observation = next_observation
            print(t)
        
        score = self.env.get_normalized_score(self.total_reward)
        self.score.append(score)
        print(self.total_reward, score)
        # np.savetxt('TestReward_{}_round'.format(it),self.total_reward)

    def save_buffer(self):
        if self.args_train.predict_action:
            buffer = np.concatenate((self.diffusion_trainer.dataset.fields['actions'],self.diffusion_trainer.dataset.fields['observations']),axis=2)
        else:
            buffer = self.diffusion_trainer.dataset.fields['observations']
        path = '/home/lihan/diffuser-maze2d/' + args_d.savepath + '/' + 'buffer_vis.npy'
        np.save(path, buffer)

    def format_episode(self,actions,next_obs,obs,rew,terminals):
        """Turn generated samples into episode format."""

        episode = {}
        episode_length = len(obs)
        if actions != None:
            episode['actions'] = np.array(actions).reshape((episode_length,-1))
        episode['next_observations'] = np.array(next_obs).reshape((episode_length,-1))
        episode['observations'] = np.array(obs).reshape((episode_length,-1))
        episode['rewards'] = np.array(rew).reshape((episode_length,-1))
        episode['terminals'] = np.array(terminals).reshape((episode_length,-1))

        return episode

    def add_to_fields(self, episode):
        """Update the field with newly-generated samples."""

        self.buffer.add_path(episode, online=True)
        print(self.buffer)

    def process_dataset(self, it):
        """Normalize and preprocess the training data."""

        # buffer_size = self.buffer['actions'].shape[0]
        # self.augment_buffer(buffer_size//4)

        dataset = self.diffusion_trainer.dataset
        self.initialize_normalizer(dataset)
        obs_energy = self.compute_buffer_energy(dataset)
        buffer_size = dataset.fields['observations'].shape[0]
        # self.ep_num[1] = self.ep_num[0] + self.ep_num[1]
        # ep_this_turn = buffer_size - self.ep_num[1]
        # self.ep_num.append(ep_this_turn)
        # sample_size=1000
        sample_size = np.random.randint(buffer_size // 2, int(buffer_size))
        print("sample_size", sample_size)
        # batch_size = min(sample_size, 64)
        # self.sample_num[1] = self.sample_num[0] + self.sample_num[1]
        # self.sample_num.append(sample_size)
        self.energy_sampling(dataset.fields, sample_size, obs_energy)
        # self._sample_data(dataset.fields, sample_size)    
        dataset.indices = dataset.make_indices(dataset.fields.path_lengths, dataset.horizon)

        # Get buffer for behaviour policy
        # if it <= 99:
        #     train_freq = 20
        #     buffer_size = it
        #     num_trainsteps = it * 10
        # elif it <= 499:
        #     train_freq = 50
        #     buffer_size = it // 2
        #     num_trainsteps = min(it * 20, 8000)
        # elif it <= 1000:
        #     train_freq = 100
        #     buffer_size = it // 2
        #     num_trainsteps = min(buffer_size * 15, 20000)
        # else:
        #     train_freq = 200
        #     buffer_size = it // 2
        #     num_trainsteps = min(buffer_size * 15, 20000)
        
        # train_dataset = copy.deepcopy(self.diffusion_trainer.dataset)
        # self._sample_data(train_dataset.fields,buffer_size)
        # train_dataset.indices = train_dataset.make_indices(train_dataset.fields.path_lengths, self.diffusion_trainer.dataset.horizon)

        # The same setting for diffusion dataset
        # args_list = ['fields', 'indices']
        # self.override_dataset_args(diffusion_dataset, train_dataset, args_list)
        # args_list = ['horizon', 'n_episodes', 'max_path_length', 'normalizer', 'path_lengths']
        # self.override_dataset_args(diffusion_dataset, self.value_trainer.dataset, args_list)


        # Create dataloader using processed dataset
        # self.value_trainer.dataloader = cycle(torch.utils.data.DataLoader(
        #     train_dataset, batch_size=32, num_workers=1, shuffle=True, pin_memory=True
        # ))
        self.diffusion_trainer.dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=32, \
            num_workers=1, shuffle=True, pin_memory=True
        ))  #### should be train_dataset
        self.diffusion_trainer.dataloader_vis = cycle(torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
            ))

        num_trainsteps = min(sample_size*5, 10000)
        return num_trainsteps

    def initialize_normalizer(self, dataset):

        dataset.fields = copy.deepcopy(self.buffer)
        dataset.fields.finalize()
        dataset.normalizer = DatasetNormalizer(dataset.fields, normalizer=self.normalizer, \
                                        path_lengths=dataset.fields['path_lengths'])
        print('Using normalizer: ',self.normalizer)
        dataset.n_episodes = dataset.fields.n_episodes
        dataset.path_lengths = dataset.fields['path_lengths']
        dataset.normalize()
        self.policy.normalizer = dataset.normalizer

    def _sample_data(self, fields, sample_size):
        """Sample training data from fields."""

        _dict = fields._dict
        obs = _dict['observations']
        obs_dim = obs.shape[0]

        index = np.zeros(obs_dim,dtype=np.int32)
        for i in range(obs_dim):
            index[i] = obs[i,:,0].nonzero()[0][-1]
        traj_len = np.array([np.linalg.norm(obs[i][index[i]][:2] - obs[i][0][:2]) for i in range(obs_dim)])
        p_sample_len = traj_len / np.sum(traj_len)
        # traj_rewards = np.array([np.exp(_dict['rewards'][i].sum()) for i in range(_dict['rewards'].shape[0])])
        # p_sample = traj_rewards / np.sum(traj_rewards)

        if self.args_train.ebm:
            if self.args_train.predict_action:
                actions = _dict['actions']
                trajectories = np.concatenate([actions, obs], axis=-1)
            else:
                trajectories = obs
            trajectories = torch.tensor(trajectories, dtype=torch.float, device=self.args_train.device)
            t = torch.full((trajectories.shape[0],), 0, device=self.args_train.device, dtype=torch.long)
            energy = self.policy.diffusion_model.model.point_energy(trajectories, None, t).sum(-1)
            p_sample_prob = (energy / energy.sum()).detach().cpu().numpy()
        else:
            obs = torch.tensor(obs[:,:,:2], dtype=torch.float, device=self.args_train.device)
            raw_density = gt_density(obs).squeeze(-1)
            for i in range(obs_dim):
                raw_density[i,index[i]+1:] = 0
            p_sample_prob = 1 / (raw_density.sum(-1))
            p_sample_prob = (p_sample_prob / p_sample_prob.sum()).detach().cpu().numpy()
        
        p_sample = p_sample_len * p_sample_prob
        p_sample = p_sample / p_sample.sum()

        # sample_index = np.random.choice(obs_dim,size=sample_size,replace=False,p=p_sample)
        sample_index = np.random.choice(obs_dim,size=sample_size,replace=True, p=p_sample_len)

        # p_new = min((self.sample_num[-2] + self.sample_num[-1]) * self.ep_num[-1] / self.sample_num[-1] \
        #         / (self.ep_num[-2] + self.ep_num[-1]), 1.) 
        # p_old = 1 - p_new
        # p_new_array = p_new * np.ones((self.ep_num[-1])) / self.ep_num[-1] 
        # p_old_array = p_old * np.ones((self.ep_num[-2])) / self.ep_num[-2]
        # p_uniform = np.concatenate((p_old_array,p_new_array))
        # print(p_new, p_old, self.ep_num, self.sample_num)        
        # sample_index = np.random.choice(obs_dim,size=sample_size,p=p_uniform,replace=True)
        for key in _dict.keys():
            _dict[key] = _dict[key][sample_index]
        fields._add_attributes()

    def energy_sampling(self, fields, sample_size, energy):
        """Sample training data from fields."""

        _dict = fields._dict
        obs = _dict['observations']
        obs_dim = obs.shape[0]

        # sample_index = np.random.choice(obs_dim,size=sample_size,replace=False,p=p_sample)
        sample_index = np.random.choice(obs_dim,size=sample_size,replace=True, p=energy/energy.sum())

        # p_new = min((self.sample_num[-2] + self.sample_num[-1]) * self.ep_num[-1] / self.sample_num[-1] \
        #         / (self.ep_num[-2] + self.ep_num[-1]), 1.) 
        # p_old = 1 - p_new
        # p_new_array = p_new * np.ones((self.ep_num[-1])) / self.ep_num[-1] 
        # p_old_array = p_old * np.ones((self.ep_num[-2])) / self.ep_num[-2]
        # p_uniform = np.concatenate((p_old_array,p_new_array))
        # print(p_new, p_old, self.ep_num, self.sample_num)        
        # sample_index = np.random.choice(obs_dim,size=sample_size,p=p_uniform,replace=True)
        for key in _dict.keys():
            _dict[key] = _dict[key][sample_index]
        fields._add_attributes()
   
    def override_dataset_args(self, dataset, args_source, args_list):
        """Set the hyperparameters for diffusion dataset."""

        for k in args_list:
            v = copy.deepcopy(args_source.__getattribute__(k))
            dataset.__setattr__(k, v)

    def augment_buffer(self, size):
        """Augment the trajectories in the buffer, e.g. reverse the trajectory, concatenate short trajectories."""

        obs = self.buffer['observations']
        obs_dim = self.buffer.n_episodes
        index = np.zeros(obs_dim,dtype=np.int32)
        for i in range(obs_dim):
            index[i] = obs[i,:,0].nonzero()[0][-1]
        traj_len = np.array([np.linalg.norm(obs[i][index[i]][:2] - obs[i][0][:2]) for i in range(obs_dim)])
        p_sample = traj_len / np.sum(traj_len)
        sample_index = np.random.choice(obs_dim,size=size,replace=True,p=p_sample)

        for ind in sample_index:
            actions = np.flipud(self.buffer['actions'][ind][:index[ind]+1])
            next_obs = np.flipud(self.buffer['next_observations'][ind][:index[ind]+1])
            obs = np.flipud(self.buffer['observations'][ind][:index[ind]+1])
            rew = np.flipud(self.buffer['rewards'][ind][:index[ind]+1])
            terminals = np.flipud(self.buffer['terminals'][ind][:index[ind]+1])
            episode = self.format_episode(actions,next_obs,obs,rew,terminals)
            self.buffer.add_path(episode)

    def compute_buffer_energy(self, dataset):
        """Randomly sample targets according to energy."""

        indices = dataset.make_new_indices(dataset.fields.path_lengths, dataset.horizon)
        sample_ind = list(range(0, indices.shape[0]))
        obs = dataset.get_obs(sample_ind, indices)
        
        if len(obs.shape) == 2:
            obs = obs.reshape((1,-1,dataset.observation_dim))
        obs = torch.tensor(obs, device=self.args_train.device, dtype=torch.float32)
        t = torch.zeros(obs.shape[0],device=obs.device)
        cond = {0:obs.clone().detach()[:,0], dataset.horizon-1:obs.clone().detach()[:,dataset.horizon-1]}
        energy = self.diffusion_trainer.ema_model.model.point_energy(obs, cond, t).cpu().detach().numpy().sum(-1)
        obs_energy = np.zeros(dataset.fields['observations'].shape[0])
        cnt = 0
        last_path_ind = -1
        for index in indices:
            path_ind, _, _ = index
            if path_ind != last_path_ind:
                k = 1
            last_path_ind = path_ind
            # obs_energy[path_ind] += 1/k*(energy[cnt]-obs_energy[path_ind])
            obs_energy[path_ind] += energy[cnt]
            k += 1
            cnt += 1
        return obs_energy

    def sample_target(self):

        # sample_idx = np.random.choice(np.prod(self.obs_energy.shape), size=1, p=self.obs_energy.flatten()/self.obs_energy.sum())[0]
        # target = self.target_set[sample_idx//self.target_set.shape[1], sample_idx-sample_idx//self.target_set.shape[1]*self.target_set.shape[1]][:2]
        
        traj_shape = (50, self.horizon, self.diffusion_trainer.model.observation_dim)
        target_traj = utils.arrays.sample_from_array(np.array(([ 0.48975977,  0.50192666, -5.2262554 , -5.2262554 ],[ 7.213778 , 10.215629 ,  5.2262554,  5.2262554]),dtype=np.float32), traj_shape=traj_shape)
        obs = torch.tensor(target_traj, device=self.args_train.device, dtype=torch.float32)
        t = torch.zeros(obs.shape[0],device=obs.device)
        cond = None
        self.obs_energy = self.diffusion_trainer.ema_model.model.point_energy(obs, cond, t).cpu().detach().numpy()
        self.target_set = obs.cpu().detach().numpy()
        self.obs_energy[self.target_set.sum(-1)==0] = 0
        sample_idx = np.random.choice(np.prod(self.obs_energy.shape), size=1, p=self.obs_energy.flatten()/self.obs_energy.sum())[0]
        target = self.target_set[sample_idx//self.target_set.shape[1], sample_idx-sample_idx//self.target_set.shape[1]*self.target_set.shape[1]][:2]
        
        # self.obs_energy[torch.div(target_ind,self.target_set.shape[1],rounding_mode='trunc'), target_ind-torch.div(target_ind,self.target_set.shape[1],rounding_mode='trunc')*self.target_set.shape[1]] = 0
        # sample_index = np.random.choice(a=self.state_dist_flat.size, p=self.state_dist_flat/self.state_dist_flat.sum())
        # adjusted_index = np.unravel_index(sample_index, self.neg_state_dist.shape)
        # target = np.array(adjusted_index) / 100


        return target

if __name__ == "__main__":
    
    #-----------------------------------------------------------------------------#
    #---------------------------------- unifying arguments -----------------------#
    #-----------------------------------------------------------------------------#
    args = Parser().parse_args('plan')
    args_train = Parser().parse_args('online_training')
    args_d = Parser().parse_args('diffusion')
    args_v = Parser().parse_args('values')
    unify_args(args_train, args_d, args_v, args)

    dirname_d = args_train.dirname_d
    args_d.savepath = _make_dir('logs/maze2d-large-v1/diffusion/',dirname_d)
    args_d.save()

    args_v.savepath = 'logs/maze2d-large-v1/values/defaults_d0.997'
    args_v.save()

    args.savepath = _make_dir('logs/maze2d-large-v1/plans/',dirname_d)
    args.save()

    #-----------------------------------------------------------------------------#
    #---------------------------------- initialize diffusion ---------------------#
    #-----------------------------------------------------------------------------#
    
    env = gym.make(args_train.env, render_mode="rgb_array")

    dataset_config_d = utils.Config(
        args_d.loader,
        savepath=(args_d.savepath, 'dataset_config_d.pkl'),
        env=env,
        horizon=args_d.horizon,
        normalizer=args_d.normalizer,
        preprocess_fns=args_d.preprocess_fns,
        use_padding=args_d.use_padding,
        max_path_length=args_d.max_path_length,
        predict_action = args_train.predict_action,
        online=args_train.online,
    )
    renderer_d = env.render
    # render_config_d = utils.Config(
    #     args_d.renderer,
    #     savepath=(args_d.savepath, 'render_config_d.pkl'),
    #     env=args_d.dataset,
    # )
    dataset_d = dataset_config_d()
    # renderer_d = render_config_d()
    observation_dim = env.observation_space['observation'].shape[0]
    action_dim = env.action_space.shape[0]
    if args_train.predict_action_only:
        transition_dim = action_dim
    else:
        transition_dim = observation_dim + action_dim
    model_config_d = utils.Config(
        args_d.model,
        savepath=(args_d.savepath, 'model_config_d.pkl'),
        horizon=args_d.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=args_d.dim_mults,
        # attention=args_d.attention,
        device=args_d.device,
    )
    
    diffusion_config_d = utils.Config(
        args_d.diffusion,
        savepath=(args_d.savepath, 'diffusion_config_d.pkl'),
        horizon=args_d.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        ddim = args_train.ddim,
        ddim_timesteps = args_train.ddim_timesteps,
        n_timesteps=args_d.n_diffusion_steps,
        loss_type=args_train.loss_type,
        clip_denoised=args_d.clip_denoised,
        predict_epsilon=args_train.predict_epsilon,
        ## loss weighting
        action_weight=args_d.action_weight,
        loss_weights=args_d.loss_weights,
        loss_discount=args_d.loss_discount,
        device=args_d.device,
        predict_action=args_train.predict_action
    )

    trainer_config_d = utils.Config(
        utils.Trainer,
        device=args.device,
        savepath=(args_d.savepath, 'trainer_config_d.pkl'),
        train_batch_size=args_d.batch_size,
        train_lr=args_d.learning_rate,
        gradient_accumulate_every=args_d.gradient_accumulate_every,
        ema_decay=args_d.ema_decay,
        sample_freq=args_d.sample_freq,
        save_freq=args_d.save_freq,
        label_freq=int(args_d.n_train_steps // args_d.n_saves),
        save_parallel=args_d.save_parallel,
        results_folder=args_d.savepath,
        bucket=args_d.bucket,
        n_reference=args_d.n_reference,
        n_samples = args_d.n_samples,
        online=args_train.online
    )
    model_d = model_config_d()

    if args_train.ebm:
        model_d = EBMDiffusionModel(model_d)
        print("ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    diffusion_d = diffusion_config_d(model_d)
    trainer_d = trainer_config_d(diffusion_d, dataset_d, renderer_d, args.device)
    if args_train.load_model:
        trainer_d.load('/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/5_9_silu_sample_t/state_80000.pt')
    #-----------------------------------------------------------------------------#
    #------------------------------- initialize value ----------------------------#
    #-----------------------------------------------------------------------------#
    # dataset_config_v = utils.Config(
    #     args_v.loader,
    #     savepath=(args_v.savepath, 'dataset_config_v.pkl'),
    #     env=args_v.dataset,
    #     horizon=args_v.horizon,
    #     normalizer=args_v.normalizer,
    #     preprocess_fns=args_v.preprocess_fns,
    #     use_padding=args_train.use_padding,
    #     max_path_length=args_d.max_path_length,
    #     ## value-specific kwargs
    #     discount=args_train.discount,
    #     termination_penalty=args_train.termination_penalty,
    #     online=True,
    #     # normed=args_train.normed,
    # )
    # render_config_v = utils.Config(
    #     args_v.renderer,
    #     savepath=(args_v.savepath, 'render_config_v.pkl'),
    #     env=args_v.dataset,
    # )
    # dataset_v = dataset_config_v()
    # renderer_v = render_config_v()

    # model_config_v = utils.Config(
    #     args_v.model,
    #     savepath=(args_v.savepath, 'model_config_v.pkl'),
    #     horizon=args_v.horizon,
    #     transition_dim=observation_dim + action_dim,
    #     cond_dim=observation_dim,
    #     dim_mults=args_v.dim_mults,
    #     device=args_v.device,
    # )
    # diffusion_config_v = utils.Config(
    #     args_v.diffusion,
    #     savepath=(args_v.savepath, 'diffusion_config_v.pkl'),
    #     horizon=args_v.horizon,
    #     observation_dim=observation_dim,
    #     action_dim=action_dim,
    #     ddim = True,
    #     ddim_timesteps = args_train.ddim_timesteps,
    #     n_timesteps=args_v.n_diffusion_steps,
    #     loss_type=args_v.loss_type,
    #     device=args_v.device,
    # )
    # trainer_config_v = utils.Config(
    #     utils.Trainer,
    #     device=args.device,
    #     savepath=(args_v.savepath, 'trainer_config_v.pkl'),
    #     train_batch_size=args_v.batch_size,
    #     train_lr=args_v.learning_rate,
    #     gradient_accumulate_every=args_v.gradient_accumulate_every,
    #     ema_decay=args_v.ema_decay,
    #     sample_freq=args_v.sample_freq,
    #     save_freq=args_v.save_freq,
    #     label_freq=int(args_v.n_train_steps // args_v.n_saves),
    #     save_parallel=args_v.save_parallel,
    #     results_folder=args_v.savepath,
    #     bucket=args_v.bucket,
    #     n_reference=args_v.n_reference,
    #     n_samples = args_v.n_samples,
    #     online=True
    # )
    # model_v = model_config_v()
    # diffusion_v = diffusion_config_v(model_v)
    # trainer_v = trainer_config_v(diffusion_v, dataset_v, renderer_v, args.device)

    ## initialize policy arguments
    diffusion = trainer_d.ema_model
    dataset = dataset_d
    renderer = renderer_d
    # value_function = trainer_v.ema_model
    if not args_train.ebm:
        gt_density = gt_density_model()
        gt_density.load_state_dict(torch.load('/home/lihan/diffuser-maze2d/useful/gt_density_model/final_gt.pth'))
        gt_density = gt_density.to(device=args_train.device)
        guide_config = utils.Config(args_train.guide, model=gt_density, verbose=False)
        guide = guide_config()
    
    else:
        ebm_density = model_d
        guide_config = utils.Config(args_train.guide, model=ebm_density, verbose=False)
        guide = guide_config()

    ## policies are wrappers around an unconditional diffusion model and a value guide
    if args_train.ddim == True:
        sample = sampling.n_step_guided_ddim_sample
    else:
        sample = sampling.n_step_guided_p_sample

    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args_train.scale,
        diffusion_model=diffusion,
        normalizer=None,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs
        sample_fn=sample,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        eta=args.eta,
        verbose=False,
        _device=args_train.device
    )

    policy = policy_config()

    _online_trainer = online_trainer(policy, args_train, \
                   trainer_d, None)
    _online_trainer.train()