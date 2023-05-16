import os
import time
import torch
import numpy as np
import copy
from os.path import join

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
# from diffuser.models.encoder import q_goal
# from diffuser.models.encoder import LinearVAE
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

def cycle(dl):
    while True:
        for data in dl:
            yield data

def _make_dir(args_path, dirname=None):
    time_now = time.gmtime()
    _time = str(time_now[1]) + '_' + str(time_now[2])
    _time = '3_31'
    if dirname == None:
        folder = args_path + _time
    else:
        folder = args_path + dirname + _time
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

    def __init__(self, policy, args, args_train, trainer_d, trainer_v):
        
        self.policy = policy 

        self.args = args
        self.args_train = args_train

        self.value_trainer = trainer_v
        self.diffusion_trainer = trainer_d


        self.init_args()
        self.refresh_buffer()

        self.env = self.value_trainer.dataset.env
        self.total_reward = 0

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

        self.buffer = ReplayBuffer(self.max_n_episodes, self.max_path_length, \
                                             self.termination_penalty)


    def train(self):
        """Online training scenerio."""

        total_reward = []
        total_score = []
        # target_reward = []

        open_loop = False
        replan_freq = 100


        for it in range(self.args_train.iterations):

        # -------------------Generating samples ----------------------------#
            self.total_reward = 0
            episode = {}
            actions, next_obs, obs, rew, terminals = [], [], [], [], []
            self.observation = self.env.reset()
            if args_train.conditional:
                print('Resetting target')
                self.env.set_target()
                # target = self.q_goal.sample()
                # self.env.set_target(target)
            target = self.env._target
            cond = {
                    self.horizon - 1: np.array([*target, 0, 0]),
                }
            self.rollout = [np.concatenate((target,(0.,0.))),self.observation.copy()]
            # dist_array = np.zeros((self.max_path_length,4))
            # state_array = np.zeros_like(dist_array)
            for t in range(self.max_path_length):
                
                state = self.env.state_vector().copy()
                if open_loop:
                    if t == 0:
                        cond[0] = self.observation
                        action, samples = self.policy(cond, it, batch_size=self.args.batch_size, verbose=self.args.verbose)
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
                        action, samples = self.policy(cond, it, batch_size=self.args.batch_size, verbose=self.args.verbose)
                    sequence = samples.observations[0]
                    try:
                        next_waypoint = sequence[t-t//self.horizon*self.horizon+1].copy()
                    except:
                        next_waypoint = sequence[t-t//self.horizon*self.horizon].copy()

                    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

                next_observation, reward, terminal, _ = self.env.step(action)

                if np.linalg.norm((next_observation - self.observation)[:2]) < 1e-3 and it!=0:
                    break

                self.total_reward += reward
                score = self.env.get_normalized_score(self.total_reward)
                
                ###

                # print(
                #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                #     f'{action}'
                # )
                if 'maze2d' in args.dataset:
                    xy = next_observation[:2]
                    goal = self.env.unwrapped._target
                    print(
                        f'maze | pos: {xy} | goal: {goal}'
                    )
                self.rollout.append(next_observation.copy())
                
                # if t % args.vis_freq == 0 or terminal:
                #     fullpath = join(args.savepath, f'{t}.png')

                #     if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


                    # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                    ## save rollout thus far
                # renderer.composite(join(args.savepath, 'rollout.png'), np.array(self.rollout)[None], ncol=1)

                    # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

                actions.append(action)
                next_obs.append(next_observation)
                obs.append(self.observation)
                rew.append(reward)
                terminals.append(terminal)
                if terminal:
                    break

                self.observation = next_observation
                print(t)
            
                ## render every `args.vis_freq` steps
            if len(actions) >= self.horizon:
                episode = self.format_episode(actions,next_obs,obs,rew,terminals)
                self.add_to_fields(episode)

            if it == 0:
                dataset = self.diffusion_trainer.dataset
                dataset.fields = copy.deepcopy(self.buffer)
                dataset.fields.finalize()
                dataset.normalizer = DatasetNormalizer(dataset.fields, normalizer=self.normalizer, \
                                                path_lengths=dataset.fields['path_lengths'])
                dataset.n_episodes = dataset.fields.n_episodes
                dataset.path_lengths = dataset.fields['path_lengths']
                dataset.normalize()
                self.policy.normalizer = dataset.normalizer

            print("----------------",it,"round finished------------------")
            print("total_reward", self.total_reward)
            total_reward.append(self.total_reward)
            print('Average total reward is:', sum(total_reward)/len(total_reward))
            print('Non-zero rewards is:', len(np.nonzero(total_reward)[0])/len(total_reward)*100, "%")
            total_score.append(score) 

        #------------------------- Training ---------------------------#
                
            if it % self.args_train.train_freq == self.args_train.train_freq - 1:
                num_trainsteps = self.process_dataset(it)
                self.save_buffer(it)
                # self.diffusion_trainer.train(num_trainsteps, online=True)
                # self.test(it)

            ## write results to json file at `args.savepath`
            # logger.finish(t, score, self.total_reward, terminal, self.diffusion_experiment, self.value_experiment)
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
                    action, samples = self.policy(cond, it, batch_size=self.args.batch_size, verbose=self.args.verbose)
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
                    action, samples = self.policy(cond, it, batch_size=self.args.batch_size, verbose=self.args.verbose)
                sequence = samples.observations[0]
                try:
                    next_waypoint = sequence[t-t//self.horizon*self.horizon+1].copy()
                except:
                    next_waypoint = sequence[t-t//self.horizon*self.horizon].copy()
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, reward, terminal, _ = self.env.step(action)
            self.total_reward += reward
            score = self.env.get_normalized_score(self.total_reward)
            
            ###

            # print(
            #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            #     f'{action}'
            # )
            if 'maze2d' in args.dataset:
                xy = next_observation[:2]
                goal = self.env.unwrapped._target
                print(
                    f'maze | pos: {xy} | goal: {goal}'
                )
            self.rollout.append(next_observation.copy())
        
            if t % 299 == 0:
                fullpath = join(args.savepath, '{}_{}_test.png'.format(it,t//self.horizon))
                renderer.composite(fullpath, samples.observations, ncol=1)

            if t % args.vis_freq == 0 or terminal:


                # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                ## save rollout thus far
                renderer.composite(join(args.savepath, '{}_test_rollout.png'.format(it)), np.array(self.rollout)[None], ncol=1)

                # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)
            if terminal:
                break

            self.observation = next_observation
            print(t)
        
            ## render every `args.vis_freq` steps
    
        print(self.total_reward)
        # np.savetxt('TestReward_{}_round'.format(it),self.total_reward)

    def save_buffer(self, it):
        buffer = np.concatenate((self.diffusion_trainer.dataset.fields['actions'],self.diffusion_trainer.dataset.fields['observations']),axis=2)
        np.save('buffer_debugging_{}'.format(it), buffer)

    def format_episode(self,actions,next_obs,obs,rew,terminals):
        """Turn generated samples into episode format."""

        episode = {}
        episode_length = len(actions)
        episode['actions'] = np.array(actions).reshape((episode_length,-1))
        episode['next_observations'] = np.array(next_obs).reshape((episode_length,-1))
        episode['observations'] = np.array(obs).reshape((episode_length,-1))
        episode['rewards'] = np.array(rew).reshape((episode_length,-1))
        episode['terminals'] = np.array(terminals).reshape((episode_length,-1))
        # episode['infos/action_log_probs'] = np.array(action_log_probs).reshape((self.max_path_length,-1))
        # episode['infos/qpos'] = np.array(qposs).reshape((self.max_path_length,-1))
        # episode['infos/qvel'] = np.array(qvels).reshape((self.max_path_length,-1))
        # episode['timeouts'] = np.array(timeouts).reshape((self.max_path_length,-1))
        return episode


    def add_to_fields(self, episode):
        """Update the field with newly-generated samples."""

        self.buffer.add_path(episode, online=True)
        # self.diffusion_trainer.dataset.fields.finalize()

        print(self.buffer)

    def process_dataset(self, it):
        """Normalize and preprocess the training data."""

        buffer_size = self.buffer['actions'].shape[0]
        # self.augment_buffer(buffer_size//4)
        buffer_size = self.buffer['actions'].shape[0]

        dataset = self.diffusion_trainer.dataset
        train_dataset = copy.deepcopy(dataset)
        train_dataset.fields = copy.deepcopy(self.buffer)
        train_dataset.fields.finalize()
        buffer_size = train_dataset.fields['actions'].shape[0]
        train_dataset.normalizer = DatasetNormalizer(train_dataset.fields, normalizer=self.normalizer, \
                                                path_lengths=train_dataset.fields['path_lengths'])
        train_dataset.n_episodes = train_dataset.fields.n_episodes
        train_dataset.path_lengths = train_dataset.fields.path_lengths
        train_dataset.normalize()

        # self._sample_data(train_dataset.fields, max(1000,buffer_size//2))    
        train_dataset.indices = train_dataset.make_indices(train_dataset.fields.path_lengths, train_dataset.horizon)

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

        self.diffusion_trainer.dataset = train_dataset
        self.diffusion_trainer.dataloader = cycle(torch.utils.data.DataLoader(
            train_dataset, batch_size=32, \
            num_workers=1, shuffle=True, pin_memory=True
        ))  #### should be train_dataset
        self.diffusion_trainer.dataloader_vis = cycle(torch.utils.data.DataLoader(
                train_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
            ))
        self.policy.normalizer = train_dataset.normalizer


        num_trainsteps = min(buffer_size * 2,8000)
        return num_trainsteps

    def _sample_data(self, fields, sample_size):
        """Sample training data from fields."""

        _dict = fields._dict
        obs = _dict['observations']
        obs_dim = obs.shape[0]
        index = np.zeros(obs_dim,dtype=np.int32)
        for i in range(obs_dim):
            index[i] = obs[i,:,0].nonzero()[0][-1]
        traj_len = np.array([np.linalg.norm(obs[i][index[i]][:2] - obs[i][0][:2]) for i in range(obs_dim)])
        p_sample = traj_len / np.sum(traj_len)
        # traj_rewards = np.array([np.exp(_dict['rewards'][i].sum()) for i in range(_dict['rewards'].shape[0])])
        # p_sample = traj_rewards / np.sum(traj_rewards)
        sample_index = np.random.choice(obs_dim,size=sample_size,replace=True,p=p_sample)
        for key in _dict.keys():
            _dict[key] = _dict[key][sample_index]
        fields._add_attributes()

    def  _sample_data_new(self, fields, sample_size):
        """Sample training data from fields."""

        _dict = fields._dict
        # traj_rewards = np.array([np.exp(_dict['rewards'][i].sum()) for i in range(_dict['rewards'].shape[0])])
        # p_sample = traj_rewards / np.sum(traj_rewards)
        # sample_index = np.random.choice(_dict['actions'].shape[0],size=sample_size,replace=True,p=p_sample)
        # for key in _dict.keys():
        #     _dict[key] = _dict[key][sample_index]
        for key in _dict.keys():
            _dict[key] = _dict[key][-10:]
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
    
    dataset_config_d = utils.Config(
        args_d.loader,
        savepath=(args_d.savepath, 'dataset_config_d.pkl'),
        env=args_d.dataset,
        horizon=args_d.horizon,
        normalizer=args_d.normalizer,
        preprocess_fns=args_d.preprocess_fns,
        use_padding=args_d.use_padding,
        max_path_length=args_d.max_path_length,
        online=True,
    )
    render_config_d = utils.Config(
        args_d.renderer,
        savepath=(args_d.savepath, 'render_config_d.pkl'),
        env=args_d.dataset,
    )
    dataset_d = dataset_config_d()
    renderer_d = dataset_d.env.render
    if dataset_d.env.name == 'maze2d-large-v1':
        observation_dim = 4
        action_dim = 2
        dataset_d.observation_dim = 4
        dataset_d.action_dim = 2
        
    model_config_d = utils.Config(
        args_d.model,
        savepath=(args_d.savepath, 'model_config_d.pkl'),
        horizon=args_d.horizon,
        transition_dim=observation_dim + action_dim,
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
        ddim = True,
        ddim_timesteps = args_train.ddim_timesteps,
        n_timesteps=args_d.n_diffusion_steps,
        loss_type=args_d.loss_type,
        clip_denoised=args_d.clip_denoised,
        predict_epsilon=args_d.predict_epsilon,
        ## loss weighting
        action_weight=args_d.action_weight,
        loss_weights=args_d.loss_weights,
        loss_discount=args_d.loss_discount,
        device=args_d.device,
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
        online=True
    )
    model_d = model_config_d()
    diffusion_d = diffusion_config_d(model_d)
    trainer_d = trainer_config_d(diffusion_d, dataset_d, renderer_d, args.device)
    trainer_d.load(200000)

    #-----------------------------------------------------------------------------#
    #------------------------------- initialize value ----------------------------#
    #-----------------------------------------------------------------------------#
    dataset_config_v = utils.Config(
        args_v.loader,
        savepath=(args_v.savepath, 'dataset_config_v.pkl'),
        env=args_v.dataset,
        horizon=args_v.horizon,
        normalizer=args_v.normalizer,
        preprocess_fns=args_v.preprocess_fns,
        use_padding=args_train.use_padding,
        max_path_length=args_d.max_path_length,
        ## value-specific kwargs
        discount=args_train.discount,
        termination_penalty=args_train.termination_penalty,
        online=True,
        # normed=args_train.normed,
    )
    render_config_v = utils.Config(
        args_v.renderer,
        savepath=(args_v.savepath, 'render_config_v.pkl'),
        env=args_v.dataset,
    )
    dataset_v = dataset_config_v()
    renderer_v = render_config_v()

    model_config_v = utils.Config(
        args_v.model,
        savepath=(args_v.savepath, 'model_config_v.pkl'),
        horizon=args_v.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim_mults=args_v.dim_mults,
        device=args_v.device,
    )
    diffusion_config_v = utils.Config(
        args_v.diffusion,
        savepath=(args_v.savepath, 'diffusion_config_v.pkl'),
        horizon=args_v.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        ddim = True,
        ddim_timesteps = args_train.ddim_timesteps,
        n_timesteps=args_v.n_diffusion_steps,
        loss_type=args_v.loss_type,
        device=args_v.device,
    )
    trainer_config_v = utils.Config(
        utils.Trainer,
        device=args.device,
        savepath=(args_v.savepath, 'trainer_config_v.pkl'),
        train_batch_size=args_v.batch_size,
        train_lr=args_v.learning_rate,
        gradient_accumulate_every=args_v.gradient_accumulate_every,
        ema_decay=args_v.ema_decay,
        sample_freq=args_v.sample_freq,
        save_freq=args_v.save_freq,
        label_freq=int(args_v.n_train_steps // args_v.n_saves),
        save_parallel=args_v.save_parallel,
        results_folder=args_v.savepath,
        bucket=args_v.bucket,
        n_reference=args_v.n_reference,
        n_samples = args_v.n_samples,
        online=True
    )
    model_v = model_config_v()
    diffusion_v = diffusion_config_v(model_v)
    trainer_v = trainer_config_v(diffusion_v, dataset_v, renderer_v, args.device)

    ## initialize policy arguments
    diffusion = trainer_d.ema_model
    dataset = dataset_d
    renderer = renderer_d
    value_function = trainer_v.ema_model
    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
    guide = guide_config()

    ## policies are wrappers around an unconditional diffusion model and a value guide
    if args_train.ddim == True:
        sample = sampling.n_step_guided_ddim_sample
    else:
        sample = sampling.n_step_guided_p_sample

    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
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

    # _online_trainer = online_trainer(policy, args, args_train, \
    #                trainer_d, trainer_v)
    # _online_trainer.train()
    


    ### for drawing density picture
    obs = np.load("/home/lihan/diffuser-maze2d/buffer_debugging_9999.npy")[:,:,2:4]
    # renderer.composite('buffer.png', obs[32:64,:,:], ncol=8)
    # gs = 50
    # grid=np.zeros((gs,gs))
    # for state in obs[...,:]:
    #     if state[0]==0.0 and state[1]==0.0:
    #         continue
    #     try:
    #         grid[int(np.floor(state[0]*gs/10)),int(np.floor(state[1]*gs/10))] += 1
    #     except:
    #         pass
    # import matplotlib.pyplot as plt
    # np.savetxt('buffer.txt',grid)
    # plt.pcolormesh(grid)
    # plt.savefig('grid.jpg')