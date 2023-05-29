import copy
import numpy as np
import os
import cv2

class OnlineTrainer:
    def __init__(self, state_model, trajectory_model, trainer_traj, trainer_state, env, dataset_traj, dataset_state, policy, predict_type):

        self.model = state_model
        self.trajectory_model = trajectory_model
        self.trainer = trainer_traj
        self.trainer_state = trainer_state
        self.env = env
        self.dataset = dataset_traj
        self.dataset_state = dataset_state
        self.policy = policy
        self.predict_type = predict_type
        self.buffer = copy.deepcopy(self.dataset.fields)
        self.total_reward = []
        self.total_score = []

        # slightly abuse abstraction for convenience

        self.max_path_length = dataset_traj.max_path_length
        self.traj_len = dataset_traj.horizon
        self.device = self.trainer.device
        self.env.device = self.device
        assert self.max_path_length >= self.traj_len, 'Wrong traj_len!'
        self.horizon = dataset_state.horizon

        # self.trainer.diffusion_model.sample_kwargs = self.policy.sample_kwargs
        # a = np.load('try.npy')
        # a = a.reshape((100,200,4))[:34]
        # for i in range(34):
        #     e = self.format_episode(None,np.zeros((200,4)),[a[i]],np.zeros((200,1)),np.zeros((200,1)))
        #     self.buffer.add_path(e)
        # num_trainsteps_traj = self.process_dataset(self.dataset)
        # self.save_buffer(self.trainer.logdir)
        # self.trainer.train(num_trainsteps_traj)     

    def train(self, train_freq, iterations):
        """Online training scenerio."""
        total_reward = 0
        for it in range(iterations):
            self.policy.diffusion_model.cnt = it
            episode = {}
            if self.predict_type == 'joint':
                actions, next_obs, obs, rew, terminals = [], [], [], [], []
            elif self.predict_type == 'obs_only':
                next_obs, obs, rew, terminals = [], [], [], []
            elif self.predict_type == 'action_only':
                actions, rew, terminals = [], [], [], []

            obs_info = self.env.reset(density=self.model, batch_size=10, device=self.device)
            # obs_info = self.env.reset()
            observation = obs_info[0]['observation']
            target_obj = obs_info[0]['achieved_goal']
            target_pos = obs_info[0]['desired_goal']
            if it % 2 == 0:
                target = target_obj
            else:
                target = target_pos
            # self.env.set_target(target)
            # cond_pos = np.random.randint(self.traj_len//2, self.traj_len-1)
            cond_targ = np.zeros(self.dataset.observation_dim)
            cond_targ[:self.dataset.observation_dim] = target
            cond = {
                self.traj_len-1: cond_targ
            }

            for t in range(self.max_path_length):
                if it <= 4000:
                    state = self.env.unwrapped.robot.get_obs().copy()
                    action = 5 * (cond[self.traj_len-1] - state[:3])
                else:
                    if t == 0:
                        cond[0] = self.env.unwrapped.robot.get_obs().copy()
                        # cnt = 0
                        samples = self.policy(cond)
                        obs_tmp = samples.observations
                        if obs_tmp.shape[0] == 1:
                            obs_tmp = obs_tmp.squeeze()
                        elif obs_tmp.shape[0] == 4:
                            obs_tmp = obs_tmp.reshape((-1, 4))
                    # design a simple controller based on observations
                    state = self.env.unwrapped.robot.get_obs().copy()
                    if t < self.traj_len-1:
                        action = 10 * (obs_tmp[t+1,:3] - obs_tmp[t,:3])
                    else:
                        action = (obs_tmp[-1,:3] - state[:3])
                # action = obs_tmp[cnt,-1,:2] - state[:2] + (obs_tmp[cnt,-1,2:] - state[2:])
                # action = obs_tmp[cnt,:2] - state[:2] + (obs_tmp[cnt,2:] - state[2:])
                # cnt += 1
                next_obs_info, reward, terminated, _, info = self.env.step(action)
                next_observation = next_obs_info['observation']
                
                # if t % 1 == 0:
                #     cv2.imwrite('trial_rendering.png',self.env.render())
                # if np.linalg.norm((next_observation - observation)[:3]) < 1e-5 and it!=0:
                #     break

                total_reward += reward
                # score = self.env.get_normalized_score(self.total_reward)
                # self.rollout.append(next_observation.copy())
                if self.env.__repr__() == 'maze2d':
                    xy = next_observation[:2]
                    goal = self.env.unwrapped._target
                    print(
                        f'it: {it} | maze | pos: {xy} | goal: {goal}'
                    )
                else:
                    xy = next_observation[:3]
                    dist = np.linalg.norm(xy-cond_targ[:3])
                    print(
                        f'it: {it} | panda | dist: {dist}'
                    )

                if self.predict_type == 'joint':
                    next_obs.append(next_observation)
                    obs.append(observation)
                    actions.append(action)
                elif self.predict_type == 'obs_only':
                    next_obs.append(next_observation)
                    obs.append(observation)
                elif self.predict_type == 'action_only':
                    actions.append(action)               
                rew.append(reward)
                terminals.append(terminated)

                if terminated:
                    break

                observation = next_observation.copy()
                print(t)

            if len(obs) != 0:
                if self.predict_type == 'joint':
                    episode = self.format_episode(actions, next_obs, obs, rew, terminals)
                elif self.predict_type == 'obs_only':
                    episode = self.format_episode(None, next_obs, obs, rew, terminals)
                elif self.predict_type == 'action_only':
                    episode = self.format_episode(actions, None, None, rew, terminals)     
                self.add_to_buffer(episode)     

            print("----------------",it,"round finished------------------")
            print("total_reward", total_reward)
            self.total_reward.append(total_reward)
            print('Average total reward is:', sum(self.total_reward)/len(self.total_reward))
            print('Non-zero rewards is:', len(np.nonzero(self.total_reward)[0])/len(self.total_reward)*100, "%")
            # self.total_score.append(score) 

            if it > 0 and it % train_freq == 0: 
                num_trainsteps_traj = self.process_dataset(self.dataset)
                self.save_buffer(self.trainer.logdir)
                self.trainer.train(num_trainsteps_traj)     
                num_trainsteps_state = self.process_dataset(self.dataset_state)
                self.trainer_state.train(num_trainsteps_state//2)            
    
        print(self.total_reward)

    def test(self, epoch):
        score_list = []
        total_reward_list = []
        for i in range(epoch):
            total_reward = 0
            observation = self.env.reset()
            # target = self.sample_target(10)
            cond_targ = np.zeros(self.dataset.observation_dim)
            # self.env.set_target(target)
            cond_targ[:2] = self.env._target
            cond = {
                self.traj_len - 1: cond_targ
            }
            rollout = [observation.copy()]
            fake_rollout = 0
            for t in range(self.max_path_length):
                
                if t % 20 == 0:
                    cond[0] = self.env.state_vector().copy()
                    cnt = 0
                    samples = self.policy(cond,batch_size=10)
                    obs_tmp = samples.observations
                    if not isinstance(fake_rollout, np.ndarray):
                        fake_rollout = obs_tmp
                    else:
                        fake_rollout = np.concatenate((fake_rollout, obs_tmp),axis=1)
                # design a simple controller based on observations
                state = self.env.state_vector().copy()
                action = obs_tmp[0,cnt,:2] - state[:2] + (obs_tmp[0,cnt,2:] - state[2:])
                cnt += 1
                next_observation, reward, terminated, info = self.env.step(action)

                total_reward += reward
                rollout.append(next_observation.copy())
                if self.env.__repr__() == 'maze2d':
                    xy = next_observation[:2]
                    goal = self.env.unwrapped._target
                    print(
                        f'it: {t} | maze | pos: {xy} | goal: {goal}'
                    )
                else:
                    xy = next_observation[:2]
                    dist = np.linalg.norm(xy-cond_targ[:2])
                    print(
                        f'it: {t} | panda | dist: {dist}'
                    )

                if terminated:
                    break

                observation = next_observation
                print(t)
            score = self.env.get_normalized_score(total_reward)
            score_list.append(score)
            total_reward_list.append(total_reward)
        rollout = np.array(rollout)[None]
        savepath = os.path.join(self.trainer.logdir, f'rollout_{i}.png')
        self.trainer.renderer.composite(savepath,rollout,ncol=1)
        savepath = os.path.join(self.trainer.logdir, f'fake_rollout_{i}.png')
        self.trainer.renderer.composite(savepath,fake_rollout,ncol=5)
        score_array = np.array(score_list)
        total_reward_array = np.array(total_reward_list)
        print('score_list:', score_list)
        print('total_reward_list:', total_reward_list)
        print('score mean and std:', score_array.mean(), score_array.std())
        print('reward mean and std:', total_reward_array.mean(), total_reward_array.std())

    def save_buffer(self, path):
        
        buffer = self.dataset.fields['observations']
        np.save(path+'/buffer_vis_traj.npy', buffer)

    def format_episode(self,actions,next_obs,obs,rew,terminals):
        """Turn generated samples into episode format."""

        episode = {}
        episode_length = len(rew)
        if actions != None:
            episode['actions'] = np.array(actions).reshape((episode_length,-1))
        if obs != None:
            episode['next_observations'] = np.array(next_obs).reshape((episode_length,-1))
            episode['observations'] = np.array(obs).reshape((episode_length,-1))
        episode['rewards'] = np.array(rew).reshape((episode_length,-1))
        episode['terminals'] = np.array(terminals).reshape((episode_length,-1))

        return episode

    def add_to_buffer(self, episode):
        """Update the field with newly-generated samples."""

        self.buffer.add_path(episode)
        print(self.buffer)

    def process_dataset(self, dataset):
        """Normalize and preprocess the training data."""

        dataset.set_fields(self.buffer)
        self.policy.normalizer = dataset.normalizer


        # obs_energy = self.compute_buffer_energy(dataset)
        # buffer_size = dataset.fields['observations'].shape[0]
        # if buffer_size <=100:
        #     sample_size = buffer_size
        #     num_trainsteps = 1000
        # else:
        #     sample_size = buffer_size // 4
        #     self.energy_sampling(dataset.fields, sample_size, obs_energy) 
        #     num_trainsteps = sample_size * 10
        if self.policy.diffusion_model.cnt <= 1000:
            num_trainsteps = 5000
        else:
            obs_energy = self.compute_buffer_energy(dataset)
            buffer_size = dataset.fields['observations'].shape[0]
            sample_size = buffer_size // 4
            self.energy_sampling(dataset.fields, sample_size, obs_energy)
            print(f'Getting {sample_size} trajectories for training!!!!')
            num_trainsteps = sample_size * 10


        if dataset == self.dataset:
            dataset.indices = dataset.make_indices(dataset.fields['path_lengths'], self.traj_len)
            if dataset.fields['observations'].shape[0] == 34:
                self.trainer.create_dataloader(1)
            else:
                self.trainer.create_dataloader()
            self.trainer.render_buffer(10, dataset.fields['observations'])
        elif dataset == self.dataset_state:
            dataset.indices = dataset.make_indices(dataset.fields['path_lengths'], self.horizon)
            self.trainer_state.create_dataloader()
            self.trainer_state.render_buffer(10, dataset.fields['observations'])
        # num_trainsteps = min(sample_size * 4, 4000)
        return num_trainsteps
    
    def sample_data(self, fields, sample_size):
        """Sample training data from fields."""

        _dict = fields._dict
        obs = _dict['observations']
        obs_dim = obs.shape[0]

        # index = np.zeros(obs_dim,dtype=np.int32)
        # for i in range(obs_dim):
        #     index[i] = obs[i,:,0].nonzero()[0][-1]
        # traj_len = np.array([np.linalg.norm(obs[i][index[i]][:2] - obs[i][0][:2]) for i in range(obs_dim)])
        # p_sample_len = traj_len / np.sum(traj_len)
        # # traj_rewards = np.array([np.exp(_dict['rewards'][i].sum()) for i in range(_dict['rewards'].shape[0])])
        # # p_sample = traj_rewards / np.sum(traj_rewards)

        # if self.args_train.ebm:
        #     if self.args_train.predict_action:
        #         actions = _dict['actions']
        #         trajectories = np.concatenate([actions, obs], axis=-1)
        #     else:
        #         trajectories = obs
        #     trajectories = torch.tensor(trajectories, dtype=torch.float, device=self.args_train.device)
        #     t = torch.full((trajectories.shape[0],), 0, device=self.args_train.device, dtype=torch.long)
        #     energy = self.policy.diffusion_model.model.point_energy(trajectories, None, t).sum(-1)
        #     p_sample_prob = (energy / energy.sum()).detach().cpu().numpy()
        # else:
        #     obs = torch.tensor(obs[:,:,:2], dtype=torch.float, device=self.args_train.device)
        #     raw_density = gt_density(obs).squeeze(-1)
        #     for i in range(obs_dim):
        #         raw_density[i,index[i]+1:] = 0
        #     p_sample_prob = 1 / (raw_density.sum(-1))
        #     p_sample_prob = (p_sample_prob / p_sample_prob.sum()).detach().cpu().numpy()
        
        # p_sample = p_sample_len * p_sample_prob
        # p_sample = p_sample / p_sample.sum()

        # sample_index = np.random.choice(obs_dim,size=sample_size,replace=False,p=p_sample)
        sample_index = np.random.choice(obs_dim,size=sample_size,replace=False)

        for key in _dict.keys():
            _dict[key] = _dict[key][sample_index]
        fields._add_attributes()

    def energy_sampling(self, fields, sample_size, energy):
        """Sample training data from fields."""

        _dict = fields._dict
        obs = _dict['observations']
        batchsize = obs.shape[0]
        sample_index = np.random.choice(batchsize,size=sample_size,replace=True, p=energy/energy.sum())
        for key in _dict.keys():
            _dict[key] = _dict[key][sample_index]
        fields._add_attributes() 

    def compute_buffer_energy(self, dataset):
        """Randomly sample targets according to energy."""

        raw_obs = dataset.fields['observations']
        energy_list = []
        
        for i in range(raw_obs.shape[0]):
            # obs = raw_obs[i][:,None]
            # obs = raw_obs[i][:,None]
            last = raw_obs[i,:,0].nonzero()[0][-1]
            obs = raw_obs[i,:last+1][:,None]
            energy = self.model.get_buffer_energy(obs, self.device).sum(-1)
            energy_list.append(energy.detach().cpu().numpy().item())
        energy_array = np.array(energy_list)
        return energy_array

    def sample_target(self, batch_size):

        # target_array, pair = self.env.sample_target(batch_size)
        # target_state = np.zeros((batch_size, 1, self.dataset.observation_dim))
        # target_state[:,0,:2] = target_array
        # target_pair = np.zeros((batch_size, 2, self.dataset.observation_dim))
        # target_pair[:,0,:2] = target_array
        # target_pair[:,1,:2] = target_array + pair
        # np.array(([ 0.48975977,  0.50192666, -5.2262554 , -5.2262554 ],[ 7.213778 , 10.215629 ,  5.2262554,  5.2262554]),dtype=np.float32)
        val = np.array(([0.55,0.35,0.05],[-0.55,-0.35,0.]),dtype=np.float32)
        target_x = np.random.rand(batch_size) * (0.55 + 0.55) - 0.55
        target_y = np.random.rand(batch_size) * (0.35 + 0.35) - 0.35
        target_z = np.random.rand(batch_size) * (0.05 + 0.)
        target_state = np.zeros((batch_size, 1, self.dataset.observation_dim))
        target_state[:,0,0] = target_x
        target_state[:,0,1] = target_y
        target_state[:,0,2] = target_z
        energy = self.model.get_target_energy(target_state, self.device)
        return target_state[np.argmax(energy)].squeeze()[:2]
    
        # target = self.target_set[sample_idx//self.target_set.shape[1], sample_idx-sample_idx//self.target_set.shape[1]*self.target_set.shape[1]][:2]
        
        # traj_shape = (50, self.horizon, 4)
        # target_traj = utils.arrays.sample_from_array(np.array(([-1,-1,-1,-1],[1,1,1,1]),dtype=np.float32), traj_shape=traj_shape)
        # obs = torch.tensor(target_traj, device=self.args_train.device, dtype=torch.float32)
        # t = torch.zeros(obs.shape[0],device=obs.device)
        # cond = None
        # self.obs_energy = self.diffusion_trainer.ema_model.model.point_energy(obs, cond, t).cpu().detach().numpy()
        # self.target_set = obs.cpu().detach().numpy()
        # self.obs_energy[self.target_set.sum(-1)==0] = 0
        # sample_idx = np.random.choice(np.prod(self.obs_energy.shape), size=1, p=self.obs_energy.flatten()/self.obs_energy.sum())[0]
        # target = self.target_set[sample_idx//self.target_set.shape[1], sample_idx-sample_idx//self.target_set.shape[1]*self.target_set.shape[1]][:2]