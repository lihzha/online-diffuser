import copy
import numpy as np
import os
# from diffuser.utils.sampler import WeightedRandomSampler
# from torch.utils.data import WeightedRandomSampler

class OnlineTrainer:
    def __init__(self, state_model, trajectory_model, trainer_traj, trainer_state, env, dataset_traj,dataset_traj_fake, dataset_state, policy, predict_type,use_fake_buffer):

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
        self.state_buffer = copy.deepcopy(self.dataset.fields)
        self.use_fake_buffer = use_fake_buffer
        if self.use_fake_buffer:
            self.dataset_fake = dataset_traj_fake
            self.buffer_fake = copy.deepcopy(self.dataset_fake.fields)

        self.total_reward = []
        self.total_score = []

        # slightly abuse abstraction for convenience

        self.max_path_length = dataset_traj.max_path_length
        self.traj_len = dataset_traj.horizon
        self.device = self.trainer.device
        assert self.max_path_length >= self.traj_len, 'Wrong traj_len!'
        self.horizon = dataset_state.horizon
        self.trainer.diffusion_model.sample_kwargs = self.policy.sample_kwargs
        ### neglect this
        # a = np.load('755_280.npy')
        # e = self.format_episode(None,np.zeros((200,4)),[a],np.zeros((200,1)),np.zeros((200,1)))
        # self.buffer.add_path(e)


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
            observation = self.env.reset()

            total_reward = 0
            target = self.sample_target(5)
            self.env.set_target(target)
            cond_targ = np.zeros(self.dataset.observation_dim)
            cond_targ[:2] = self.env.get_target()
            cond = {
                self.traj_len-1: cond_targ
            }

            warm_start = 800
            for t in range(self.max_path_length):
                # first collect some good trajectories with hand-crafted controller (cheating for the time being)
                if it < warm_start:
                    state = self.env.state_vector().copy()
                    action = cond_targ[:2] - state[:2] + (0 - state[2:])
                else:
                    # if t % self.traj_len == 0:
                    if t == 0:
                        cond[0] = self.env.state_vector().copy()
                        # target = self.sample_target(5)
                        cond_targ = np.zeros(self.dataset.observation_dim)
                        cond[self.traj_len-1] = cond_targ    
                        cnt = 0
                        samples = self.policy(cond)
                        obs_tmp = samples.observations
                        if self.trainer.diffusion_model.condition_type == 'extend':
                            obs_tmp = obs_tmp[:,:,4:]
                        else:
                            obs_tmp = obs_tmp[:,:,:]
                        assert obs_tmp.shape[0] == 1
                        obs_tmp = obs_tmp.squeeze()
                    # [0,traj_lan] use planning result, [traj_len,max_path] use simple controller
                    state = self.env.state_vector().copy()
                    if t<self.traj_len:
                        action = obs_tmp[cnt,:2] - state[:2] + (obs_tmp[cnt,2:] - state[2:])
                        cnt += 1
                    if t>=self.traj_len:
                        action = cond_targ[:2] - state[:2] + (0 - state[2:])
                next_observation, reward, terminated, info = self.env.step(action)
                
                # cv2.imwrite('trial_rendering.png',self.env.render())
                if np.linalg.norm((next_observation - observation)[:2]) < 1e-3:
                    break

                total_reward += reward
                # score = self.env.get_normalized_score(self.total_reward)
                # self.rollout.append(next_observation.copy())
                # if self.env.__repr__() == 'maze2d':
                #     xy = next_observation[:2]
                #     goal = self.env.unwrapped._target
                #     print(
                #         f'it: {it} | maze | pos: {xy} | goal: {goal}'
                #     )
                # else:
                #     xy = next_observation[:2]
                #     dist = np.linalg.norm(xy-cond_targ[:2])
                #     print(
                #         f'it: {it} | panda | dist: {dist}'
                #     )

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
            
            if len(obs) >= 200:
                if total_reward>0:
                # if len(obs) >= 100:
                    if self.predict_type == 'joint':
                        episode,episode_real_len = self.format_episode(actions, next_obs, np.array(obs), rew, terminals)
                    elif self.predict_type == 'obs_only':
                        episode,episode_real_len = self.format_episode(None, next_obs, np.array(obs), rew, terminals)
                    elif self.predict_type == 'action_only':
                        episode,episode_real_len = self.format_episode(actions, None, None, rew, terminals)     
                    self.add_to_buffer(episode)
                # save fake path lead to zero reward
                if self.use_fake_buffer and it>warm_start and total_reward==0:
                    obs_fake = samples.observations[0,:,4:] if self.trainer.diffusion_model.condition_type == 'extend' else samples.observations[0]
                    episode_fake,_ = self.format_episode(None, obs_fake,obs_fake,rew,terminals)
                    self.add_to_buffer_fake(episode_fake,length=episode_real_len)

                # self.trainer.renderer.composite('b.png',np.array(episode['observations'])[None],ncol=1)
            
            print("----------------",it,"round finished------------------")
            # print("total_reward", total_reward)
            print('save_path_length',t,",total_reward",total_reward)
            self.total_reward.append(total_reward)
            # print('Average total reward is:', sum(self.total_reward)/len(self.total_reward))
            print('Non-zero rewards is:', len(np.nonzero(self.total_reward)[0])/len(self.total_reward)*100, "%")
            # self.total_score.append(score) 

            if it > 0 and it % train_freq == 0: 
                num_trainsteps_traj = self.process_dataset(use_fake_buffer=self.use_fake_buffer)
                # self.save_buffer(self.trainer.logdir)
                self.trainer.train(num_trainsteps_traj,use_fake_buffer=self.use_fake_buffer)     
                # num_trainsteps_state = self.process_dataset(self.dataset_state)
                # self.trainer_state.train(num_trainsteps_state//2)            
    
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
                        f'it: {i} | panda | dist: {dist}'
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
        episode_length = obs.shape[0]
        if episode_length % 20 != 0:
            episode_real_len = int((episode_length // 20 + 1) * 20)
        else:
            episode_real_len = int((episode_length // 20) * 20)
        if actions != None:
            episode['actions'] = np.array(actions).reshape((episode_length,-1))
        # if obs != None:
        episode['next_observations'] = np.zeros((episode_real_len, self.dataset.observation_dim), dtype=np.float32)
        episode['next_observations'][:episode_length] = np.array(next_obs).reshape((episode_length,-1))
        episode['next_observations'][episode_length:] = episode['next_observations'][episode_length-1]
        episode['observations'] = np.zeros((episode_real_len, self.dataset.observation_dim), dtype=np.float32)
        episode['observations'][:episode_length] = np.array(obs).reshape((episode_length,-1))
        episode['observations'][episode_length:] = episode['observations'][episode_length-1]

        episode['rewards'] = np.zeros((episode_real_len,1))
        # episode['rewards'][:episode_length,0] = np.array(rew)
        # episode['rewards'][episode_length:,0] = np.array(rew)[-1]
        episode['terminals'] = np.zeros((episode_real_len,1))
        # episode['terminals'][:episode_length,0] = np.array(terminals)
        # episode['terminals'][episode_length:,0] = np.array(terminals)[-1]

        return episode,episode_real_len

    def add_to_buffer(self, episode):
        """Update the field with newly-generated samples."""

        self.buffer.add_path(episode)
        print(self.buffer)
    
    def add_to_buffer_fake(self, episode, length):
        self.buffer_fake.add_path(episode, length)

    def process_dataset(self, use_fake_buffer):
        """Normalize and preprocess the training data."""
        
        # dataset.set_fields(self.buffer)
        # self.policy.normalizer = dataset.normalizer

        self.dataset.set_fields(self.buffer)
        self.policy.normalizer = self.dataset.normalizer
        self.dataset.indices = self.dataset.make_fix_indices()
        if use_fake_buffer:
            self.dataset_fake.set_fields(self.buffer_fake)
            self.dataset_fake.indices = self.dataset_fake.make_fix_indices()
        self.trainer.create_dataloader(use_fake_buffer=use_fake_buffer)

        # if dataset == self.dataset or dataset == self.dataset_fake:
        #     # dataset.indices = dataset.make_new_indices(dataset.fields['path_lengths'], self.traj_len)
        #     # dataset.indices = dataset.make_uneven_indices(dataset.fields['path_lengths'])
        #     # normed_obs = dataset.get_all_item()
        #     # energy = self.compute_energy('both',normed_obs)
        #     # sampler = WeightedRandomSampler(weights=energy, num_samples=self.trainer.batch_size, replacement=False)
        #     # self.trainer.create_dataloader(sampler=sampler)
        #     dataset.indices = dataset.make_fix_indices()
        #     dataset.indices = dataset.make_fix_indices()
        #     self.trainer.create_dataloader(dataset)
        #     # self.trainer.render_buffer(10, dataset.fields['observations'])
        # elif dataset == self.dataset_state:
        #     dataset.indices = dataset.make_indices(dataset.fields['path_lengths'], self.horizon)
        #     # normed_obs = dataset.get_all_item()
        #     # energy = self.compute_energy('state', normed_obs)
        #     # sampler = WeightedRandomSampler(weights=energy, num_samples=self.trainer.batch_size)
        #     self.trainer_state.create_dataloader(sampler=None)
        #     self.trainer_state.render_buffer(10, dataset.fields['observations'])
        # # num_trainsteps = min(sample_size * 4, 4000)
        # # num_trainsteps = dataset.fields['normed_observations'].shape[0] * 5
        num_trainsteps = 2000
        return num_trainsteps
    
    def sample_target(self, batch_size):

        # np.array(([ 0.48975977,  0.50192666, -5.2262554 , -5.2262554 ],[ 7.213778 , 10.215629 ,  5.2262554,  5.2262554]),dtype=np.float32)
        target_x = np.random.rand(batch_size) * (7.213778 - 0.48975977) + 0.48975977
        target_y = np.random.rand(batch_size) * (10.215629 - 0.50192666) + 0.50192666
        target_state = np.zeros((batch_size, 1, self.dataset.observation_dim))
        target_state[:,0,0] = target_x
        target_state[:,0,1] = target_y
        energy = self.model.get_target_energy(target_state, self.device)
        energy_med = np.median(energy)
        idx = np.where(energy==energy_med)[0]
        return target_state[idx].squeeze()[:2]
    
    def compute_energy(self, typ, normed_obs):

        shape = normed_obs.shape
        state_energy = np.array([])
        bs = 200
        for i in range(1, shape[0]//bs+2):
            input_normed_obs = normed_obs[(i-1)*bs:i*bs].reshape((-1,1,shape[2]))
            state_energy_i = self.model.get_buffer_energy(input_normed_obs, self.device).squeeze().reshape((-1,shape[1])).sum(-1).detach().cpu().numpy()
            state_energy = np.append(state_energy, state_energy_i)
        if typ == 'both':
            traj_energy = np.array([])
            for i in range(1, shape[0]//bs + 2):
                input_normed_obs = normed_obs[(i-1)*bs:i*bs]
                traj_energy_i = self.trajectory_model.get_buffer_energy(input_normed_obs, self.device).squeeze().detach().cpu().numpy()
                traj_energy = np.append(traj_energy, traj_energy_i)
            total_energy = traj_energy + state_energy
        elif typ == 'state':
            total_energy = state_energy
        else:   
            raise NotImplementedError
        self.dataset_state.fields['energy'] = total_energy
        return total_energy

