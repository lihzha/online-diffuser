import copy
import numpy as np
import os
# from diffuser.utils.sampler import WeightedRandomSampler
# from torch.utils.data import WeightedRandomSampler

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
        assert self.max_path_length >= self.traj_len, 'Wrong traj_len!'
        self.horizon = dataset_state.horizon
        self.trainer.diffusion_model.sample_kwargs = self.policy.sample_kwargs
        # a = np.load('755_280.npy')
        # a = np.load('/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/5_29_only_hardcoded_3000_mid_value_sampling_gtdensity/traj/buffer_vis_traj.npy')
        # e = self.format_episode(None,np.zeros((200,4)),[a],np.zeros((200,1)),np.zeros((200,1)))
        # self.buffer.add_path(e)
        # a = np.load('try2.npy')
        # e = self.format_episode(None,np.zeros((200,4)),[a],np.zeros((200,1)),np.zeros((200,1)))
        # self.buffer.add_path(e)
        # a = a.reshape((100,200,4))[:34]
        # for i in range(a.shape[0]):
        #     e = self.format_episode(None,np.zeros((280,4)),[a[i]],np.zeros((280)),np.zeros((280)))
        #     self.buffer.add_path(e)
        # _ = self.process_dataset(self.dataset)
        # self.save_buffer(self.trainer.logdir)
        # self.trainer.train(int(10e6))     
        # cond_targ = np.zeros(self.dataset.observation_dim)
        # cond_targ[:2] = [3,1]
        # cond = {
        #     280-1: cond_targ
        # }
        # cond[0] = np.array([6,1.8,0,0]).repeat(10,0)
        # cond = {}
        # cond[0] = a[2,0][None].repeat(10,0)
        # cond[280*2-1] = a[0,-1][None].repeat(10,0)
        # samples = self.policy(cond)
        # samples = samples.observations
        # self.trainer.renderer.composite('a.png',samples,ncol=5)

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
            if total_reward != 0 or it % 10 == 0:
                target = self.sample_target(5)
                total_reward = 0
            else:
                pass
            self.env.set_target(target)
            # cond_pos = np.random.randint(self.traj_len//2, self.traj_len-1)
            cond_targ = np.zeros(self.dataset.observation_dim)
            cond_targ[:2] = self.env.get_target()
            cond = {
                self.traj_len-1: cond_targ
            }

            # if it > 2000:
            #     energy = self.dataset.fields['energy']
            #     idx = np.random.choice(energy.shape[0], 1, p=energy/energy.sum()).item()
            #     epi = self.dataset.__getitem__(idx).trajectories
            #     epi_last = epi[:,0].nonzero()[0][-1]
            #     epi = epi[:epi_last+1]
            #     epi = self.dataset.normalizer.unnormalize(epi, 'observations')
            #     if epi_last > 150:
            #         cond[range(125,275)] = epi[:150]
            #     else:
            #         cond[range(100,100+epi_last+1)] = epi

            # epi = self.buffer['observations'][58]
            # epi_last = epi[:,0].nonzero()[0][-1]
            # # epi = self.dataset.normalizer.unnormalize(epi, 'observations')
            # if epi_last > 150:
            #     cond[range(125,275)] = epi[:150]
            # else:
            #     cond[range(100,100+epi_last+1)] = epi
            
            for t in range(self.max_path_length):
                if it <= 10:
                    state = self.env.state_vector().copy()
                    action = cond_targ[:2] - state[:2] + (0 - state[2:])
                else:
                    if t % self.traj_len == 0:
                        cond[0] = self.env.state_vector().copy()
                        # target = self.sample_target(5)
                        # cond_targ = np.zeros(self.dataset.observation_dim)
                        # cond[self.traj_len-1] = cond_targ    
                        cnt = 0
                        samples = self.policy(cond)
                        obs_tmp = samples.observations
                        obs_tmp = obs_tmp[:,:,4:]
                        # self.trainer.renderer.composite('b.png',obs_tmp,ncol=1)
                        if obs_tmp.shape[0] == 1:
                            obs_tmp = obs_tmp.squeeze()
                        elif obs_tmp.shape[0] == 4:
                            obs_tmp = obs_tmp.reshape((-1, 4))
                    # design a simple controller based on observations
                    state = self.env.state_vector().copy()
                    # if t < self.traj_len:
                    # action = obs_tmp[t,:2] - state[:2] + (obs_tmp[t,2:] - state[2:])
                    # else:
                    # action = obs_tmp[-1,:2] - state[:2] + (0 - state[2:])
                    # action = obs_tmp[cnt,-1,:2] - state[:2] + (obs_tmp[cnt,-1,2:] - state[2:])
                    action = obs_tmp[cnt,:2] - state[:2] + (obs_tmp[cnt,2:] - state[2:])
                    cnt += 1
                next_observation, reward, terminated, info = self.env.step(action)
                
                # cv2.imwrite('trial_rendering.png',self.env.render())
                if np.linalg.norm((next_observation - observation)[:2]) < 1e-3:
                    break

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
                    xy = next_observation[:2]
                    dist = np.linalg.norm(xy-cond_targ[:2])
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

            if len(obs) >= 100:
                if self.predict_type == 'joint':
                    episode = self.format_episode(actions, next_obs, obs, rew, terminals)
                elif self.predict_type == 'obs_only':
                    episode = self.format_episode(None, next_obs, obs, rew, terminals)
                elif self.predict_type == 'action_only':
                    episode = self.format_episode(actions, None, None, rew, terminals)     
                self.add_to_buffer(episode)     

                # self.trainer.renderer.composite('b.png',np.array(episode['observations'])[None],ncol=1)
            
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
        episode_length = len(rew)
        if episode_length % 20 != 0:
            episode_real_len = int((episode_length // 20 + 1) * 20)
        else:
            episode_real_len = int((episode_length // 20) * 20)
        if actions != None:
            episode['actions'] = np.array(actions).reshape((episode_length,-1))
        if obs != None:
            episode['next_observations'] = np.zeros((episode_real_len, self.dataset.observation_dim), dtype=np.float32)
            episode['next_observations'][:episode_length] = np.array(next_obs).reshape((episode_length,-1))
            episode['next_observations'][episode_length:] = episode['next_observations'][episode_length-1]
            episode['observations'] = np.zeros((episode_real_len, self.dataset.observation_dim), dtype=np.float32)
            episode['observations'][:episode_length] = np.array(obs).reshape((episode_length,-1))
            episode['observations'][episode_length:] = episode['observations'][episode_length-1]
        episode['rewards'] = np.zeros((episode_real_len,1))
        episode['rewards'][:episode_length,0] = np.array(rew)
        episode['rewards'][episode_length:,0] = np.array(rew)[-1]
        episode['terminals'] = np.zeros((episode_real_len,1))
        episode['terminals'][:episode_length,0] = np.array(terminals)
        episode['terminals'][episode_length:,0] = np.array(terminals)[-1]

        return episode

    def add_to_buffer(self, episode):
        """Update the field with newly-generated samples."""

        self.buffer.add_path(episode)
        print(self.buffer)

    def process_dataset(self, dataset):
        """Normalize and preprocess the training data."""
        
        dataset.set_fields(self.buffer)
        self.policy.normalizer = dataset.normalizer

        if dataset == self.dataset:
            # dataset.indices = dataset.make_new_indices(dataset.fields['path_lengths'], self.traj_len)
            # dataset.indices = dataset.make_uneven_indices(dataset.fields['path_lengths'])
            # normed_obs = dataset.get_all_item()
            # energy = self.compute_energy('both',normed_obs)
            # sampler = WeightedRandomSampler(weights=energy, num_samples=self.trainer.batch_size, replacement=False)
            # self.trainer.create_dataloader(sampler=sampler)
            dataset.indices = dataset.make_fix_indices()
            self.trainer.create_dataloader()
            # self.trainer.render_buffer(10, dataset.fields['observations'])
        elif dataset == self.dataset_state:
            dataset.indices = dataset.make_indices(dataset.fields['path_lengths'], self.horizon)
            # normed_obs = dataset.get_all_item()
            # energy = self.compute_energy('state', normed_obs)
            # sampler = WeightedRandomSampler(weights=energy, num_samples=self.trainer.batch_size)
            self.trainer_state.create_dataloader(sampler=None)
            self.trainer_state.render_buffer(10, dataset.fields['observations'])
        # num_trainsteps = min(sample_size * 4, 4000)
        # num_trainsteps = dataset.fields['normed_observations'].shape[0] * 5
        num_trainsteps = 3000
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

