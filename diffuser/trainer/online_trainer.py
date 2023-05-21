import copy
import numpy as np

class OnlineTrainer:
    def __init__(self, ebm_model, trainer, env, dataset, policy, predict_type):

        self.model = ebm_model
        self.trainer = trainer
        self.env = env
        self.dataset = dataset
        self.policy = policy
        self.predict_type = predict_type
        self.buffer = copy.deepcopy(self.dataset.fields)

        self.total_reward = []
        self.total_score = []

        # slightly abuse abstraction for convenience

        self.max_path_length = dataset.max_path_length
        self.traj_len = dataset.traj_len
        self.device = self.trainer.device
        assert self.max_path_length >= self.traj_len, 'Wrong traj_len!'
        self.horizon = dataset.horizon

    def train(self, train_freq, iterations):
        """Online training scenerio."""

        for it in range(iterations):

            total_reward = 0
            episode = {}
            if self.predict_type == 'joint':
                actions, next_obs, obs, rew, terminals = [], [], [], [], []
            elif self.predict_type == 'obs_only':
                next_obs, obs, rew, terminals = [], [], [], []
            elif self.predict_type == 'action_only':
                actions, rew, terminals = [], [], [], []
            obs_info = self.env.reset()
            observation = obs_info[0]['observation']

            # target = self.sample_target()
            t1=np.random.binomial(1,0.5)
            if t1==0:
                target = obs_info[0]['desired_goal'][:3]
            elif t1==2:
                target = obs_info[0]['desired_goal'][3:]
            elif t1==1:
                target = obs_info[0]['achieved_goal'][:3]
            elif t1==3:
                target = obs_info[0]['achieved_goal'][3:]

            cond_targ = np.zeros(self.dataset.observation_dim)
            cond_targ[:target.shape[0]] = target
            # TODO: change cond according to target
            cond = {
                self.traj_len - 1: cond_targ
            }

            for t in range(self.max_path_length):
                
                if t % self.traj_len == 0:
                    cond[0] = observation
                    cnt = 0
                    samples = self.policy(cond)
                    obs_tmp = samples.observations
                # design a simple controller based on observations
                # action = np.concatenate((obs_tmp[cnt, 1, :self.dataset.observation_dim-1] - obs_tmp[cnt, 0, :self.dataset.observation_dim-1],obs_tmp[cnt, 1, self.dataset.observation_dim-1:]))
                action = obs_tmp[cnt, 1, :self.dataset.observation_dim] - obs_tmp[cnt, 0, :self.dataset.observation_dim] 
                cnt += 1
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                next_observation = next_observation['observation']
                
                # cv2.imwrite('trial_rendering.png',self.env.render())
                # if np.linalg.norm((next_observation - self.observation)[:3]) < 1e-3 and it!=0:
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
                    dist = np.linalg.norm(xy-target)
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

                observation = next_observation
                print(t)

            if len(obs) >= self.horizon:
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
                num_trainsteps = self.process_dataset()
                self.save_buffer()
                self.trainer.train(num_trainsteps)

        print(self.total_reward)

    def test(self, it):

        open_loop = False
        total_reward = 0
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
        
        buffer = self.dataset.fields['observations']
        np.save('buffer_vis.npy', buffer)

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

    def process_dataset(self):
        """Normalize and preprocess the training data."""

        self.dataset.set_fields(self.buffer)
        self.policy.normalizer = self.dataset.normalizer
        obs_energy = self.compute_buffer_energy(self.dataset)
        sample_size = obs_energy.shape[0]//4
        self.energy_sampling(self.dataset.fields, sample_size, obs_energy) 
        self.trainer.create_dataloader()
        num_trainsteps = min(sample_size * 4, 4000)
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
        sample_index = np.random.choice(batchsize,size=sample_size,replace=False, p=energy/energy.sum())
        for key in _dict.keys():
            _dict[key] = _dict[key][sample_index]
        fields._add_attributes() 

    def compute_buffer_energy(self, dataset):
        """Randomly sample targets according to energy."""

        raw_obs = dataset.fields['observations']
        
        for i in range(raw_obs.shape[0]):
            obs_pair = np.zeros((raw_obs.shape[1]-1,2,raw_obs.shape[-1]))
            obs_pair[:,0,:] = raw_obs[i,:raw_obs.shape[1]-1]
            obs_pair[:,1,:] = raw_obs[i,1:raw_obs.shape[1]]
            if i == 0:
                obs = obs_pair
            else:
                obs = np.concatenate((obs,obs_pair), axis=0)

        energy = self.model.get_buffer_energy(obs, self.device)
        energy = energy.reshape((raw_obs.shape[0],-1)).mean(-1)
        energy = energy.detach().cpu().numpy()
        return energy

    def sample_target(self):

        # sample_idx = np.random.choice(np.prod(self.obs_energy.shape), size=1, p=self.obs_energy.flatten()/self.obs_energy.sum())[0]
        # target = self.target_set[sample_idx//self.target_set.shape[1], sample_idx-sample_idx//self.target_set.shape[1]*self.target_set.shape[1]][:2]
        
        target = self.env.unwrapped.task.sample_targets(self.ebm_model, self.robot, 100, self.args_train.device)
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
        
        # self.obs_energy[torch.div(target_ind,self.target_set.shape[1],rounding_mode='trunc'), target_ind-torch.div(target_ind,self.target_set.shape[1],rounding_mode='trunc')*self.target_set.shape[1]] = 0
        # sample_index = np.random.choice(a=self.state_dist_flat.size, p=self.state_dist_flat/self.state_dist_flat.sum())
        # adjusted_index = np.unravel_index(sample_index, self.neg_state_dist.shape)
        # target = np.array(adjusted_index) / 100


        return target