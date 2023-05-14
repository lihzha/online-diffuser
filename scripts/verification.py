import pdb

import diffuser.utils as utils
import diffuser
# import diffuser.sampling as sampling
from diffuser.datasets.buffer import ReplayBuffer
import torch
from diffuser.datasets.normalization import DatasetNormalizer
import numpy as np
import copy
from os.path import join
import diffuser.models.helpers as helpers
import os
import time

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

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

def cycle(dl):
    while True:
        for data in dl:
            yield data

def unify_args(args_train, args_d, args_v, args):
    args_list = ['device', 'n_diffusion_steps', 'horizon', 'max_path_length']
    for k in args_list:
        v = args_train.__getattribute__(k)
        args_v.__setattr__(k, v)
        args_d .__setattr__(k, v) 
        args.__setattr__(k, v)
    print(args.device)
    print(args_train.device)
    print(args_d.device)
    print(args_v.device)

class online_trainer:

    def __init__(self, policy, args, args_train, trainer_d, trainer_v):
        
        self.policy = policy 

        self.args = args
        self.args_train = args_train

        self.value_trainer = trainer_v
        self.diffusion_trainer = trainer_d

        self.cnt=[]
        self.rew=[]
        self.score = []

        self.init_args()
        # self.refresh_buffer()

        self.env = self.diffusion_trainer.dataset.env
        self.total_reward = 0

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
        data = np.load('/home/lihan/diffuser-maze2d/buffer_for_visualization_1000.npy')
        for i in range(data.shape[0]-1):
            # if i in [1,7,12,15,19,20,21,22,23,24,25,29,30,34,37,40,42,44,46,49,54,58,59,63,64,69,\
            #          70,71,77,78,81,82,84,85,88,89,97,99,100,101,103,105,110,113,115,121,127,132,133,136,138,\
            #             139,140,146,149,151,154,158,160,168,171,173,174,177,185,188,189,192,197]:
            #     pass
            # else:
            episode = {}
            observations = data[i,:self.horizon,2:]
            episode['observations'] = observations
            actions = data[i,:self.horizon,:2]
            episode['actions'] = actions
            episode['terminals'] = np.zeros((actions.shape[0],1))
            self.diffusion_trainer.dataset.fields.add_path(episode, online=True)
        self.diffusion_trainer.dataset.fields.finalize()
        # renderer.composite('verification_rollout1.png', self.diffusion_trainer.dataset.fields['observations'], ncol=20)
        # renderer.composite('verification_rollout1.png', self.diffusion_trainer.dataset.fields['observations'][:132,:128,:], ncol=12)
        # renderer.composite('verification_rollout2.png', data[100:200,:128,2:], ncol=20)
        # renderer.composite('verification_rollout3.png', data[200:300,:128,2:], ncol=20)
        # renderer.composite('verification_rollout4.png', data[300:400,:128,2:], ncol=20)
        # renderer.composite('verification_rollout5.png', data[400:500,:128,2:], ncol=20)
        # renderer.composite('verification_rollout6.png', data[500:600,:128,2:], ncol=20)
        # renderer.composite('verification_rollout7.png', data[600:700,:128,2:], ncol=20)
        # renderer.composite('verification_rollout8.png', data[700:800,:128,2:], ncol=20)

    def load(self, loadpath):
        '''
            loads model and ema from disk
        '''
        data = torch.load(loadpath)

        self.step = data['step']
        self.diffusion_trainer.model.load_state_dict(data['model'])
        self.diffusion_trainer.ema_model.load_state_dict(data['ema'])

        self.diffusion_trainer.ema_model.ddim = args_train.ddim
        if not self.diffusion_trainer.ema_model.ddim:
            self.load_ddpm()
        else:
            self.load_ddim()
    
    def load_ddpm(self):
        betas = helpers.cosine_beta_schedule(self.args_train.n_diffusion_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        a = self.diffusion_trainer.ema_model
        a.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(args_train.device))
        a.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(args_train.device))
        a.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).to(args_train.device))
        a.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to(args_train.device))
        a.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).to(args_train.device))
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        a.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(args_train.device))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        a.register_buffer('posterior_variance', posterior_variance.to(args_train.device))
        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        a.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)).to(args_train.device))
        a.register_buffer('posterior_mean_coef1',
            (betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(args_train.device))
        a.register_buffer('posterior_mean_coef2',
            ((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(args_train.device))

    def load_ddim(self):
        betas = helpers.cosine_beta_schedule(self.args_train.n_diffusion_steps)
        betas_ddim = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
        alphas_ddim = 1. - betas_ddim
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        a = self.diffusion_trainer.ema_model
        alphas_cumprod_ddim = torch.cumprod(alphas_ddim, axis=0)
        a.register_buffer('betas_ddim', betas_ddim.to(args_train.device))
        a.register_buffer('alphas_cumprod_ddim', alphas_cumprod_ddim.to(args_train.device))
        a.skip = a.n_timesteps // a.ddim_timesteps
        a.seq = list(range(0, a.n_timesteps, a.skip))
        a.next_seq = list(range(-a.skip, a.n_timesteps-a.skip, a.skip))
        a.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(args_train.device))
        a.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(args_train.device))

    def test_model(self,epoch,name,loadpath):
        self.load(loadpath)
        for _ in range(1):
            folder = _make_dir(name)
            # num_trainsteps = self.process_dataset()
            sum = 0
            sum_score = 0
            cnt = 0
            for it in range(epoch):
                rew, score = self.test(it*10+1, folder)
                sum += rew
                cnt += (rew>0)
                self.score.append(score)
            self.cnt.append(cnt)
            self.rew.append(sum/epoch)
            print(cnt, sum/epoch)
        print(self.cnt, self.rew)


    def train(self):
        """Online training scenerio."""

    #------------------------- Training ---------------------------#
        # num_trainsteps = self.process_dataset()
        n_epochs = int(args_train.n_train_steps // args_train.n_steps_per_epoch)
        for it in range(n_epochs):      
            
            self.diffusion_trainer.train(args_train.n_steps_per_epoch, online=False)
            print(it)
            # if it % 1 == 0 and it > 0:
            #     self.test(it)
            #     data = {
            #     'step': it,
            #     'model': self.diffusion_trainer.model.state_dict(),
            #     'ema': self.diffusion_trainer.ema_model.state_dict()
            #     }
            #     savepath = f'state_{it}.pt'
            #     torch.save(data, savepath)
            #     print(f'[ utils/training ] Saved model to {savepath}')


    def test(self, it, folder):
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
                    actions, samples = self.policy(cond, it, batch_size=self.args.batch_size, verbose=self.args.verbose)
                    sequence = samples.observations[0]
                    fullpath = folder + '/{}_{}_test.png'.format(it,t//self.horizon)
                    renderer.composite(fullpath, samples.observations, ncol=1)
                    # action = actions[0]

                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                    # action = actions[t]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0
                    # action= next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            else:
                if t % self.horizon == 0:
                    cond[0] = self.observation
                    state = self.env.state_vector().copy()
                    actions, samples = self.policy(cond, it, batch_size=self.args.batch_size, verbose=self.args.verbose)
                    fullpath = folder + '/{}_{}_test.png'.format(it,t//self.horizon)
                    renderer.composite(fullpath, samples.observations, ncol=1)
                    # action = actions[0]
                sequence = samples.observations[0]
                try:
                    next_waypoint = sequence[t-t//self.horizon*self.horizon+1].copy()
                    # action=actions[t]
                except:
                    next_waypoint = sequence[t-t//self.horizon*self.horizon].copy()
                    # action=actions[t-1]
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

            next_observation, reward, terminal, _ = self.env.step(action)
            self.total_reward += reward
        
            
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
        
            # if t % 299 == 0:
            #     # fullpath = join(args.savepath, '{}_{}_test.png'.format(it,t//self.horizon))
            #     fullpath = '{}_{}_test.png'.format(it,t//self.horizon)
            #     renderer.composite(fullpath, samples.observations, ncol=1)

            if t % args.vis_freq == 0 or terminal:


                # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                ## save rollout thus far
                # renderer.composite(join(args.savepath, '{}_test_rollout.png'.format(it)), np.array(self.rollout)[None], ncol=1)
            
                save_path = folder + '/{}_test_rollout.png'.format(it)
                renderer.composite(save_path, np.array(self.rollout)[None], ncol=1)
                # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)
            if terminal:
                break

            self.observation = next_observation
            print(t)
        
            ## render every `args.vis_freq` steps
        score = self.env.get_normalized_score(self.total_reward)
        print(self.total_reward, score)
        return self.total_reward, score


    def save_buffer(self, it):
        np.save('buffer_for_visualization_{}'.format(it), self.value_trainer.dataset.fields['observations'])


    def process_dataset(self):
        """Normalize and preprocess the training data."""
        
        for dataset in [self.diffusion_trainer.dataset]:
            dataset.normalizer = DatasetNormalizer(dataset.fields, normalizer=self.normalizer, \
                                                    path_lengths=dataset.fields['path_lengths'])
            dataset.n_episodes = dataset.fields.n_episodes
            dataset.path_lengths = dataset.fields.path_lengths
            dataset.normalize()
        
        self.diffusion_trainer.dataset.indices = self.diffusion_trainer.dataset.make_indices(self.diffusion_trainer.dataset.fields.path_lengths, self.diffusion_trainer.dataset.horizon)



        self.diffusion_trainer.dataloader = cycle(torch.utils.data.DataLoader(
            self.diffusion_trainer.dataset, batch_size=8, \
            num_workers=1, shuffle=True, pin_memory=True
        ))
        self.policy.normalizer = self.diffusion_trainer.dataset.normalizer

        num_trainsteps = 2000
        return num_trainsteps

    
    def override_dataset_args(self, dataset, args_source, args_list):
        """Set the hyperparameters for diffusion dataset."""

        for k in args_list:
            v = copy.deepcopy(args_source.__getattribute__(k))
            dataset.__setattr__(k, v)

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
        device=args_train.device
    )
    dataset_d = dataset_config_d()
    data = data = np.load('/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/4_10_gt_pexplore0.5_uniformsample_load/buffer_vis.npy')
    dataset_d.reinit(data)

    renderer_d = render_config_d()
    if dataset_d.env.name == 'maze2d-large-v1':
        observation_dim = 4
        action_dim = 2
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
        ddim = False,
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
    )
    model_d = model_config_d()
    diffusion_d = diffusion_config_d(model_d)
    trainer_d = trainer_config_d(diffusion_d, dataset_d, renderer_d, args.device)


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


    if args_train.ddim == True:
        sample = diffuser.sampling.n_step_guided_ddim_sample
    else:
        sample = diffuser.sampling.n_step_guided_p_sample

    ## policies are wrappers around an unconditional diffusion model and a value guide
    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs
        sample_fn=sample,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        # eta=args.eta,
        verbose=False,
    )

    policy = policy_config()


    
    _online_trainer = online_trainer(policy, args, args_train, \
                   trainer_d, trainer_v)
    # _online_trainer.train()
    name = 'train_ddim_closeloop_{}'.format(args_train.ddim_timesteps)
    _online_trainer.test_model(2,name=name,\
        loadpath='/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/4_10_gt_pexplore0.5_uniformsample_load/state_360000.pt')
