
logbase ='logs'

base = {

    'online_training': {

        'dirname':'debug_addbuffer_resample',

        # environment:
        'env_wrapper': 'environments.d4rl_env_wrapper',
        'env': 'maze2d-large-v1',
        'render_mode': 'rgb_array',

        # dataset
        'loader': 'datasets.GoalDataset',
        'horizon': 1,
        'normalizer': 'LimitsNormalizer',
        'max_path_length': 800,
        'max_n_episodes': 3000,
        'predict_type': 'obs_only',
        'device': 'cuda:0',
        'use_fake_buffer':True,

        # renderer
        'renderer': 'utils.Maze2dRenderer',

        # model
        'wrapper': 'models.model_wrapper',
        'model_name': 'TemporalUnet',
        'dim_mults_trajectory': (1, 4, 8),
        'dim_mults_pair': (1, 4, 8),
        'ebm': True,

        # diffusion
        'diffusion': 'models.GaussianDiffusion',
        'ddim': True,
        'ddim_timesteps': 8,
        'n_diffusion_steps': 128,
        'clip_denoised': True,
        'predict_epsilon': False,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'condition_type': 'extend', #normal

        # trainer
        'traj_batchsize': 1,
        'state_batchsize': 512,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'sample_freq': 1000,
        'save_freq': 1000,
        'label_freq': 40000,
        'save_parallel': False,
        'bucket': None,
        'n_reference': 50,
        'n_samples': 1,
        'loadpath_traj': None,
        'loadpath_state': None,

        # plan
        # 'guide': 'sampling.EBM_DensityGuide',
        'policy': 'sampling.GuidedPolicy',
        'scale': 0.001, # grad coef in guide
        'n_guide_steps': 1,
        't_stopgrad': 2,
        'scale_grad_by_std': False,
        'eta': 0.0,
        'verbose': True,

        # online trainer
        'train_freq': 1000,
        'iterations': 300001,
        'traj_len': 400,

    },

}
