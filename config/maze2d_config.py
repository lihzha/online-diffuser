
logbase ='logs'

base = {

    'online_training': {

        'dirname':'maze2d',

        # environment:
        'env_wrapper': 'environments.d4rl_env_wrapper',
        'env': 'maze2d-large-v1',
        'render_mode': 'rgb_array',

        # dataset
        'loader': 'datasets.GoalDataset',
        'horizon': 2,
        'normalizer': 'LimitsNormalizer',
        'max_path_length': 600,
        'max_n_episodes': 2000,
        'predict_type': 'obs_only',
        'device': 'cuda:1',

        # renderer
        'renderer': 'utils.Maze2dRenderer',

        # model
        'wrapper': 'models.model_wrapper',
        'model_name': 'TemporalUnet',
        'dim_mults': (1, 4, 8),
        'ebm': True,

        # diffusion
        'diffusion': 'models.GaussianDiffusion',
        'ddim': True,
        'ddim_timesteps': 8,
        'n_diffusion_steps': 128*2,
        'clip_denoised': True,
        'predict_epsilon': True,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,

        # trainer
        'train_batch_size': 256,
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
        'loadpath': None,

        # plan
        'guide': 'sampling.EBM_DensityGuide',
        'policy': 'sampling.GuidedPolicy',
        'scale': 0.001, # grad coef in guide
        'n_guide_steps': 1,
        't_stopgrad': 2,
        'scale_grad_by_std': False,
        'eta': 0.1,
        'verbose': True,

        # online trainer
        'train_freq': 50,
        'iterations': 30001,
        'traj_len': 300,

    },

}
