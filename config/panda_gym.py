import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

logbase ='logs'

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        # 'preprocess_fns': ['maze2d_set_terminals'],
        'preprocess_fns': None,
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 20000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        # 'horizon': 256,
        # 'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## value-specific kwargs
        'discount': 0.997,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        # 'max_path_length': 200,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        # 'n_steps_per_epoch': 10000,
        'n_steps_per_epoch': 8000,
        'loss_type': 'value_l2',
        # 'n_train_steps': 200e3,
        'n_train_steps': 8000 * 20,
        'batch_size': 8,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:1',
        # 'seed': None,
        'n_samples': 10,
    },


    'plan': {
        'guide': 'sampling.EBM_DensityGuide',
        'policy': 'sampling.GuidedPolicy',
        'batch_size': 1,
        'device': 'cuda',
        'preprocess_fns': [],
        'verbose': True,
        # 'seed': None,

        ## sample_kwargs
        'n_guide_steps': 1,
        't_stopgrad': 2,
        'scale_grad_by_std': False,
        'eta': 0.1,
        'discount': 0.997,

        ## diffusion model
        'horizon': 325,
        'n_diffusion_steps': 1540,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/try_verification',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': True,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

    'online_training': {
        
        ## useful
        'dirname_d':'panda_finetune_target',
        'env':'PandaStack-v3',
        'horizon': 2,
        'max_path_length': 400,
        'batch_size': 1,
        'n_diffusion_steps': 128*2,        
        'train_freq': 10,
        'ddim_timesteps': 8,
        'device': 'cuda:3',
        'ddim': True,
        'online': True,
        'normalizer': 'LimitsNormalizer',
        'max_n_episodes': 5000,
        'conditional': True,
        'p_explore': 0.0,   #default: 0.2
        'set_t':0.1,
        'ebm':True,
        'predict_epsilon':True,
        'guide': 'sampling.EBM_DensityGuide',
        'load_model':False,
        'reinit_buffer':False,
        'scale': 0.001,
        'predict_action':False,
        'predict_action_only':False,
        'loss_type': 'l2',


        'discount': 0.997,
        'termination_penalty': -100,
        'normed': False,
        'use_padding': False,
        'n_train_steps': 2e6,
        'n_steps_per_epoch': 10000,
        "iterations": 30001,
        'buffer_size': 5,
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,
        'online_train_freq': 1,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'verbose':True,
        # 'seed': None,
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

