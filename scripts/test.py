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

    def __init__(self, policy, args, args_train, trainer_d, trainer_v):
        
        self.policy = policy 

        self.args = args
        self.args_train = args_train

        self.value_trainer = trainer_v
        self.diffusion_trainer = trainer_d

        self.init_args()

        self.initialize_normalizer(self.diffusion_trainer.dataset)

        self.env = self.diffusion_trainer.dataset.env


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

    def test(self, epoch):
        open_loop = True
        self.score = []
        for e in range(epoch):
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
                        if args_train.predict_action:
                            action, samples = self.policy(cond, 1, batch_size=self.args.batch_size, verbose=self.args.verbose, p_explore=0)
                        else:
                            samples = self.policy(cond, 1, batch_size=self.args.batch_size, verbose=self.args.verbose, p_explore=0)
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
                        if args_train.predict_action:
                            action, samples = self.policy(cond,1, batch_size=self.args.batch_size, verbose=self.args.verbose, p_explore=0)
                        else:
                            samples = self.policy(cond, 1, batch_size=self.args.batch_size, verbose=self.args.verbose, p_explore=0)
                    sequence = samples.observations[0]
                    try:
                        next_waypoint = sequence[t-t//self.horizon*self.horizon+1].copy()
                    except:
                        next_waypoint = sequence[t-t//self.horizon*self.horizon].copy()
                    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

                next_observation, reward, terminal, _ = self.env.step(action)
                self.total_reward += reward
                
                if 'maze2d' in args.dataset:
                    xy = next_observation[:2]
                    goal = self.env.unwrapped._target
                    print(
                        f'maze | pos: {xy} | goal: {goal}'
                    )
                self.rollout.append(next_observation.copy())
            
                if t % 299 == 0:
                    fullpath = join(args.savepath, '{}_{}_test.png'.format(e,t//self.horizon))
                    renderer.composite(fullpath, samples.observations, ncol=1)

                if t % args.vis_freq == 0 or terminal:

                    # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                    renderer.composite(join(args.savepath, '{}_test_rollout.png'.format(e)), np.array(self.rollout)[None], ncol=1)

                    # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)
                if terminal:
                    break

                self.observation = next_observation
                print(t)
            
            score = self.env.get_normalized_score(self.total_reward)
            self.score.append(score)
            print(self.total_reward, score)
        # np.savetxt('TestReward_{}_round'.format(it),self.total_reward)
        return self.score

    def initialize_normalizer(self, dataset):

        self.policy.normalizer = dataset.normalizer

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
        predict_action = args_train.predict_action,
        online=False,
    )
    render_config_d = utils.Config(
        args_d.renderer,
        savepath=(args_d.savepath, 'render_config_d.pkl'),
        env=args_d.dataset,
    )
    dataset_d = dataset_config_d()
    renderer_d = render_config_d()
    if dataset_d.env.name == 'maze2d-large-v1':
        observation_dim = 4
        dataset_d.observation_dim = 4
        if args_train.predict_action:
            action_dim = 2
            dataset_d.action_dim = 2
            transition_dim = observation_dim + action_dim
        else:
            action_dim = 0
            dataset_d.action_dim = 0
            transition_dim = observation_dim

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
        print(args.device, args_d.device, args_train.device)
        print("ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ebm!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    diffusion_d = diffusion_config_d(model_d)
    trainer_d = trainer_config_d(diffusion_d, dataset_d, renderer_d, args.device)

    trainer_d.load('/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/5_9_silu_sample_t/state_120000.pt')
    
    ## initialize policy arguments
    diffusion = trainer_d.ema_model
    dataset = dataset_d
    renderer = renderer_d
    # value_function = trainer_v.ema_model
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

    _online_trainer = online_trainer(policy, args, args_train, \
                   trainer_d, None)
    score = _online_trainer.test(25)
    score = np.array(score)
    std = score.std()
    mean = score.mean()
    print('mean:',mean,'deviation',std/np.sqrt(len(score)))
