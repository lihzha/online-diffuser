import diffuser.utils as utils
from diffuser.trainer import OnlineTrainer
import os
import time


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    # dataset: str = 'PandaReach-v3'
    config: str = 'config.maze2d_config'

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

def main():

    args = Parser().parse_args('online_training')
    dirname = args.dirname
    diffusion_savepath = _make_dir(f'logs/{args.env}/diffusion/',dirname)
    if not os.path.exists(diffusion_savepath+'/state'):
        os.makedirs(diffusion_savepath+'/state')
        os.makedirs(diffusion_savepath+'/traj')
    args.save(diffusion_savepath)
    plan_savepath = _make_dir(f'logs/{args.env}/plans/',dirname)
    args.save(plan_savepath)

    env_config = utils.Config(
        args.env_wrapper,
        env=args.env,
    )
    env = env_config()

    render_config = utils.Config(
        args.renderer,
        savepath=(diffusion_savepath, 'render_config.pkl'),
        env=args.env,
    )
    renderer = render_config()

    if args.predict_type not in ['obs_only', 'ation_only', 'joint']:
        raise ValueError('Unknown predict type!')

    dataset_config = utils.Config(
        args.loader,
        savepath=(diffusion_savepath, 'dataset_config_d.pkl'),
        env=env,
        normalizer=args.normalizer,
        max_path_length=args.max_path_length,
        max_n_episodes=args.max_n_episodes,
        predict_type = args.predict_type,
    )

    dataset_state = dataset_config(horizon=args.horizon)
    dataset_traj = dataset_config(horizon=args.traj_len)
    dataset_traj_fake = dataset_config(horizon=args.traj_len)

    if args.condition_type == 'extend':
        observation_dim = 2*env.observation_space.shape[0]
    elif args.condition_type == 'normal':
        observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if args.predict_type == 'obs_only':
        transition_dim = observation_dim
    elif args.predict_type == 'action_only':
        transition_dim = action_dim
    elif args.predict_type == 'joint':
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        args.wrapper,
        model=args.model_name,
        ebm=args.ebm,
        condition_type=args.condition_type,
        savepath=(diffusion_savepath, 'model_config_d.pkl'),
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        device=args.device,
    )
    
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(diffusion_savepath, 'diffusion_config_d.pkl'),
        horizon=args.horizon,
        condition_type=args.condition_type,
        observation_dim=observation_dim,
        action_dim=action_dim,
        ddim = args.ddim,
        traj_len=args.traj_len,
        ddim_timesteps = args.ddim_timesteps,
        n_timesteps=args.n_diffusion_steps,
        predict_type=args.predict_type,
        device=args.device,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        ## loss weighting
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount
    )

    trainer_config = utils.Config(
        utils.Trainer,
        device=args.device,
        savepath=(diffusion_savepath, 'trainer_config_d.pkl'),
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=args.label_freq,
        save_parallel=args.save_parallel,
        bucket=args.bucket,
        n_reference=args.n_reference,
        n_samples=args.n_samples,
    )

    state_model = model_config(dim_mults=args.dim_mults_pair, horizon=args.horizon)
    trajectory_model = model_config(dim_mults=args.dim_mults_trajectory, horizon=args.traj_len)

    diffusion = diffusion_config(model=trajectory_model, state_model=state_model)

    trainer_state = trainer_config(diffusion, state_model, dataset_state, None ,args.device, renderer, 
                                   args.state_batchsize, diffusion_savepath+'/state', args.loadpath_state)
    trainer_traj = trainer_config(diffusion, trajectory_model, dataset_traj, dataset_traj_fake, args.device, renderer, 
                                  args.traj_batchsize, diffusion_savepath+'/traj', args.loadpath_traj)
    if args.eval:
        trainer_traj.load(args.model_path)
        trainer_traj.diffusion_model.model = trainer_traj.ema_model
    else:
        diffusion = trainer_traj.diffusion_model

    policy_config = utils.Config(
        args.policy,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset_traj.normalizer,
        ## sampling kwargs
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        eta=args.eta,
        verbose=args.verbose,
        predict_type=args.predict_type,
        _device=args.device
    )

    policy = policy_config()

    _online_trainer = OnlineTrainer(state_model, trajectory_model, trainer_traj, trainer_state, env, dataset_traj, dataset_traj_fake, dataset_state, policy, args.predict_type,args.use_fake_buffer)
    if not args.eval:
        _online_trainer.train(args.train_freq, args.iterations)
    else:
        _online_trainer.test(args.eval_epoch)
        
if __name__ == "__main__":
    main()


