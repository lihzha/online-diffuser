import diffuser.utils as utils
from diffuser.trainer import OnlineTrainer
import os
import time


class Parser(utils.Parser):
    # dataset: str = 'maze2d-large-v1'
    dataset: str = 'PandaReach-v3'
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

def main():

    args = Parser().parse_args('online_training')
    dirname = args.dirname
    diffusion_savepath = _make_dir(f'logs/{args.env}/diffusion/',dirname)
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
        traj_len=args.traj_len,
        horizon=args.horizon,
        normalizer=args.normalizer,
        max_path_length=args.max_path_length,
        max_n_episodes=args.max_n_episodes,
        predict_type = args.predict_type,
    )
    dataset = dataset_config()


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
        savepath=(diffusion_savepath, 'model_config_d.pkl'),
        horizon=args.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=args.dim_mults,
        device=args.device,
    )
    
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(diffusion_savepath, 'diffusion_config_d.pkl'),
        horizon=args.horizon,
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
        train_batch_size=args.train_batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=args.label_freq,
        save_parallel=args.save_parallel,
        results_folder=diffusion_savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
        n_samples=args.n_samples,
        loadpath=args.loadpath,
    )
    model = model_config()

    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, args.device, renderer)

    
    diffusion = trainer.ema_model

    guide_config = utils.Config(args.guide, model=model, verbose=False)
    guide = guide_config()


    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
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

    _online_trainer = OnlineTrainer(model, trainer, env, dataset, policy, args.predict_type)
    _online_trainer.train(args.train_freq, args.iterations)
        

if __name__ == "__main__":
    
    main()


