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
        transition_dim=transition_dim,
        cond_dim=observation_dim,
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

    trainer_state = trainer_config(diffusion, state_model, dataset_state, args.device, renderer, 
                                   args.state_batchsize, diffusion_savepath+'/state', args.loadpath_state)
    trainer_traj = trainer_config(diffusion, trajectory_model, dataset_traj, args.device, renderer, 
                                  args.traj_batchsize, diffusion_savepath+'/traj', args.loadpath_traj)
    state_model = trainer_state.ema_model
    diffusion = trainer_traj.diffusion_model

    

    import numpy as np
    import torch
    side_x = np.linspace(0.5,7.2,68)
    side_y = np.linspace(0.5,10.2,98)
    X, Y = np.meshgrid(side_x, side_y)
    Z = np.zeros((68,98))
    first=True
    for i in side_x:
        for j in side_y:
            if first:
                m = np.array([i,j]).reshape((1,2))
                first=False
            else:
                m = np.concatenate((m,np.array([i,j]).reshape((1,2))),axis=0)
    m = np.concatenate((m, np.zeros_like(m)),axis=1)
    m = m[:,None]
    m = dataset_state.normalizer.normalize(m, 'observations')
    x = torch.tensor(m, dtype=torch.float32, device=args.device)
    noise_num = 10
    for _ in range(noise_num):
        for t in range(0, 5):
            t = torch.ones(13328//2, dtype=torch.long, device=args.device) * t
            x_noisy = diffusion.q_sample(x_start=x, t=t)
            energy_ij = state_model.point_energy(x_noisy, None, t)
            energy_ij = energy_ij.squeeze().detach().cpu().numpy()
            energy_ij = energy_ij.reshape((68,98))
            Z += energy_ij
    Z /= noise_num
    import matplotlib.pyplot as plt
    np.save('density.npy', Z)
    plt.pcolormesh(X, Y, Z.T, shading='auto')
    plt.savefig('density_sum2.png')

        

if __name__ == "__main__":
    
    main()

