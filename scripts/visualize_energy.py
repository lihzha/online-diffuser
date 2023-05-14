import torch
import numpy as np
import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.models.EBM import EBMDiffusionModel

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#-----------------------------------------------------------------------------#
#---------------------------------- unifying arguments -----------------------#
#-----------------------------------------------------------------------------#
args = Parser().parse_args('plan')
args_train = Parser().parse_args('online_training')
args_d = Parser().parse_args('diffusion')
args_v = Parser().parse_args('values')


dataset_config_d = utils.Config(
    args_d.loader,
    savepath=None,
    env=args_d.dataset,
    horizon=args_d.horizon,
    normalizer=args_d.normalizer,
    preprocess_fns=args_d.preprocess_fns,
    use_padding=args_d.use_padding,
    max_path_length=args_d.max_path_length,
    predict_action = args_train.predict_action,
    online=True,
)
render_config_d = utils.Config(
    args_d.renderer,
    savepath=None,
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
    savepath=None,
    horizon=args_d.horizon,
    transition_dim=transition_dim,
    cond_dim=observation_dim,
    dim_mults=args_d.dim_mults,
    # attention=args_d.attention,
    device=args_train.device,
)

diffusion_config_d = utils.Config(
    args_d.diffusion,
    savepath=None,
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
    device=args_train.device,
    predict_action=args_train.predict_action
)

trainer_config_d = utils.Config(
    utils.Trainer,
    device=args_train.device,
    savepath=None,
    train_batch_size=args_d.batch_size,
    train_lr=args_d.learning_rate,
    gradient_accumulate_every=args_d.gradient_accumulate_every,
    ema_decay=args_d.ema_decay,
    sample_freq=args_d.sample_freq,
    save_freq=args_d.save_freq,
    label_freq=int(args_d.n_train_steps // args_d.n_saves),
    save_parallel=args_d.save_parallel,
    results_folder=None,
    bucket=args_d.bucket,
    n_reference=args_d.n_reference,
    n_samples = args_d.n_samples,
    online=True
)
model_d = model_config_d()

model_d = EBMDiffusionModel(model_d)

diffusion_d = diffusion_config_d(model_d)
trainer_d = trainer_config_d(diffusion_d, dataset_d, renderer_d, args_train.device)
trainer_d.load('/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/4_13_ebm_scratch_pexplore0.5/state_0.pt')

ebm = trainer_d.ema_model.model

gs_x = 80
gs_y = 110
gs = 10
data = np.empty((gs_x,gs_y))
t = torch.full((1,), 0, device=args_train.device, dtype=torch.long)
for i in range(gs_x):
    for j in range(gs_y):
        inp = torch.full((1,384,4), 0, device=args_train.device, dtype=torch.float)
        inp[:,0,0] = i/gs
        inp[:,0,1] = j/gs
        prob = ebm.neg_logp_unnorm(inp, None, t)[0][0]
        data[i,j] = prob.detach().cpu().numpy()

plt.pcolormesh(data)
plt.savefig('final_result.jpg')
np.savetxt('final_gt.txt',data)
