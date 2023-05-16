import hydra
import utils
import torch
import logging
import gymnasium as gym
from hydra.utils import instantiate
from gymnasium.wrappers import RecordEpisodeStatistics
from core import train
from dotmap import DotMap
from omegaconf import OmegaConf
from buffer import get_buffer
import panda_gym
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_float32_matmul_precision('middle')


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    env = RecordEpisodeStatistics(gym.make(cfg.env_name, render_mode="rgb_array"))

    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)
    log_dict = utils.get_log_dict(cfg.agent._target_)
    for seed in cfg.seeds:
        utils.set_seed_everywhere(env, seed)
        buffer = get_buffer(cfg.buffer, state_size=state_size, action_size=action_size, device=device, seed=seed)
        agent = instantiate(cfg.agent, state_size=state_size, action_size=action_size, action_space=env.action_space, device=device)
        logger.info(f"Training seed {seed} for {cfg.train.timesteps} timesteps with {agent} and {buffer}")
        # get_attr of omega_conf is slow, so we convert it to dotmap
        train_cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
        eval_mean = train(train_cfg, env, agent, buffer, seed, log_dict)
        logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")


if __name__ == "__main__":
    main()
