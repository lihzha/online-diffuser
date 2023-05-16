
import hydra
import torch
import logging
from utils import config_logging, get_log_dict
from core_mp import train
import torch.multiprocessing as mp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
config_logging("main_mp.log")
torch.set_float32_matmul_precision('high')


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    logger = logging.getLogger(__name__)
    manager = mp.Manager()
    num_seeds = len(cfg.seeds)
    barrier = manager.Barrier(num_seeds)
    log_dict = get_log_dict(cfg.agent._target_, manager, num_seeds)
    pool = mp.Pool(num_seeds)
    pool.starmap(train, [(cfg, seed, log_dict, idx, logger, barrier) for (idx, seed) in enumerate(cfg.seeds)])
    pool.close()
    pool.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')  # set spawn for linux servers
    main()
