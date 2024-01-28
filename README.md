## Installation

```
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```

If you encounter any missing packages after running above codes, please install them on your own and report the bugs to me. Thanks for your understanding.

## Run experiments

```
python scripts/train.py
```
For the usage of arguments, please refer to `config/maze2d_config.py`, which defines all hyper-parameters, configurations for training and testing.

All the results will be saved in the folder `logs/`, including the visualizations. Make sure to check them occasionally during the training process to make sure things are going right.

## Code structure
`diffuser/datasets`: contains codes about how data/trajectories are loaded, processed and setored in the buffer. If you want to use offline dataset, please refer to codes in this folder.

`diffuser/environments`: defines all mujoco environments. Currently we don't need to care about them since we run environments in maze-2d.

`diffuser/models`: contains all core codes about diffusion models and energy-based diffusion models. You should carefully review the codes in `./diffusion.py` and `./EBM.py` and make sure you fully understand them. If you are interested in how U-Net looks like, you can refer to `./temporal.py`. You don't need to check `./id.py` and `./value_function.py`.

`diffuser/sampling`: contains codes about the core reverse sampling process (`./functions.py`), definitions of different guides (`./guides.py`, you only need to check class EBM_DensityGuide), and the wrapper implementation of the diffuser policy (`./policies.py`).

`diffuser/trainer`: defines all functions for online training in `./online_trainer.py`, and defines the training process of diffusion models in `./trainer.py`. Make sure to check both files and understand what each functions are doing.

`diffuser/utils`: defines all utilization functions., e.g. rendering, transformation and timing. You only need to check them when you need to debug them (which is rare).

`scripts`: defines all runnable scripts. Leave all the scripts in `scripts/old_scripts` alone. You need to carefully check the code in `./train.py`.

## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.