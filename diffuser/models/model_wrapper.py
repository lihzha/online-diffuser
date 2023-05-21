from .EBM import EBMDiffusionModel
from .temporal import TemporalUnet, TemporalValue, ValueFunction

def model_wrapper(model, ebm, **kwargs):
    model = eval(model)(**kwargs)
    if ebm:
        model = EBMDiffusionModel(model)
    return model
