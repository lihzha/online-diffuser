from .EBM import EBMDiffusionModel
from .temporal import TemporalUnet, TemporalValue, ValueFunction

def model_wrapper(model, ebm, dim_mults, horizon, condition_type, **kwargs):
    model = eval(model)(dim_mults=dim_mults, horizon=horizon,**kwargs)
    if ebm:
        model = EBMDiffusionModel(model, condition_type=condition_type)
    return model
