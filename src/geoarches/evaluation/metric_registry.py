"""Available metrics to compute in eval_multistep.py script."""

from typing import List, Type

import torchmetrics
from geoarches.metrics.brier_skill_score import Era5BrierSkillScore
from geoarches.metrics.ensemble_metrics import Era5EnsembleMetrics

# Registry holds available metrics.
# Maps metric_name to tuple holding a class and its arguments (class_reference, args, kwargs).
# Will only instantiate metric classes on demand.
registry = {}


def available_metrics() -> List[str]:
    return list(registry.keys())


# Function to register a metric class with its arguments.
def register_class(metric_name: str, class_ref: Type[torchmetrics.Metric], **kwargs):
    registry[metric_name] = (class_ref, kwargs)


# Instantiate the metric on demand.
def instantiate_metric(metric_name: str, **extra_kwargs):
    if metric_name in registry:
        class_ref, kwargs = registry[metric_name]
        return class_ref(**kwargs, **extra_kwargs)
    else:
        raise ValueError(
            f"Metric {metric_name} not found in registry. "
            f"Available metrics: {available_metrics()}"
        )


#######################################################
###### Registering classes with their arguments. ######
#######################################################
register_class(
    "era5_ensemble_metrics",
    Era5EnsembleMetrics,
    save_memory=True,
)
# Need different instantiations of brier skill score because
# implementation uses quantiles computed from groundtruth.
register_class(
    "era5_brier_skill_score",
    Era5BrierSkillScore,
    quantiles_filepath="era5-quantiles-2016_2022.nc",
)
register_class(
    "hres_brier_skill_score",
    Era5BrierSkillScore,
    quantiles_filepath="hres-quantiles-2016_2022.nc",
)
