"""Available metrics to compute in eval_multistep.py script."""

from typing import List, Type

import torchmetrics
from geoarches.metrics.brier_skill_score import Era5BrierSkillScore
from geoarches.metrics.ensemble_metrics import Era5EnsembleMetrics
from geoarches.metrics.rank_histogram import Era5RankHistogram
from geoarches.metrics.spherical_power_spectrum import Era5PowerSpectrum

# Registry holds available metrics.
# Maps metric_name to tuple holding a class and its arguments (class_reference, args, kwargs).
# Will only instantiate metric classes on demand.
registry = {}


def available_metrics() -> List[str]:
    return list(registry.keys())


# Function to register a metric class with its arguments.
def register_metric(metric_name: str, class_ref: Type[torchmetrics.Metric], **kwargs):
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
register_metric(
    "era5_ensemble_metrics",
    Era5EnsembleMetrics,
    save_memory=True,
)
register_metric(
    "era5_rank_histogram_50_members",
    Era5RankHistogram,
    n_members=50,
)
register_metric(
    "era5_rank_histogram_25_members",
    Era5RankHistogram,
    n_members=25,
)
register_metric(
    "era5_power_spectrum",
    Era5PowerSpectrum,
)
register_metric(
    "era5_power_spectrum_with_ref",
    Era5PowerSpectrum,
    compute_target_spectrum=True,
)
# Need different instantiations of brier skill score because
# implementation uses quantiles computed from groundtruth.
register_metric(
    "era5_brier_skill_score",
    Era5BrierSkillScore,
    quantiles_filepath="era5-quantiles-2016_2022.nc",
)
register_metric(
    "hres_brier_skill_score",
    Era5BrierSkillScore,
    quantiles_filepath="hres-quantiles-2016_2022.nc",
)
