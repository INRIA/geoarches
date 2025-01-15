import pytest

from geoarches.evaluation import metric_registry


def test_class_instantiation():
    for metric_name in metric_registry.registry:
        try:
            metric_registry.instantiate_metric(metric_name)
        except Exception as e:
            pytest.fail(f"Instantiating metric `{metric_name}` from registry failed: {e}")
