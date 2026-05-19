"""
metrics/registry_loaders/feature_attribution_ts_multivariate.py

Register all feature-attribution multivariate time-series metrics
via side-effect imports.
"""

import metrics.metrics.feature_attribution.timeseries_multivariate.complexity.complexity_entropy  # noqa: F401,E501
import metrics.metrics.feature_attribution.timeseries_multivariate.complexity.sparseness  # noqa: F401,E501
import metrics.metrics.feature_attribution.timeseries_multivariate.faithfulness.faithfulness_correlation  # noqa: F401,E501
import metrics.metrics.feature_attribution.timeseries_multivariate.faithfulness.pixel_flipping  # noqa: F401,E501
import metrics.metrics.feature_attribution.timeseries_multivariate.robustness.avg_sensitivity  # noqa: F401,E501
import metrics.metrics.feature_attribution.timeseries_multivariate.robustness.continuity  # noqa: F401,E501

