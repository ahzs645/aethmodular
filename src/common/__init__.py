"""Shared helpers consolidated from notebooks and research/ scripts.

Single home for functions that were previously copy-pasted across dozens of
notebooks and research subdirectories (Deming/ODR regression, OLS regression
statistics, filter-ID normalization, unit coercion, season assignment, repo
path bootstrap). Import from here instead of redefining inline:

    from src.common import deming, regression_stats, base_filter_id, to_ugm3
"""

from .regression import deming, deming_lambda, regression_stats, fit_fabs_ec
from .filter_ids import base_filter_id, normalize_filter_id
from .units import to_ugm3
from .seasons import season_for_month, assign_season, SITE_SEASONS
from .paths import find_repo_root

__all__ = [
    "deming",
    "deming_lambda",
    "regression_stats",
    "fit_fabs_ec",
    "base_filter_id",
    "normalize_filter_id",
    "to_ugm3",
    "season_for_month",
    "assign_season",
    "SITE_SEASONS",
    "find_repo_root",
]
