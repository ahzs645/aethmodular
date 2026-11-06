"""Thin command router; reusable scientific implementation lives in src.pmf."""
from src.pmf.cli import main
from src.pmf.io import read_matrix as _read


def load_pair(concentrations, uncertainties):
    """Compatibility import for older callers; no missing-value coercion."""
    from src.pmf.io import from_csv
    data = from_csv(concentrations, uncertainties)
    return data.sample_ids.tolist(), data.features.tolist(), data.X, data.U


if __name__ == '__main__':
    raise SystemExit(main())
