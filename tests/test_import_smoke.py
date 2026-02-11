"""Import smoke tests for canonical package entry points."""


def test_import_smoke():
    import src  # noqa: F401
    import src.analysis.advanced.statistical_analysis  # noqa: F401
    import src.analysis.aethalometer.smoothening  # noqa: F401
