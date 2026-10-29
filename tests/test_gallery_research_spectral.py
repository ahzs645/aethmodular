"""Reading an executed notebook's saved tables, and the Research tab's spectral export built from them."""
import json

import numpy as np
import pandas as pd
import pytest

from gallery.data.notebook_outputs import ExecutedNotebook, parse_html_tables
from gallery.data import export_research_spectral as exporter


def test_round_trip_drops_a_range_index_and_keeps_types():
    frame = pd.DataFrame({"group": ["Addis", "smoke"], "n": [296, 906], "r": [0.9379, np.nan], "flag": [True, False]})
    (back,) = parse_html_tables(frame.to_html())
    assert list(back.columns) == ["group", "n", "r", "flag"]
    assert back["n"].tolist() == [296.0, 906.0]
    assert back["r"].iloc[0] == 0.9379 and pd.isna(back["r"].iloc[1])
    assert back["flag"].tolist() == [True, False]


def test_a_named_or_label_index_becomes_a_column():
    frame = pd.DataFrame({"median": [1.67, 2.39]}, index=pd.Index(["Dry", "Belg"], name="season"))
    (back,) = parse_html_tables(frame.to_html())
    assert back.to_dict("list") == {"season": ["Dry", "Belg"], "median": [1.67, 2.39]}
    unnamed = pd.DataFrame({"x": [1.0, 2.0]}, index=["a", "b"])
    (back,) = parse_html_tables(unnamed.to_html())
    assert back["index"].tolist() == ["a", "b"]


def test_a_multiindex_row_label_is_carried_down():
    frame = pd.DataFrame({"count": [188, 27, 188]},
                         index=pd.MultiIndex.from_tuples([("k8", "addis"), ("k8", "indh"), ("k9", "addis")], names=["config", "target"]))
    (back,) = parse_html_tables(frame.to_html())
    assert back["config"].tolist() == ["k8", "k8", "k9"]
    assert back["target"].tolist() == ["addis", "indh", "addis"]


def test_a_truncated_display_is_refused():
    wide = pd.DataFrame(np.arange(60).reshape(2, 30))
    with pd.option_context("display.max_columns", 6):
        html = wide._repr_html_()
    with pytest.raises(ValueError, match="truncated"):
        parse_html_tables(html)


def _notebook(tmp_path, cells):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    return ExecutedNotebook(path)


def _code(source, outputs):
    return {"cell_type": "code", "source": source, "outputs": outputs, "metadata": {}, "execution_count": 1}


def test_cells_are_found_by_a_marker_and_must_be_unique(tmp_path):
    table = pd.DataFrame({"a": [1.0]}).to_html()
    nb = _notebook(tmp_path, [
        _code("stats.to_csv('x.csv')\ndisplay(stats)", [
            {"output_type": "stream", "name": "stdout", "text": ["n=12\n"]},
            {"output_type": "display_data", "data": {"text/html": [table], "text/plain": ["   a\n0  1"]}, "metadata": {}},
        ]),
        _code("other.to_csv('y.csv')", []),
        _code("other.to_csv('z.csv')", []),
    ])
    assert nb.table("stats.to_csv").to_dict("list") == {"a": [1.0]}
    assert nb.stream("stats.to_csv") == "n=12\n"
    with pytest.raises(LookupError, match="2 code cells"):
        nb.table("other.to_csv")
    with pytest.raises(LookupError, match="displayed 1 tables"):
        nb.table("stats.to_csv", nth=1)


def test_ranked_sites_keeps_only_complete_entries():
    frame = pd.DataFrame({"target": ["Addis (ETAD)"], "near": ["NOGA1 (0.9902), CHAS1 (0.9885), LTCC1 (0.9866)..."],
                          "mix": ["GRSA1 (0.010), CAVE1 (0.011), PHOE5 (0.012), P..."]})
    near = exporter.ranked_sites(frame, "near", "r")
    assert near["site"].tolist() == ["NOGA1", "CHAS1", "LTCC1"] and near["rank"].tolist() == [1, 2, 3]
    assert exporter.ranked_sites(frame, "mix", "js")["site"].tolist() == ["GRSA1", "CAVE1", "PHOE5"]


@pytest.mark.parametrize("build", exporter.PAGES, ids=lambda b: b.__name__)
def test_every_page_builds_from_the_committed_notebooks(build):
    page = build()
    assert page.tables
    for name, frame in page.tables.items():
        assert len(frame), name
    for source in page.sources:
        assert source["read"] in ("csv", "tracked csv", "executed notebook output", "executed notebook output (printed)",
                                  "recomputed from the Drive (FTIR/apps, FTIR/notebook/extracted)")


def test_headline_numbers_match_what_the_notebooks_print():
    band = exporter.band_1617()
    addis = band.tables["centers"].set_index("group").loc["Addis"]
    assert addis["center_median"] == pytest.approx(1618.654282, abs=1e-4)
    mac = exporter.mac_choice()
    like = mac.tables["bridge_summary"].set_index("population").loc["Addis-like OC/EC ≤ 2.27"]
    # the CSV (when present locally) is unrounded; the notebook prints 2 dp
    assert like["n"] == 6503 and like["median"] == pytest.approx(10.05, abs=0.005)
    setups = mac.tables["setups"].set_index("setup")
    # ftir_19's structural result: the slope at MAC 6 is 0.6 × the slope at MAC 10
    assert np.allclose(setups["slope @ MAC 6"], 0.6 * setups["slope @ MAC 10"], atol=0.006)
    assert len(exporter.committee().tables["five_site"]) == 15
    sites = exporter.network_map().tables["nearest_sites"]
    assert sites[(sites.target == "Addis (ETAD)") & (sites["rank"] == 1)]["site"].tolist()[0] == "NOGA1"
