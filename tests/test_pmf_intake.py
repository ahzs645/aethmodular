"""Future-batch intake regression tests."""
from aethmodular_cli.pmf_intake import evaluate


def row(filter_id, **changes):
    value = {
        "filter_id": filter_id, "site": "ETBI", "particle_fraction": "PM2.5",
        "analysis_id": filter_id + "-scan1", "analysis_status": "retained",
        "date": "2026-11-01", "volume_m3": "7.2", "lot": "254", "blank_count": "3",
    }
    value.update(changes)
    return value


def test_new_filter_count_is_unbounded():
    rows = [row(f"ETBI-{i:04d}-1") for i in range(90)]
    results = evaluate(rows)
    assert len(results) == 90
    assert all(r["status"] == "eligible" for r in results)


def test_new_filter_not_historical():
    results = evaluate([row("ETAD-9999-1", site="ETAD")],
                       historical_ids=["ETAD-0001-1"])
    assert results[0]["historical_102"] is False


def test_holds_and_exclusions_do_not_block_good_samples():
    rows = [
        row("good"), row("missing", volume_m3=""),
        row("blank", blank_count="1"),
        row("hole", exclusion_reason="large hole"),
        row("pm10", particle_fraction="PM10"),
    ]
    statuses = {r["filter_id"]: r["status"] for r in evaluate(rows)}
    assert statuses == {
        "good": "eligible", "missing": "held", "blank": "held",
        "hole": "excluded", "pm10": "excluded",
    }


def test_duplicate_retained_scans_are_held():
    records = [row("repeat", analysis_id="A"), row("repeat", analysis_id="B")]
    result = evaluate(records)
    assert result[0]["status"] == "held"
