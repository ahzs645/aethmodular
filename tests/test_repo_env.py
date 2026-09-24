"""The repo-root .env loader: parsing, precedence, and record-safe paths."""

from pathlib import Path

from aethmodular_cli import env


def test_parse_handles_comments_quotes_and_export():
    text = """
# comment
export AETHMODULAR_DRIVE_ROOT="~/Drive/My Drive"
ZOER_URL='https://zoer.example.org'
EMPTY=
not a pair
"""
    assert env._parse(text) == {
        "AETHMODULAR_DRIVE_ROOT": "~/Drive/My Drive",
        "ZOER_URL": "https://zoer.example.org",
        "EMPTY": "",
    }


def test_shell_environment_wins_and_blank_values_are_skipped(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("ZOER_URL=https://from-file\nOPENRESEARCH_URL=http://from-file\nBLANK=\n")
    monkeypatch.setenv("ZOER_URL", "https://from-shell")
    monkeypatch.delenv("OPENRESEARCH_URL", raising=False)
    monkeypatch.delenv("BLANK", raising=False)

    applied = env.load_repo_env(path)

    assert applied == {"OPENRESEARCH_URL": "http://from-file"}
    assert env.os.environ["ZOER_URL"] == "https://from-shell"
    assert "BLANK" not in env.os.environ


def test_missing_file_is_a_no_op(tmp_path):
    assert env.load_repo_env(tmp_path / "absent.env") == {}


def test_display_path_is_repo_relative_or_home_relative():
    assert env.display_path(env.REPO_ROOT / "docs" / "x.md") == "docs/x.md"
    assert env.display_path(Path.home() / ".local" / "y") == "~/.local/y"
