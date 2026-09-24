"""Refresh this run's proxy credentials using the installed Colab CLI SDK.

Run with the CLI's own Python interpreter. This never allocates a runtime and
only reconnects the exact endpoint recorded in CLOUD_RUN.json.
"""

import argparse
import json
from pathlib import Path

from colab_cli.auth import AuthProvider
from colab_cli.common import state
from colab_cli.commands.session import spawn_keep_alive
from colab_cli.state import SessionState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    endpoint = record["endpoint"]
    name = record["session"]
    state.auth_provider = AuthProvider.ADC
    matches = [a for a in state.client.list_assignments() if a.endpoint == endpoint]
    if len(matches) != 1:
        raise SystemExit("Recorded Colab runtime is no longer assigned; checkpoints remain local.")
    assignment = matches[0]
    session = state.store.get(name)
    reconnect = session is None
    if session is not None and session.endpoint != endpoint:
        raise SystemExit("Session name points at another runtime; refusing to modify it.")
    if reconnect:
        session = SessionState(name=name, endpoint=endpoint, token="", url="")
    session.token = assignment.runtime_proxy_info.token
    session.url = assignment.runtime_proxy_info.url
    state.store.add(session)
    if reconnect:
        session.keep_alive_pid = spawn_keep_alive(endpoint, name, auth_provider=state.auth_provider)
        state.store.add(session)
    print(json.dumps({"session": name, "reconnected": reconnect,
                      "credential_lifetime_seconds": assignment.runtime_proxy_info.token_expires_in_seconds}))


if __name__ == "__main__":
    main()
