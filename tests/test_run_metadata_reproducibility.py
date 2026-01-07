"""
Run metadata reproducibility tests.
"""

from ssp.workflows.metadata import collect_env_fingerprint, hash_payload


def test_hash_payload_stable() -> None:
    a = {"b": 2, "a": 1, "x": {"z": 3, "y": 4}}
    b = {"a": 1, "b": 2, "x": {"y": 4, "z": 3}}

    ha = hash_payload(a)
    hb = hash_payload(b)
    assert ha == hb


def test_env_fingerprint_has_keys() -> None:
    env = collect_env_fingerprint()
    assert "python" in env
    assert "platform" in env
