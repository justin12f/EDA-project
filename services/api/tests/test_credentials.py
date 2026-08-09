"""DSN encryption. `data_sources.dsn_encrypted` has carried the comment
"Encrypted before it is written" since the first migration with nothing
behind it — this is that."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from lumen_api.credentials import CredentialError, decrypt_dsn, encrypt_dsn
from lumen_api.settings import get_settings

DSN = "postgresql://acme:hunter2@db.acme.example:5432/production"


@pytest.fixture(autouse=True)
def _credential_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest.py's _isolate_environment fixture strips every ambient env
    var each test, so nothing configures CREDENTIAL_ENCRYPTION_KEY unless a
    test supplies one itself."""
    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_round_trip():
    assert decrypt_dsn(encrypt_dsn(DSN)) == DSN


def test_the_plaintext_never_appears_in_the_ciphertext():
    stored = encrypt_dsn(DSN)
    for fragment in ("hunter2", "acme", "db.acme.example", "production"):
        assert fragment not in stored


def test_the_stored_form_carries_a_key_version():
    """So a future rotation needs no migration — the prefix says which key
    encrypted this row."""
    assert encrypt_dsn(DSN).startswith("v1:")


def test_a_tampered_token_raises():
    stored = encrypt_dsn(DSN)
    tampered = stored[:-4] + ("aaaa" if not stored.endswith("aaaa") else "bbbb")
    with pytest.raises(CredentialError):
        decrypt_dsn(tampered)


def test_an_unknown_key_version_raises_clearly():
    with pytest.raises(CredentialError, match="version"):
        decrypt_dsn("v99:whatever")


def test_a_malformed_value_raises():
    with pytest.raises(CredentialError):
        decrypt_dsn("not-even-prefixed")


def test_two_encryptions_of_the_same_dsn_differ():
    """Fernet includes a random IV, so identical inputs must not produce
    identical ciphertext — otherwise the column leaks which customers share
    a database."""
    assert encrypt_dsn(DSN) != encrypt_dsn(DSN)
