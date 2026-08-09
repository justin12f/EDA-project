"""Encryption for customer database credentials.

`data_sources.dsn_encrypted` has carried the comment "Never returned by any
read endpoint. Encrypted before it is written" since the first migration,
with no implementation behind it. This is that implementation.

The plaintext DSN exists only inside an adapter at connection time. It never
enters a response, a log, an agent's context, or a proposal's spec.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from lumen_api.settings import get_settings

# The stored form is "<version>:<token>". Carrying the version means a key
# rotation is a code change plus a lazy re-encrypt, never a migration that
# has to rewrite every row at once.
_CURRENT_VERSION = "v1"


class CredentialError(RuntimeError):
    """A credential that cannot be encrypted or decrypted."""


def _cipher() -> Fernet:
    settings = get_settings()
    if settings.credential_encryption_key is None:
        raise CredentialError(
            "CREDENTIAL_ENCRYPTION_KEY is not configured — customer database "
            "sources cannot be connected without it."
        )
    try:
        return Fernet(settings.credential_encryption_key.get_secret_value().encode())
    except (ValueError, TypeError) as exc:
        raise CredentialError("CREDENTIAL_ENCRYPTION_KEY is not a valid Fernet key") from exc


def encrypt_dsn(plain: str) -> str:
    token = _cipher().encrypt(plain.encode()).decode()
    return f"{_CURRENT_VERSION}:{token}"


def decrypt_dsn(stored: str) -> str:
    version, _, token = stored.partition(":")
    if not token:
        raise CredentialError("stored credential is malformed")
    if version != _CURRENT_VERSION:
        raise CredentialError(f"unknown credential key version {version!r}")
    try:
        return _cipher().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise CredentialError("stored credential failed authentication") from exc
