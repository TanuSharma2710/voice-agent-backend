from __future__ import annotations

import base64
import hashlib
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _get_fernet() -> Fernet:
    key_str = os.getenv("ENCRYPTION_KEY")
    if not key_str:
        raise ValueError("ENCRYPTION_KEY environment variable is not set.")

    salt = b"voice-agent-salt"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(key_str.encode()))
    return Fernet(key)


def encrypt_url(url: str) -> str:
    fernet = _get_fernet()
    return fernet.encrypt(url.encode()).decode()


def decrypt_url(encrypted: str) -> str:
    fernet = _get_fernet()
    return fernet.decrypt(encrypted.encode()).decode()