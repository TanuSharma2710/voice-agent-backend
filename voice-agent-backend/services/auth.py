from __future__ import annotations

import logging
from typing import Any

import requests

from config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY

logger = logging.getLogger(__name__)


class AuthError(Exception):
    pass


class AuthService:
    def __init__(self) -> None:
        self.supabase_url = SUPABASE_URL
        self.supabase_publishable_key = SUPABASE_PUBLISHABLE_KEY

    def verify_token(self, token: str) -> dict[str, Any]:
        if not self.supabase_url:
            raise AuthError("SUPABASE_URL is not configured.")
        if not self.supabase_publishable_key:
            raise AuthError("SUPABASE_PUBLISHABLE_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": self.supabase_publishable_key,
        }
        user_url = f"{self.supabase_url}/auth/v1/user"
        response = requests.get(user_url, headers=headers, timeout=10)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise AuthError("Invalid or expired token.")
        else:
            raise AuthError(f"Token verification failed: {response.status_code}")

    def get_user_id(self, token: str) -> str:
        user_data = self.verify_token(token)
        user_id = user_data.get("id")
        if not user_id:
            raise AuthError("Token does not contain a user ID.")
        return user_id


_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service