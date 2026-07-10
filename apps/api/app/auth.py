"""Opt-in API-key auth for sensitive/mutating endpoints (PRD Phase 9 hardening)
plus a per-user auth scaffold + data-isolation identity resolver (additive,
default-off).

When `FLORALENS_API_KEY` is unset (the default — local demo, existing e2e),
`require_api_key` is a no-op: every endpoint stays exactly as open as it is
today. When set, it requires a matching key on whichever endpoints declare it
as a dependency, via either an `X-API-Key` header or an `Authorization:
Bearer <key>` header — a missing/incorrect key gets a 401.

Read-only public endpoints (health, search, pipeline, galaxy,
preprocess-preview, specimen images) intentionally do NOT depend on this and
stay open regardless of the env var, per the hardening scope.

**Per-user identity (`resolve_user` / `issue_token`).** When `FLORALENS_JWT_SECRET`
is unset (the default), the app stays single-user: every request resolves to
the `DEFAULT_USER` sentinel, byte-for-byte the same behavior as before this
scaffold existed. Setting the secret turns on a minimal HS256 JWT identity
check (`Authorization: Bearer <jwt>`, `sub` claim = user id) that
`/api/garden`, `/api/memory`, and `/api/assistant` use to scope data per user.
This module intentionally does NOT implement login/password/OAuth — see
`issue_token`'s docstring and the `/api/auth/token` endpoint in `main.py` for
the (dev-only) scaffold used to mint tokens until a real login flow lands.
"""
from __future__ import annotations

import hmac
import os
import time

from fastapi import HTTPException, Request

# Owner of all data when per-user auth is off (no FLORALENS_JWT_SECRET) or a
# request has no other resolvable identity — i.e. today's single-user mode.
DEFAULT_USER = "public"


def _configured_api_key() -> str | None:
    # Read the env live (not cached at import time) so tests can toggle
    # FLORALENS_API_KEY per-test via monkeypatch.setenv without reloading the
    # app/module, and so an operator can rotate the key with a process env
    # change + restart rather than a code change.
    key = os.environ.get("FLORALENS_API_KEY", "")
    return key or None


def _extract_presented_key(request: Request) -> str | None:
    header_key = request.headers.get("X-API-Key")
    if header_key:
        return header_key
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[len("Bearer ") :].strip() or None
    return None


async def require_api_key(request: Request) -> None:
    """FastAPI dependency: enforces `FLORALENS_API_KEY` when one is configured.

    No-op when the env var is unset, so the local demo and existing tests
    (which never set it) keep working unchanged. Uses a constant-time compare
    to avoid leaking key material through response-timing differences.
    """
    expected = _configured_api_key()
    if expected is None:
        return
    presented = _extract_presented_key(request)
    # Compare as bytes: hmac.compare_digest raises TypeError on non-ASCII str
    # (Starlette decodes headers as latin-1), which would surface as a 500
    # instead of a clean 401 for a malformed key.
    if presented is None or not hmac.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def _configured_jwt_secret() -> str | None:
    # Read live (not cached at import time), same rationale as
    # _configured_api_key: tests toggle FLORALENS_JWT_SECRET per-test via
    # monkeypatch.setenv without reloading the module, and an operator can
    # rotate the secret with a process env change + restart.
    secret = os.environ.get("FLORALENS_JWT_SECRET", "")
    return secret or None


def jwt_auth_configured() -> bool:
    """Whether per-user JWT auth is turned on (FLORALENS_JWT_SECRET set)."""
    return _configured_jwt_secret() is not None


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[len("Bearer ") :].strip() or None
    return None


async def resolve_user(request: Request) -> str:
    """FastAPI dependency: resolves the caller's user id for data isolation.

    Secret unset (default) -> always `DEFAULT_USER`, no token required:
    single-user mode, identical to the app's behavior before this scaffold.
    Secret set -> requires a valid `Authorization: Bearer <jwt>` HS256 token
    signed with that secret; the token's `sub` claim IS the user id. Missing,
    malformed, unsigned/wrong-signature, or expired tokens all raise a 401 —
    never silently fall back to DEFAULT_USER once auth is turned on, since
    that would let an unauthenticated caller read/write the default bucket.
    """
    secret = _configured_jwt_secret()
    if secret is None:
        return DEFAULT_USER

    import jwt  # lazy: module import must not require PyJWT when auth is off

    token = _extract_bearer_token(request)
    if token is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="invalid or expired token") from exc
    sub = payload.get("sub")
    if not sub or not isinstance(sub, str):
        raise HTTPException(status_code=401, detail="token missing 'sub' claim")
    return sub


def issue_token(user_id: str, expires_in_s: int = 86400) -> str:
    """Sign a minimal HS256 JWT (`sub`=user_id, `iat`/`exp`) for `resolve_user`
    to accept.

    **Dev scaffold only** — there is no login/password/OAuth check here or in
    `/api/auth/token` (main.py), which is this function's only caller today;
    anyone able to invoke it can mint a token for ANY user_id. It exists so
    the per-user auth/data-isolation plumbing (resolve_user, garden/memory
    scoping) can be exercised end-to-end before a real login/OAuth round
    replaces it. Raises if no secret is configured — never issues an
    unsigned or default-secret token.
    """
    secret = _configured_jwt_secret()
    if secret is None:
        raise RuntimeError(
            "cannot issue a token: FLORALENS_JWT_SECRET is not configured"
        )

    import jwt  # lazy: module import must not require PyJWT when auth is off

    now = int(time.time())
    payload = {"sub": user_id, "iat": now, "exp": now + expires_in_s}
    return jwt.encode(payload, secret, algorithm="HS256")
