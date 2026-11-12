"""who is calling: the fake sign-in's session token, and the case token every other api asks for.

session token: the made-up app signs a customer in and hands the chat screen this token. hivemind checks it.
case token: hivemind mints it for one customer's support chat, with the scopes it may use. the backend takes the
customer from this token, never from a path or a query, so a model-picked argument cannot reach another customer.
both are hs256 jwts; the two keys come from the environment (see example.env).
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, Header
from starlette.exceptions import HTTPException

ALGORITHM = "HS256"
SESSION_ISSUER = "hivemart-app"
SESSION_AUDIENCE = "hivemind-support"
SESSION_TTL = timedelta(hours=12)
CASE_ISSUER = "hivemind"
CASE_AUDIENCE = "simulated-business"
CASE_TOKEN_MAX_TTL = timedelta(minutes=30)
MIN_KEY_BYTES = 32


def signing_key(name: str) -> str:
    key = os.environ.get(name, "")
    if len(key.encode()) < MIN_KEY_BYTES:
        raise RuntimeError(f"{name} is not set or shorter than {MIN_KEY_BYTES} characters; see example.env")
    return key


def issue_session_token(customer_id: str, session_id: str, now: datetime | None = None) -> tuple[str, datetime]:
    now = now or datetime.now(timezone.utc)
    expires = now + SESSION_TTL
    claims = {"iss": SESSION_ISSUER, "aud": SESSION_AUDIENCE, "sub": customer_id, "sid": session_id,
              "iat": now, "exp": expires}
    return jwt.encode(claims, signing_key("HM_SESSION_SECRET"), algorithm=ALGORITHM), expires


@dataclass(frozen=True)
class Caller:
    customer_id: str
    session_id: str
    scopes: frozenset[str]


def unauthenticated(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail)


def read_case_token(token: str) -> Caller:
    try:
        claims = jwt.decode(token, signing_key("HM_CASE_TOKEN_SECRET"), algorithms=[ALGORITHM],
                            audience=CASE_AUDIENCE, issuer=CASE_ISSUER,
                            options={"require": ["exp", "iat", "sub", "sid", "scopes"]})
    except jwt.ExpiredSignatureError:
        raise unauthenticated("the case token has expired") from None
    except jwt.InvalidTokenError:
        raise unauthenticated("the case token is not valid") from None
    if claims["exp"] - claims["iat"] > CASE_TOKEN_MAX_TTL.total_seconds():
        raise unauthenticated("the case token lives longer than 30 minutes")
    scopes = claims["scopes"]
    if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
        raise unauthenticated("the case token is not valid")
    return Caller(customer_id=claims["sub"], session_id=claims["sid"], scopes=frozenset(scopes))


def caller(authorization: Annotated[str | None, Header()] = None) -> Caller:
    if not authorization or not authorization.startswith("Bearer "):
        raise unauthenticated("send the case token as `Authorization: Bearer <token>`")
    return read_case_token(authorization.removeprefix("Bearer "))


def scope(needed: str):
    """a dependency that lets the call through only when the case token carries this scope."""

    def check(who: Annotated[Caller, Depends(caller)]) -> Caller:
        if needed not in who.scopes:
            raise HTTPException(status_code=403, detail=f"the case token lacks the scope {needed}")
        return who

    return Depends(check)
