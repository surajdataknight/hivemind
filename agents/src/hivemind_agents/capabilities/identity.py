"""c1 session & identity verifier: checks the app's session token and mints the case token for this chat.

fails closed: a token that does not verify, or a customer the business does not know, gets no case token.
"""

from datetime import datetime, timedelta, timezone

import jwt

from hivemind_agents.contracts import Identity

ALGORITHM = "HS256"
SESSION_ISSUER = "hivemart-app"
SESSION_AUDIENCE = "hivemind-support"
CASE_ISSUER = "hivemind"
CASE_AUDIENCE = "simulated-business"
CASE_TOKEN_TTL = timedelta(minutes=10)

# what this build's capabilities read and write; a capability can never use more than the token carries
SCOPES = ["profile:read:self", "orders:read:self", "delivery:read:self", "payments:read:self", "inventory:read",
          "promotions:read:self", "cases:write:self", "policy:read", "history:read:self", "risk:read"]


class NotSignedIn(Exception):
    pass


class IdentityVerifier:
    def __init__(self, session_key: str, case_key: str):
        self._session_key = session_key
        self._case_key = case_key

    def verify_session(self, session_token: str) -> tuple[str, str]:
        """(customer id, session id) from a valid session token."""
        try:
            claims = jwt.decode(session_token, self._session_key, algorithms=[ALGORITHM],
                                audience=SESSION_AUDIENCE, issuer=SESSION_ISSUER,
                                options={"require": ["exp", "iat", "sub", "sid"]})
        except jwt.ExpiredSignatureError:
            raise NotSignedIn("the session has expired; sign in again") from None
        except jwt.InvalidTokenError:
            raise NotSignedIn("the session token is not valid") from None
        return claims["sub"], claims["sid"]

    def mint_case_token(self, customer_id: str, session_id: str, conversation_id: str,
                        now: datetime | None = None) -> tuple[Identity, str]:
        now = now or datetime.now(timezone.utc)
        claims = {"iss": CASE_ISSUER, "aud": CASE_AUDIENCE, "sub": customer_id, "sid": session_id,
                  "conv": conversation_id, "scopes": SCOPES, "iat": now, "exp": now + CASE_TOKEN_TTL}
        token = jwt.encode(claims, self._case_key, algorithm=ALGORITHM)
        return Identity(customer_id=customer_id, session_id=session_id, scopes=SCOPES), token
