import hmac

from .fastapi import Depends, HTTPException
from .fastapi.security import HTTPBearer

token_auth_scheme = HTTPBearer(scheme_name="Authorization")


# The reason why hmac is used and not a == b is to protect it against
# timing attack, more here: https://sqreen.github.io/DevelopersSecurityBestPractices/timing-attack/python
class TokenAuthScheme:
    def __init__(self, secret: str):
        self.secret = secret

    async def get_token_header(self, token: str = Depends(token_auth_scheme)):
        if not hmac.compare_digest(token.credentials, self.secret):
            raise HTTPException(status_code=400, detail="Invalid Credentials")
