# This is a base router that provides basic health and readiness checks
# as well as a redirect to the latest version of the documentation.
# It is intended to be used as a base for other routers
# .
import importlib

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()
__version__ = importlib.import_module("version").__version__

router.init = False


@router.get("/info")
def info() -> dict[str, str]:
    return {"version": __version__}


@router.get("/healthz")
def health() -> Response:
    return Response(status_code=200, content="Healthy")


@router.get("/readyz")
def ready(model=router.dependencies[0].dependency()) -> dict[str, str]:
    # If model is not ready the model == None
    if model:
        resp = Response(status_code=200, content="Ready")
    else:
        resp = Response(status_code=503, content="Not ready yet")
    return resp
