# This is a modified version of uvicorn.main module from the Uvicorn project.
# It is used to run the FastAPI application.
# Instead of reading the FastAPI application from an existing python file or package it creates it on the fly.
# Then it proceed to call the run function from the uvicorn module.
import asyncio
import importlib.metadata
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from uvicorn import run as uvicorn_run
from uvicorn.config import (
    HTTPProtocolType,
    InterfaceType,
    LifespanType,
    LoopSetupType,
    WSProtocolType,
)

from .config import APP_AUTH_TOKEN, MODEL_DIR, ROOT_PATH
from .exceptions import HandledException
from .fastapi.io import create_request_model, create_response_model
from .status_code import StatusCode
from .utils.async_model import create_async_boot
from .utils.importer import import_from_string


def run(
    model: str,
    host: str,
    port: int,
    uds: str,
    fd: int,
    loop: LoopSetupType,
    http: HTTPProtocolType,
    ws: WSProtocolType,
    ws_max_size: int,
    ws_ping_interval: float,
    ws_ping_timeout: float,
    ws_per_message_deflate: bool,
    lifespan: LifespanType,
    interface: InterfaceType,
    reload: bool,
    reload_dirs: list[str],
    reload_includes: list[str],
    reload_excludes: list[str],
    reload_delay: float,
    workers: int,
    env_file: str,
    log_config: str,
    log_level: str,
    access_log: bool,
    proxy_headers: bool,
    server_header: bool,
    date_header: bool,
    forwarded_allow_ips: str,
    limit_concurrency: int,
    backlog: int,
    limit_max_requests: int,
    timeout_keep_alive: int,
    timeout_graceful_shutdown: int | None,
    ssl_keyfile: str,
    ssl_certfile: str,
    ssl_keyfile_password: str,
    ssl_version: int,
    ssl_cert_reqs: int,
    ssl_ca_certs: str,
    ssl_ciphers: str,
    headers: list[str],
    use_colors: bool,
    app_dir: str,
    h11_max_incomplete_event_size: int | None,
    factory: bool,
    log_format: str,
    root_path: str = ROOT_PATH,
    handledExceptionClass: Exception = HandledException,
    successStatusCode: StatusCode = StatusCode.Success,
    errorStatusCode: StatusCode = StatusCode.Error,
    debug: bool = False,
    force_json: bool = False,
    doc_url: str = "/",
    redoc_url: str = None,
) -> None:
    """
    All args in this method are passed by the click command.
    Those are the original args from the uvicorn.run method except for the model arg that replaced the 'app' arg.
    Uvicorn version 0.22.0
    """
    dependencies = []
    if APP_AUTH_TOKEN:
        from .security import TokenAuthScheme

        tos = TokenAuthScheme(APP_AUTH_TOKEN)
        dependencies = [Depends(tos.get_token_header)]

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Discovering model")
    (
        MyModel,
        (MyModelInputArg, MyModelInput),
        (_, MyModelOutput),
    ) = import_from_string(model)

    logger.debug(f"Model: {MyModel.__name__}")
    logger.debug(f"ModelInputArg: {MyModelInputArg}")
    logger.debug(f"ModelInput: {MyModelInput.__name__}")
    logger.debug(f"ModelOutput: {MyModelOutput.__name__}")

    def get_logger():
        return logger

    modelInstance = create_async_boot(MyModel, logger, MODEL_DIR)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        logger.info(f"Starting {MyModel.__name__}'s service")
        logger.info(f"Waiting for {MyModel.__name__} to load")
        asyncio.create_task(modelInstance.async_boot())
        yield

        logger.info("Shutting requested")
        logger.info("Shutting down")

    model_version = MyModel.version() if hasattr(MyModel, "version") else None
    if model_version is None:
        model_version = importlib.metadata.version(model.split(".")[0]).split(".")[:3]
        model_version = ".".join(model_version)

    app = FastAPI(
        lifespan=_lifespan,
        title=f"{MyModel.__name__}",
        description=MyModel.__annotations__,
        redoc_url=redoc_url,
        docs_url=doc_url,
        version=model_version or "0.0.0",
        root_path=ROOT_PATH,
    )

    # This create a response model with optional fields
    modelResponse = create_response_model(MyModelOutput)
    modelResponse.force_json = force_json
    modelRequest = create_request_model(MyModelInput)

    modelRequest.returns = MyModelInput
    modelResponse._version = model_version

    description = modelResponse.model_json_schema()

    @app.post(
        "/",
        tags=["BackBone"],
        dependencies=dependencies,
        responses={
            200: {
                "description": f"{MyModel.__name__} response",
                "content": {
                    "application/json": {
                        "example": description,
                    }
                },
            }
        },
    )
    def predict(
        model_input: modelRequest = Depends(modelRequest._validate),
        logger=Depends(get_logger),
        model=Depends(modelInstance.get),
    ) -> modelResponse:
        """
        Predict the output
        """

        response = modelResponse(status=0)
        if not model:
            response.status = StatusCode.ModelNotReady
            response.message = StatusCode.ModelNotReady.msg
            return response.generate_streaming_response()
        try:
            logger.info(f"Processing")
            _response = modelResponse(status=0)
            try:
                model_args = {MyModelInputArg: model_input}
                _response: MyModelOutput = model(**model_args)
            except ValueError as e:
                logger.error(f"Invalid input: {e}")
                _response.status = StatusCode.InvalidInput
                _response.message = StatusCode.InvalidInput.msg
            except TypeError as e:
                logger.exception(f"Invalid input {e}")
                _response.status = StatusCode.InvalidInput
                _response.message = StatusCode.InvalidInput.msg
            # except TypeError as e:
            #     model_args = {MyModelInputArg: model_input.process()}
            #     _response: MyModelOutput = model(**model_args)
            # except ValueError as e:
            #     model_args = {MyModelInputArg: model_input.process()}
            #     _response: MyModelOutput = model(**model_args)

            response.status = successStatusCode
            response.message = successStatusCode.msg
            for key, value in _response.model_dump().items():
                setattr(response, key, value)
        except handledExceptionClass as e:
            # This is handled exception,
            logger.error(f"{handledExceptionClass.__class__.__name__} occured: {e}")
            response.status = e.status
            response.message = e.message
        except Exception as e:
            logger.exception(f"Exception occured: {e}")
            response.status = errorStatusCode
            response.message = errorStatusCode.msg

        return response.generate_streaming_response()

    uvicorn_run(
        app,
        host=host,
        port=port,
        uds=uds,
        fd=fd,
        loop=loop,
        http=http,
        ws=ws,
        ws_max_size=ws_max_size,
        ws_ping_interval=ws_ping_interval,
        ws_ping_timeout=ws_ping_timeout,
        ws_per_message_deflate=ws_per_message_deflate,
        lifespan=lifespan,
        env_file=env_file,
        log_config=log_config,
        log_level=log_level,
        access_log=access_log,
        interface=interface,
        reload=reload,
        reload_dirs=reload_dirs or None,
        reload_includes=reload_includes or None,
        reload_excludes=reload_excludes or None,
        reload_delay=reload_delay,
        workers=workers,
        proxy_headers=proxy_headers,
        server_header=server_header,
        date_header=date_header,
        forwarded_allow_ips=forwarded_allow_ips,
        root_path=root_path,
        limit_concurrency=limit_concurrency,
        backlog=backlog,
        limit_max_requests=limit_max_requests,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_cert_reqs=ssl_cert_reqs,
        ssl_ca_certs=ssl_ca_certs,
        ssl_ciphers=ssl_ciphers,
        headers=[header.split(":", 1) for header in headers],
        use_colors=use_colors,
        factory=factory,
        app_dir=app_dir,
        h11_max_incomplete_event_size=h11_max_incomplete_event_size,
    )
