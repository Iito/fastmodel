import asyncio
import logging
from pathlib import Path


def create_async_boot(model_class: object, logger: logging, model_dir: str | Path):
    kwargs = {}
    if hasattr(model_class.__init__, "__annotations__"):
        for arg in model_class.__init__.__annotations__:  # type: ignore
            if "logger" in arg:
                kwargs[arg] = logger
            if "model_path" in arg:
                kwargs[arg] = model_dir
            if "model_dir" in arg:
                kwargs[arg] = model_dir
            if arg == "model":
                kwargs[arg] = model_dir

    class AsyncBoot(model_class):
        def __init__(self, **kwargs):
            self.ready = False
            self._kwargs = kwargs

            self.logger = logger
            self._class_name = model_class.__qualname__

        async def async_boot(self):
            await asyncio.to_thread(self._boot)
            if hasattr(self, "warmup"):
                await asyncio.to_thread(self.warmup)


        def _boot(self):
            self.logger.info(f"Booting {self._class_name}")
            super().__init__(**self._kwargs)
            self.ready = True
            self.logger.info(f"{self._class_name} is ready")

        def get(self):
            if self.ready:
                return self
            else:
                return None

    return AsyncBoot(**kwargs)
