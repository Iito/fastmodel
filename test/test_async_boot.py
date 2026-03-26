"""Tests for the async model boot wrapper."""

import asyncio
import logging

import pytest

from fastmodel.utils.async_model import create_async_boot
from test.conftest import ProtocolModel, LegacyModel, SimpleInput


class TestAsyncBoot:
    def test_not_ready_before_boot(self):
        logger = logging.getLogger("test")
        instance = create_async_boot(ProtocolModel, logger, None)
        assert instance.get() is None
        assert instance.ready is False

    @pytest.mark.asyncio
    async def test_ready_after_boot(self):
        logger = logging.getLogger("test")
        instance = create_async_boot(ProtocolModel, logger, None)
        await instance.async_boot()
        assert instance.ready is True
        assert instance.get() is instance

    @pytest.mark.asyncio
    async def test_model_callable_after_boot(self):
        logger = logging.getLogger("test")
        instance = create_async_boot(ProtocolModel, logger, None)
        await instance.async_boot()

        model = instance.get()
        result = model(SimpleInput(text="test", count=2))
        assert result.result == "testtest"
        assert result.length == 8

    @pytest.mark.asyncio
    async def test_legacy_model_boots(self):
        logger = logging.getLogger("test")
        instance = create_async_boot(LegacyModel, logger, None)
        await instance.async_boot()
        assert instance.ready is True

        model = instance.get()
        result = model(SimpleInput(text="hi"))
        assert result.result == "hi"
        assert result.length == 2

    def test_warmup_called_if_present(self):
        warmup_called = False

        class ModelWithWarmup:
            def __init__(self):
                pass

            def warmup(self):
                nonlocal warmup_called
                warmup_called = True

            def __call__(self, input):
                pass

        logger = logging.getLogger("test")
        instance = create_async_boot(ModelWithWarmup, logger, None)
        asyncio.get_event_loop().run_until_complete(instance.async_boot())
        assert warmup_called is True

    def test_model_dir_passed_to_init(self):
        received_dir = None

        class ModelWithDir:
            def __init__(self, model_dir: str = None):
                nonlocal received_dir
                received_dir = model_dir

            def __call__(self, input):
                pass

        logger = logging.getLogger("test")
        instance = create_async_boot(ModelWithDir, logger, "/tmp/models")
        asyncio.get_event_loop().run_until_complete(instance.async_boot())
        assert received_dir == "/tmp/models"
