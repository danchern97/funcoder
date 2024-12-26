import asyncio
import pathlib
from typing import Any, Callable, Coroutine

from typing_extensions import ParamSpec

from funcoder import CodeGenContext, LangRT, LLMConfig, create_llm_engine, get_eval_config
from funcoder.utils.logger import Logger


class UnitTestConfig:
    """Configuration for unit tests and unit tests only."""

    def __init__(self):
        return

    def test_llm(self) -> bool:
        # WARNING: YOU WILL PAY FOR THIS (LITERALLY) IF SET TO TRUE
        return False

    def mk_llm_config(self) -> "LLMConfig":
        cfg = get_eval_config()
        llm_cfg = cfg.llm["for_unittest"]
        return llm_cfg

    def mk_code_gen_ctx(self) -> "CodeGenContext":
        cfg = get_eval_config()
        lrt_cfg = cfg.langrt["py3"]
        return CodeGenContext(
            log=Logger(),
            llm=create_llm_engine(self.mk_llm_config()),
            lrt=LangRT.python(
                sandbox_root=pathlib.Path(lrt_cfg.sandbox_root),
                parallelism=lrt_cfg.parallelism,
            ),
            cfg_silent=False,
        )

    pass


TArgs = ParamSpec("TArgs")


def async_test_case(fn: Callable[TArgs, Coroutine[Any, Any, None]]) -> Callable[TArgs, None]:
    """Decorate an async test case function to run it synchronously."""

    def _wrapper(*args, **kwargs) -> None:
        return asyncio.run(fn(*args, **kwargs))

    _wrapper.__name__ = fn.__name__
    return _wrapper
