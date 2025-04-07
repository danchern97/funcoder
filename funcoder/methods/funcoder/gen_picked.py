from typing import Literal

from ...utils.types import guard_never
from ..shared import CodeGenMethod
from .gen import FuncoderGen
from .prompts.humaneval import (
    FuncoderHumanEvalConquerPrompt,
    FuncoderHumanEvalDividePrompt,
    FuncoderHumanEvalFuncCallPrompt,
)
from .prompts.injected import FuncoderInjectedStdioPrompt
from .prompts.maths import FuncoderMathsConquerPrompt, FuncoderMathsDividePrompt
from .prompts.sys_test import FuncoderSysTestArgsPrompt
from .prompts.xcodeeval import FuncoderXCodeEvalConquerPrompt, FuncoderXCodeEvalDividePrompt


class FuncoderCherryGen(CodeGenMethod):
    """Funcoder is an efficient method of generating code iteratively. This
    class contains the cherry-picked defaults for this method."""

    def __init__(self, task: Literal["humanevalplus", "maths", "mbppplus", "xcodeeval", "livecodebench"]):
        self.task = task
        if task in ["humanevalplus", "mbppplus"]:
            self._impl = FuncoderGen(
                dfs_max_depth=6,
                divide_gen_prompt=FuncoderHumanEvalDividePrompt(),
                divide_temperature=0.2,
                divide_retries=5,
                fc_root_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_root_sys_test_prompt=FuncoderSysTestArgsPrompt(),
                fc_branch_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_branch_sys_test_prompt=None,
                fc_temperature=0.2,
                fc_retries=5,
                conquer_gen_prompt=FuncoderHumanEvalConquerPrompt(),
                conquer_temperature=0.8,
                conquer_samples=5,
                conquer_min_samples=10,
                conquer_retries=5,
            )
        elif task == "maths":
            self._impl = FuncoderGen(
                dfs_max_depth=6,
                divide_gen_prompt=FuncoderMathsDividePrompt(),
                divide_temperature=0.2,
                divide_retries=5,
                fc_root_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_root_sys_test_prompt=None,
                fc_branch_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_branch_sys_test_prompt=None,
                fc_temperature=0.2,
                fc_retries=5,
                conquer_gen_prompt=FuncoderMathsConquerPrompt(),
                conquer_temperature=0.8,
                conquer_samples=5,
                conquer_min_samples=10,
                conquer_retries=5,
            )
        elif task == "xcodeeval":
            self._impl = FuncoderGen(
                dfs_max_depth=6,
                divide_gen_prompt=FuncoderXCodeEvalDividePrompt(),
                divide_temperature=0.2,
                divide_retries=5,
                fc_root_test_prompt=None,
                # warning: stdio tests must be manually injected
                fc_root_sys_test_prompt=FuncoderInjectedStdioPrompt(),
                fc_branch_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_branch_sys_test_prompt=None,
                fc_temperature=0.2,
                fc_retries=5,
                conquer_gen_prompt=FuncoderXCodeEvalConquerPrompt(),
                conquer_temperature=0.8,
                conquer_samples=5,
                conquer_min_samples=10,
                conquer_retries=5,
            )
        elif task == "livecodebench":
            # Two implementations: one for stdio tests, one for function tests
            self._impl_stdio = FuncoderGen(
                dfs_max_depth=6,
                divide_gen_prompt=FuncoderXCodeEvalDividePrompt(),
                divide_temperature=0.2,
                divide_retries=5,
                fc_root_test_prompt=None,
                # warning: stdio tests must be manually injected
                fc_root_sys_test_prompt=FuncoderInjectedStdioPrompt(),
                fc_branch_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_branch_sys_test_prompt=None,
                fc_temperature=0.2,
                fc_retries=5,
                conquer_gen_prompt=FuncoderXCodeEvalConquerPrompt(),
                conquer_temperature=0.8,
                conquer_samples=5,
                conquer_min_samples=10,
                conquer_retries=5,
            )
            self._impl_func = FuncoderGen(
                dfs_max_depth=6,
                divide_gen_prompt=FuncoderHumanEvalDividePrompt(),
                divide_temperature=0.2,
                divide_retries=5,
                fc_root_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_root_sys_test_prompt=None,
                fc_branch_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
                fc_branch_sys_test_prompt=None,
                fc_temperature=0.2,
                fc_retries=5,
                conquer_gen_prompt=FuncoderHumanEvalConquerPrompt(),
                conquer_temperature=0.8,
                conquer_samples=5,
                conquer_min_samples=10,
                conquer_retries=5,
            )
        else:
            guard_never(task)

    async def gen(self, ctx, ancestors, func, descendants):
        if self.task == "livecodebench":
            if func.name == "main":
                return await self._impl_stdio.gen(ctx, ancestors, func, descendants)
            else:
                return await self._impl_func.gen(ctx, ancestors, func, descendants)
        else:
            return await self._impl.gen(ctx, ancestors, func, descendants)

    pass
