from utils import UnitTestConfig, async_test_case

from funcoder.langrt.types import LrtFunctionDef
from funcoder.methods.funcoder import (
    FuncoderGen,
    FuncoderHumanEvalConquerPrompt,
    FuncoderHumanEvalDividePrompt,
    FuncoderHumanEvalFuncCallPrompt,
    FuncoderSysTestArgsPrompt,
)


@async_test_case
async def test_methods_funcoder_gen() -> None:
    UTC = UnitTestConfig()
    if not UTC.test_llm():
        return

    problem_code = '''
from typing import List
        
def are_primes(args: List[int]) -> List[bool]:
    """Checks if a list of numbers are prime. Returns a boolean for each number
    in the input list. Examples:

    >>> are_primes([])
    []
    >>> are_primes([2, 3, 6, 7, 9])
    [True, True, False, True, False]
    """
    raise NotImplementedError()
'''
    ctx = UTC.mk_code_gen_ctx()
    inp_prog = ctx.lrt.parse(module=(), code=problem_code)
    func = inp_prog.find(LrtFunctionDef, "are_primes")
    assert func is not None
    ancestors = inp_prog.excluding(func)

    gen = FuncoderGen(
        dfs_max_depth=5,
        # divide
        divide_gen_prompt=FuncoderHumanEvalDividePrompt(),
        divide_temperature=0.2,
        divide_retries=5,
        # functional consistency
        fc_root_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
        fc_root_sys_test_prompt=FuncoderSysTestArgsPrompt(),
        fc_branch_test_prompt=FuncoderHumanEvalFuncCallPrompt(),
        fc_branch_sys_test_prompt=None,
        fc_temperature=0.2,
        fc_retries=5,
        # conquer
        conquer_gen_prompt=FuncoderHumanEvalConquerPrompt(),
        conquer_temperature=0.2,
        conquer_samples=11,
        conquer_min_samples=5,
        conquer_retries=5,
    )
    program, journal = await gen.gen(ctx, ancestors, func, [])
    assert program is not None
    assert bool(journal)

    async def _assert_result(args: list[int], expected: list[bool]):
        result = await ctx.lrt.run_program(program=program, func_name="are_primes", args=[args], kwargs={})
        assert result.result == expected

    await _assert_result([], [])
    await _assert_result([2, 3, 6, 7, 9], [True, True, False, True, False])
    await _assert_result([7, 11, 13, 8], [True, True, True, False])
    await _assert_result([997], [True])
