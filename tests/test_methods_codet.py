from utils import UnitTestConfig, async_test_case

from funcoder.langrt.types import LrtFunctionDef
from funcoder.methods.codet.gen import CodeTGen
from funcoder.methods.funcoder import FuncoderHumanEvalDividePrompt, FuncoderHumanEvalUnitTestPrompt


@async_test_case
async def test_methods_codet_gen() -> None:
    UTC = UnitTestConfig()
    if not UTC.test_llm():
        return

    problem_code = '''
def longest_common_prefix(strs: list[str]) -> str:
    """Given a list of strings, return the longest common prefix (of all the
    given strings). If there's nothing to compare, return an empty string."""

    raise NotImplementedError()
'''
    ctx = UTC.mk_code_gen_ctx()
    inp_prog = ctx.lrt.parse(module=(), code=problem_code)
    func = inp_prog.find(LrtFunctionDef, "longest_common_prefix")
    assert func is not None
    ancestors = inp_prog.excluding(func)

    gen = CodeTGen(
        gen_prompt=FuncoderHumanEvalDividePrompt(),
        temperature=0.2,
        samples=11,
        min_samples=5,
        retries=5,
        ut_test_prompt=FuncoderHumanEvalUnitTestPrompt(),
        ut_temperature=0.2,
        ut_retries=5,
    )
    program, journal = await gen.gen(ctx, ancestors, func, [])
    assert program is not None
    assert bool(journal)

    async def _assert_result(args: list[str], expected: str):
        result = await ctx.lrt.run_program(program=program, func_name="longest_common_prefix", args=[args], kwargs={})
        assert result.result == expected

    await _assert_result([], "")
    await _assert_result(["flower", "flow", "flight"], "fl")
    await _assert_result(["dog", "racer", "car"], "")
