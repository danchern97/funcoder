from utils import UnitTestConfig, async_test_case

from funcoder.langrt.types import LrtFunctionDef
from funcoder.methods.funcoder import FuncoderHumanEvalDividePrompt
from funcoder.methods.vanilla.gen import VanillaGen


@async_test_case
async def test_methods_vanilla_gen() -> None:
    UTC = UnitTestConfig()
    if not UTC.test_llm():
        return

    problem_code = '''
def is_palindrome(s: str) -> bool:
    """Checks if a certain string is a palindrome."""
    raise NotImplementedError()
'''
    ctx = UTC.mk_code_gen_ctx()
    func = ctx.lrt.parse(module=(), code=problem_code)
    func = func.cast_as(LrtFunctionDef)

    gen = VanillaGen(
        gen_prompt=FuncoderHumanEvalDividePrompt(),
        temperature=0.0,
        retries=3,
    )
    program, journal = await gen.gen(ctx, [], func, [])
    assert program is not None
    assert bool(journal)

    result = await ctx.lrt.run_program(program=program, func_name="is_palindrome", args=["14641"], kwargs={})
    assert result.result is True
    result = await ctx.lrt.run_program(program=program, func_name="is_palindrome", args=["arg"], kwargs={})
    assert result.result is False
