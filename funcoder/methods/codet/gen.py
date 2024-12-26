from ...langrt import LrtFunctionDef, LrtNode, LrtProgram
from ..funcoder.gen_once import GenOncePrompt, funcoder_gen_once, gen_collect_program
from ..funcoder.make_test import MakeTestPrompt, funcoder_make_test
from ..funcoder.runner import RunnerCaseResult, funcoder_runner
from ..shared import CodeGenContext, CodeGenJournal, CodeGenMethod


class CodeTGen(CodeGenMethod):
    """FunCoder is an efficient method of generating code iteratively."""

    def __init__(
        self,
        gen_prompt: GenOncePrompt,
        temperature: float,
        samples: int,
        min_samples: int,
        retries: int,
        ut_test_prompt: MakeTestPrompt,
        ut_temperature: float,
        ut_retries: int,
    ):
        self.gen_prompt = gen_prompt
        self.temperature = temperature
        self.samples = samples
        self.min_samples = min_samples
        self.retries = retries
        self.ut_test_prompt = ut_test_prompt
        self.ut_temperature = ut_temperature
        self.ut_retries = ut_retries

    async def gen(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ) -> tuple[LrtProgram | None, CodeGenJournal]:
        ctx.log.in_scope(f"codet[{func.name}]")

        _results, journal = await funcoder_runner(
            ctx=ctx,
            opt_include_architect=True,
            opt_samples=self.samples,
            gen_pass=lambda _ctx, _anc, _func, _desc, _n: funcoder_gen_once(
                ctx=_ctx,
                opt_prompt=self.gen_prompt,
                opt_temperature=self.temperature,
                opt_samples=_n,
                opt_min_samples=self.min_samples,
                opt_retries=self.retries,
                ancestors=_anc,
                func=_func,
                descendants=_desc,
            ),
            test_pass=lambda _ctx, _anc, _func_samples: funcoder_make_test(
                ctx=_ctx,
                opt_prompt=self.ut_test_prompt,
                opt_temperature=self.ut_temperature,
                opt_retries=self.ut_retries,
                ancestors=_anc,
                func_samples=_func_samples,
            ),
            score_pass=runner_score_codet,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )
        if _results is None:
            return None, journal
        func_impl, rest_impl = _results

        program = gen_collect_program(ctx, ancestors, func_impl, rest_impl, descendants)
        ctx.log.code("python", "final result", ctx.lrt.fmt(program))
        return program, journal

    pass


def runner_score_codet(results: list[list[RunnerCaseResult]]) -> list[float]:
    """CodeT has his own ideas. Pass UT and it works and not otherwise."""

    scores = list[float]()
    for program in results:
        aggregate = 0.0
        for case in program:
            aggregate += 1.0 if case.ok else 0.0
        scores.append(aggregate)
    return scores
