from ...langrt import LrtFunctionDef, LrtNode, LrtProgram
from ..funcoder.dfs_1pass import FunCoderDfs1Pass
from ..funcoder.gen_once import GenOncePrompt, funcoder_gen_once, gen_collect_program
from ..shared import CodeGenContext, CodeGenJournal, CodeGenMethod


class FcAblationOnePassGen(CodeGenMethod):
    """DFS with only the 'divide' stage (a.k.a. pass #1)."""

    def __init__(
        self,
        # dfs mechanism
        dfs_max_depth: int,
        # divide
        divide_gen_prompt: GenOncePrompt,
        divide_temperature: float,
        divide_retries: int,
    ):
        self.dfs_max_depth = dfs_max_depth
        self.divide_gen_prompt = divide_gen_prompt
        self.divide_temperature = divide_temperature
        self.divide_retries = divide_retries

    async def gen(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ) -> tuple[LrtProgram | None, CodeGenJournal]:
        ctx.log.in_scope(f"fc_ablation_1pass[{func.name}]")

        dfs_2p = FunCoderDfs1Pass(
            ctx=ctx,
            opt_max_depth=self.dfs_max_depth,
            gen_pass_1=lambda _ctx, _anc, _func, _desc: self._pass_1(
                ctx=_ctx, ancestors=_anc, func=_func, descendants=_desc
            ),
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )

        _results, journal = await dfs_2p.run()
        if _results is None:
            return None, journal
        func_impl, rest_impl = _results

        program = gen_collect_program(ctx, ancestors, func_impl, rest_impl, descendants)
        ctx.log.code("python", "final result", ctx.lrt.fmt(program))
        return program, journal

    async def _pass_1(
        self, ctx: CodeGenContext, ancestors: list[LrtNode], func: LrtFunctionDef, descendants: list[LrtNode]
    ):
        samples, sj = await funcoder_gen_once(
            ctx=ctx,
            opt_prompt=self.divide_gen_prompt,
            opt_temperature=self.divide_temperature,
            opt_samples=1,
            opt_min_samples=1,
            opt_retries=self.divide_retries,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )
        return samples[0] if samples else None, sj

    pass
