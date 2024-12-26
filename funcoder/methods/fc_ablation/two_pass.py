from ...langrt import LrtFunctionDef, LrtNode, LrtProgram
from ..codet.gen import runner_score_codet
from ..funcoder.dfs_2pass import FunCoderDfs2Pass
from ..funcoder.gen import amend_func_samples_for_sys_tests, is_root_func
from ..funcoder.gen_once import GenOncePrompt, funcoder_gen_once, gen_collect_program
from ..funcoder.make_test import MakeTestPrompt, TestType, funcoder_make_test
from ..funcoder.runner import RunnerScoreSig, funcoder_runner
from ..shared import CodeGenContext, CodeGenJournal, CodeGenJournalist, CodeGenMethod


class FcAblationTwoPassImpl(CodeGenMethod):
    def __init__(
        self,
        # dfs mechanism
        dfs_max_depth: int,
        # divide
        divide_gen_prompt: GenOncePrompt,
        divide_temperature: float,
        divide_retries: int,
        # testing
        ts_method: RunnerScoreSig,
        ts_root_test_prompt: MakeTestPrompt | None,
        ts_root_sys_test_prompt: MakeTestPrompt | None,
        ts_branch_test_prompt: MakeTestPrompt | None,
        ts_branch_sys_test_prompt: MakeTestPrompt | None,
        ts_temperature: float,
        ts_retries: int,
        # conquer
        conquer_gen_prompt: GenOncePrompt,
        conquer_temperature: float,
        conquer_samples: int,
        conquer_min_samples: int,
        conquer_retries: int,
    ):
        self.dfs_max_depth = dfs_max_depth
        self.divide_gen_prompt = divide_gen_prompt
        self.divide_temperature = divide_temperature
        self.divide_retries = divide_retries
        self.ts_method = ts_method
        self.ts_root_test_prompt = ts_root_test_prompt
        self.ts_root_sys_test_prompt = ts_root_sys_test_prompt
        self.ts_branch_test_prompt = ts_branch_test_prompt
        self.ts_branch_sys_test_prompt = ts_branch_sys_test_prompt
        self.ts_temperature = ts_temperature
        self.ts_retries = ts_retries
        self.conquer_gen_prompt = conquer_gen_prompt
        self.conquer_temperature = conquer_temperature
        self.conquer_samples = conquer_samples
        self.conquer_min_samples = conquer_min_samples
        self.conquer_retries = conquer_retries

    async def gen(
        self,
        ctx: CodeGenContext,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ) -> tuple[LrtProgram | None, CodeGenJournal]:
        ctx.log.in_scope(f"fc_ablation_2pass[{func.name}]")
        init_ancestors = ancestors
        init_func = func

        dfs_2p = FunCoderDfs2Pass(
            ctx=ctx,
            opt_max_depth=self.dfs_max_depth,
            opt_refine_leaf=True,
            opt_patch_refine_root_docstring=True,
            gen_pass_1=lambda _ctx, _anc, _func, _desc: self._pass_1(
                ctx=_ctx, ancestors=_anc, func=_func, descendants=_desc
            ),
            gen_pass_2=lambda _ctx, _anc, _func, _desc: self._pass_2(
                ctx=_ctx,
                init_ancestors=init_ancestors,
                init_func=init_func,
                ancestors=_anc,
                func=_func,
                descendants=_desc,
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

    async def _pass_2(
        self,
        ctx: CodeGenContext,
        init_ancestors: list[LrtNode],
        init_func: LrtFunctionDef,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ):
        return await funcoder_runner(
            ctx=ctx,
            opt_include_architect=True,
            opt_samples=self.conquer_samples,
            gen_pass=lambda _ctx, _anc, _func, _desc, _n: self._pass_2_gen(
                ctx=_ctx, ancestors=_anc, func=_func, descendants=_desc, n=_n
            ),
            test_pass=lambda _ctx, _anc, _func_samples: self._pass_2_test(
                ctx=_ctx,
                init_ancestors=init_ancestors,
                init_func=init_func,
                pass_func=func,
                pass_descendants=descendants,
                ancestors=_anc,
                func_samples=_func_samples,
            ),
            score_pass=self.ts_method,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )

    async def _pass_2_gen(
        self, ctx: CodeGenContext, ancestors: list[LrtNode], func: LrtFunctionDef, descendants: list[LrtNode], n: int
    ):
        return await funcoder_gen_once(
            ctx=ctx,
            opt_prompt=self.conquer_gen_prompt,
            opt_temperature=self.conquer_temperature,
            opt_samples=n,
            opt_min_samples=self.conquer_min_samples,
            opt_retries=self.conquer_retries,
            ancestors=ancestors,
            func=func,
            descendants=descendants,
        )

    async def _pass_2_test(
        self,
        ctx: CodeGenContext,
        init_ancestors: list[LrtNode],
        init_func: LrtFunctionDef,
        pass_func: LrtFunctionDef,
        pass_descendants: list[LrtNode],
        ancestors: list[LrtNode],
        func_samples: list[LrtFunctionDef],
    ):
        ctx.log.in_scope(f"ablation_pass_2_test")
        _sj = CodeGenJournalist(ctx, "ablation_pass_2_test", (ancestors, pass_func, pass_descendants))
        is_root = is_root_func(init_ancestors, init_func, ancestors, pass_func)
        ts_test_prompt = self.ts_root_test_prompt if is_root else self.ts_branch_test_prompt
        ts_sys_test_prompt = self.ts_root_sys_test_prompt if is_root else self.ts_branch_sys_test_prompt
        tests: list[tuple[TestType, LrtProgram, LrtFunctionDef]] = []

        # extract unit tests from requirements
        if ts_sys_test_prompt is not None:
            sys_tests, _sj_ch = await funcoder_make_test(
                ctx=ctx,
                opt_prompt=ts_sys_test_prompt,
                opt_temperature=self.ts_temperature,
                opt_retries=self.ts_retries,
                ancestors=ancestors,
                func_samples=amend_func_samples_for_sys_tests(ctx, pass_func, func_samples),
            )
            _sj.append(_sj_ch)
            tests.extend(sys_tests)

        # collect self-tests (no ground truth)
        if ts_test_prompt is not None:
            self_tests, _sj_ch = await funcoder_make_test(
                ctx=ctx,
                opt_prompt=ts_test_prompt,
                opt_temperature=self.ts_temperature,
                opt_retries=self.ts_retries,
                ancestors=ancestors,
                func_samples=func_samples,
            )
            _sj.append(_sj_ch)
            tests.extend(self_tests)

        return tests, _sj.collect_test(tests)

    pass


class FcAblationTwoPassGen(FcAblationTwoPassImpl):
    """DFS with only the 'divide' & 'conquer' stage (a.k.a. pass #1 + #2)."""

    def __init__(
        self,
        # dfs mechanism
        dfs_max_depth: int,
        # divide
        divide_gen_prompt: GenOncePrompt,
        divide_temperature: float,
        divide_retries: int,
        # conquer
        conquer_gen_prompt: GenOncePrompt,
        conquer_temperature: float,
        conquer_samples: int,
        conquer_min_samples: int,
        conquer_retries: int,
    ):
        super().__init__(
            dfs_max_depth=dfs_max_depth,
            divide_gen_prompt=divide_gen_prompt,
            divide_temperature=divide_temperature,
            divide_retries=divide_retries,
            ts_method=runner_score_codet,
            ts_root_test_prompt=None,
            ts_root_sys_test_prompt=None,
            ts_branch_test_prompt=None,
            ts_branch_sys_test_prompt=None,
            ts_temperature=0.0,
            ts_retries=5,
            conquer_gen_prompt=conquer_gen_prompt,
            conquer_temperature=conquer_temperature,
            conquer_samples=conquer_samples,
            conquer_min_samples=conquer_min_samples,
            conquer_retries=conquer_retries,
        )

    pass
