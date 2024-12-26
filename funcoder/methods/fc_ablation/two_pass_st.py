from ..codet.gen import runner_score_codet
from ..funcoder.gen_once import GenOncePrompt
from ..funcoder.make_test import MakeTestPrompt
from .two_pass import FcAblationTwoPassImpl


class FcAblationTwoPassSelfTestGen(FcAblationTwoPassImpl):
    """On top of the traditional 2-pass, apply self-testing. Validate using
    'whether the function passed the tests or not'."""

    def __init__(
        self,
        # dfs mechanism
        dfs_max_depth: int,
        # divide
        divide_gen_prompt: GenOncePrompt,
        divide_temperature: float,
        divide_retries: int,
        # testing
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
        super().__init__(
            dfs_max_depth=dfs_max_depth,
            divide_gen_prompt=divide_gen_prompt,
            divide_temperature=divide_temperature,
            divide_retries=divide_retries,
            ts_method=runner_score_codet,
            ts_root_test_prompt=ts_root_test_prompt,
            ts_root_sys_test_prompt=ts_root_sys_test_prompt,
            ts_branch_test_prompt=ts_branch_test_prompt,
            ts_branch_sys_test_prompt=ts_branch_sys_test_prompt,
            ts_temperature=ts_temperature,
            ts_retries=ts_retries,
            conquer_gen_prompt=conquer_gen_prompt,
            conquer_temperature=conquer_temperature,
            conquer_samples=conquer_samples,
            conquer_min_samples=conquer_min_samples,
            conquer_retries=conquer_retries,
        )

    pass
