import hashlib
import json

from ..codet.gen import runner_score_codet
from ..funcoder.gen_once import GenOncePrompt
from ..funcoder.make_test import MakeTestPrompt
from ..funcoder.runner import RunnerCaseResult
from .two_pass import FcAblationTwoPassImpl


class FcAblationTwoPassClusteringGen(FcAblationTwoPassImpl):
    """On top of the traditional 2-pass, apply AlphaCode-like clustering."""

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


def runner_score_clustering(results: list[list[RunnerCaseResult]]) -> list[float]:
    """AlphaCode clusters programs exactly by the identical-ness of outputs."""

    signatures = list[str]()
    for program in results:
        sig = []
        for case in program:
            sig.append(case.result if case.ok else {"error": "<ERROR>"})
        sig_s = json.dumps(sig, indent=None, ensure_ascii=True, sort_keys=True)
        sig_h = hashlib.sha256(sig_s.encode("utf-8", "ignore")).hexdigest()
        signatures.append(sig_h)

    # the larger the cluster is, the more correct we think it is
    count = dict[str, int]()
    for sig in signatures:
        if sig not in count:
            count[sig] = 0
        count[sig] += 1
    scores = [float(count[sig]) for sig in signatures]
    return scores
