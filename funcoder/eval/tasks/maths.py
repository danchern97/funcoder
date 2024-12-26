import json
import pathlib
import random
from typing import Generator

from typing_extensions import TypedDict

from ...langrt import LrtSolution
from ...llm import ChatMessage, LLMEngine
from ...methods.funcoder.defaults import DEFAULT_IMPORTS
from ...methods.shared import CodeGenContext, CodeGenJournal
from ..types import CodeGenEvalTasks, EvalResult


class InputMaths(TypedDict):
    """Compacted `MATH_test.json` format."""

    task: str  # need to fill in, not in original data
    ident: str  # same ^
    problem: str
    level: str
    type: str
    solution: str


class VerdictMaths(TypedDict):
    # verdict: ok -> 1.0, fail -> 0.0, no code -> None
    ok: bool
    ret_code: int
    stderr: str  # or empty string


class _JudgeEqPrompt:
    def wrap_prompt(self, gt: str, hyp: str) -> list[ChatMessage]:
        raise NotImplementedError()

    def verdict(self, result: str) -> bool | None:
        raise NotImplementedError()

    pass


class MathsEvalTasks(CodeGenEvalTasks[InputMaths, VerdictMaths]):
    """Measuring Mathematical Problem Solving With the MATH Dataset
    https://github.com/hendrycks/math"""

    name = "MATH"

    def __init__(self, json_path: pathlib.Path, samples: int | None, llm_engine: LLMEngine):
        self._json_path = json_path
        self._take_samples = samples
        self._llm_engine = llm_engine

    def iter(self) -> Generator[tuple[str, InputMaths], None, None]:
        # load data
        with open(pathlib.Path(self._json_path), "r", encoding="utf-8") as f:
            data: dict[str, InputMaths] = json.load(f)
        for k, v in data.items():
            v["task"], v["ident"] = k.split("/")
        items = list(data.values())

        # reproducible randomization
        if self._take_samples is not None:
            rand = random.Random()
            rand.seed(42)
            rand.shuffle(items)
            items = items[: self._take_samples]

        for item in items:
            task_id = f"{item['task']}_{item['ident']}"
            yield task_id, item
        return

    async def execute(self, ctx, method, task_id, task) -> tuple[EvalResult[InputMaths, VerdictMaths], CodeGenJournal]:
        # gap: default imports
        assert ctx.lrt.lang == "python"  # TODO: support other languages

        r_docstring = "\n".join("    " + line for line in task["problem"].split("\n"))
        r_docstring = r_docstring.strip()
        prompt_func = ctx.lrt._parse.make_stub_function_def_from_params(
            name="solution",
            docstring=r_docstring,
            args=[],
            ret=None,
        )

        program, _sj = await method.gen(ctx, [], prompt_func, [])
        if program is None:
            raise ValueError("failed to generate program")
        code = ctx.lrt.pretty_fmt(program)

        return {
            "id": task_id,
            "task": task,
            "code": code,
            "_code_error": None,
            "_code_tree": None,
            "verdict": None,
            "_verdict_info": None,
        }, _sj

    async def judge(self, ctx, result) -> EvalResult[InputMaths, VerdictMaths]:
        draft_code = result["code"]
        if draft_code is None:
            result["verdict"] = None
            result["_verdict_info"] = {"ok": False, "ret_code": -1, "stderr": "no code to execute"}
            return result
        draft_code = DEFAULT_IMPORTS + "\n\n\n" + draft_code

        exec_program = ctx.lrt.parse(module=(), code=draft_code)
        exec_solution = LrtSolution(modules=[exec_program])
        exec_result = await ctx.lrt.run_solution(exec_solution, (), "solution", args=[], kwargs={}, timeout=2.5)
        print(f"================ {result['task']} / {result['id']} ================")
        print(exec_result.error)

        run_ok = exec_result.ok and exec_result.ret_code == 0
        if not run_ok:
            result["verdict"] = 0.0
            result["_verdict_info"] = {
                "ok": False,
                "ret_code": exec_result.ret_code,
                "stderr": exec_result.error,
            }
            return result
        judge_gt = self._extract_original_answer(result["task"]["solution"])
        judge_hyp = str(exec_result.result).strip()
        judge_ok = await self._judge_answer_eq(ctx, 3, _MathsJudgeEqPrompt(), judge_gt, judge_hyp)
        if judge_ok is None:
            result["verdict"] = 0.0
            result["_verdict_info"] = {"ok": False, "ret_code": -1, "stderr": "judger failed"}
            return result

        result["verdict"] = 1.0 if judge_ok else 0.0
        result["_verdict_info"] = {
            "ok": judge_ok,
            "ret_code": exec_result.ret_code,
            "stderr": exec_result.error,
        }
        return result

    def _extract_original_answer(self, solution: str) -> str:
        """Extract \\boxed{} from string"""
        after_string = solution.split("\\boxed{")[1]
        balance = 1
        content = ""
        for ch in after_string:
            if ch == "{":
                balance += 1
            elif ch == "}":
                balance -= 1
            if balance == 0:
                break
            content += ch
        return content

    async def _judge_answer_eq(
        self, ctx: CodeGenContext, opt_retries: int, prompt: _JudgeEqPrompt, gt: str, hyp: str
    ) -> bool | None:
        """Creates a LLM-based judger for inputs."""

        # shortcut for exact match
        if gt == hyp:
            return True
        try:
            return int(gt) == int(hyp)
        except ValueError:
            pass
        try:
            return abs(float(gt) - float(hyp)) < 1e-4
        except ValueError:
            pass

        ctx.log.in_scope(f"judge_eq[]")
        result: bool | None = None

        for _retry in range(1, opt_retries + 1):
            history = prompt.wrap_prompt(gt, hyp)
            temperature = min(0.7, (_retry - 1) / 5)  # 0.0, 0.2, 0.4, 0.6, 0.7
            _ret = await self._llm_engine.call(history, n=1, temperature=temperature)
            if not _ret.ok:
                ctx.log.string(f"attempt {_retry} failed: llm call failed")
                continue
            responses = _ret.ok
            if len(responses) < 1:
                ctx.log.string(f"attempt {_retry} failed: llm did not return anything")
                continue
            response = responses[0]
            result = prompt.verdict(response)
            if result is not None:
                break
        return result

    pass


class _MathsJudgeEqPrompt(_JudgeEqPrompt):
    def wrap_prompt(self, gt, hyp) -> list[ChatMessage]:
        next: ChatMessage = {"role": "user", "content": self._format_cmp(gt, hyp)}
        return self._few_shot_examples + [next]

    @staticmethod
    def _format_cmp(gt: str, hyp: str) -> str:
        return f"""Answer: {gt}
Prediction: {hyp}
"""

    _few_shot_examples: list[ChatMessage] = [
        {
            "role": "system",
            "content": """You are a mathematical teacher, your task is to:
- judge whether the prediction is matching the answer
- output "Judge: Correct." or "Judge: Wrong.", please do not output redundant words
- numerical errors should be ignored ($1$ is equal to $0.99999998$)
- some answer might be represent in latex format, and some might be float number, this should be consider as correct ($\\frac{{1}}{{2}}$ is equal to $0.5$, $3\\sqrt{{66}}$ is equal to $24.37211$)
- some answer ignores the unit, and should be consider as correct ($13 cm^2$ is equal to $13.0$, $\\$13$ is equal to $13$)
Please output "Judge: Correct." if two answer is literally the same, or "Judge: Wrong." for not same, please do not output redundant words.""",
        },
        {"role": "user", "content": _format_cmp("-1", "-1.0")},
        {"role": "assistant", "content": "Judge: Correct."},
        {"role": "user", "content": _format_cmp("\\$36", "36.0")},
        {"role": "assistant", "content": "Judge: Correct."},
        {"role": "user", "content": _format_cmp("106^\\circ", "90")},
        {"role": "assistant", "content": "Judge: Wrong."},
        {"role": "user", "content": _format_cmp("\\frac{8}{15}", "0.5333333333")},
        {"role": "assistant", "content": "Judge: Correct."},
        {"role": "user", "content": _format_cmp("3", "2.9999999999999996")},
        {"role": "assistant", "content": "Judge: Correct."},
        {"role": "user", "content": _format_cmp("\\frac{14}{3}", "7.0")},
        {"role": "assistant", "content": "Judge: Wrong."},
    ]

    def verdict(self, result) -> bool | None:
        if "judge: correct" in result.lower():
            return True
        elif "judge: wrong" in result.lower():
            return False
        return None

    pass
