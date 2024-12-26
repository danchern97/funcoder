import asyncio
import pathlib
import tempfile

from utils import async_test_case

from funcoder.langrt.py_exec.executor import PyExecutor
from funcoder.langrt.types import LrtExecutionEnv, LrtExecutionResult
from funcoder.utils.strings import code_block


@async_test_case
async def test_langrt_py_executor_works():
    type_defs_code = """
        import pydantic

        class InputType(pydantic.BaseModel):
            strings: list[str]
            length: int
            pass

        class OutputType(pydantic.BaseModel):
            expanded: str
            pass
    """
    type_defs_code = code_block(type_defs_code)
    main_code = '''
        import time
        from typing import Dict

        import my_pkg.types as mp_types
        from .my_pkg.types import OutputType as TheOutputType

        def batch_repeat(rule: mp_types.InputType, mapping: Dict[str, str]) -> list[TheOutputType]:
            """Map `a[i]` according to mapper and repeat `b` times into `[i].item`."""

            time.sleep(3.0)  # use to stress test
            res: list[TheOutputType] = []
            modify = int(input())
            print(modify + 2333)  # will reflect in output
            for s in rule.strings:
                s = mapping.get(s, s)
                s *= rule.length + modify
                res.append(TheOutputType(expanded=s))
            return res
    '''
    main_code = code_block(main_code)

    env = LrtExecutionEnv(
        code={
            "my_pkg.types": type_defs_code,
            "": main_code,
        },
        imports=[
            (True, "my_pkg.types", "mp_types", None),
            (True, "my_pkg.types", None, [("OutputType", "TheOutputType")]),
            (True, "", None, [("batch_repeat", "batch_repeat")]),
            (False, "typing", None, [("Dict", "Dict")]),
        ],
        func_name="batch_repeat",
        func_args=["mp_types.InputType", "Dict[str, str]"],
        func_kwargs={},
        func_ret="list[TheOutputType]",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        py_exec = PyExecutor(
            sandbox_root=pathlib.Path(tmpdir),
            parallelism=4,
        )
        # cspell: disable
        rf = lambda: py_exec.run(
            env=env,
            args=[
                {
                    "strings": ["Hori", "Miyamura"],
                    "length": 3,
                },
                {
                    "Hori": "Kyouko",
                    "Miyamura": "Izumi",
                },
            ],
            kwargs={},
            stdin="1\n",  # 'modify'
            timeout=5.0,
        )
        expected = [{"expanded": "KyoukoKyoukoKyoukoKyouko"}, {"expanded": "IzumiIzumiIzumiIzumi"}]
        # cspell: enable
        # stress test
        rs = [rf() for _ in range(16)]
        done: list[LrtExecutionResult] = await asyncio.gather(*rs)
        py_exec.close()
    for r in done:
        assert r.ok
        assert r.result == expected
        assert r.stdout == "2334\n"
        assert r.duration >= 3.0
    assert len(done) == 16


@async_test_case
async def test_langrt_py_executor_allow_null_types():
    main_code = """
        def the_main() :
            print(input() * 2)
    """
    main_code = code_block(main_code)
    env = LrtExecutionEnv(
        code={"": main_code},
        imports=[
            (True, "", None, [("the_main", "the_main")]),
        ],
        func_name="the_main",
        func_args=[],
        func_kwargs={},
        func_ret="None",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        py_exec = PyExecutor(
            sandbox_root=pathlib.Path(tmpdir),
            parallelism=1,
        )
        ret = await py_exec.run(
            env=env,
            args=[],
            kwargs={},
            stdin="123\n",
            timeout=1.0,
        )
        py_exec.close()
        assert ret.ok
        assert ret.stdout == "123123\n"
