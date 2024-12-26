import pathlib
import tempfile

from utils import async_test_case

from funcoder.langrt.types import LrtSolution
from funcoder.langrt.wrapper import LangRT
from funcoder.utils.strings import code_block


@async_test_case
async def test_langrt_wrapper_on_python_program():
    code = """
        def foo(x: int, y: int) -> int:
            return x + y
    """
    code = code_block(code)
    with tempfile.TemporaryDirectory() as tmpdir:
        with LangRT.python(sandbox_root=pathlib.Path(tmpdir), parallelism=1) as rt:
            program = rt.parse(module=(), code=code)
            result = await rt.run_program(
                program=program,
                func_name="foo",
                args=[1, 2],
                kwargs={},
            )
            assert result.ok
            assert result.result == 1 + 2
    return


@async_test_case
async def test_langrt_wrapper_on_python_solution():
    deep_types = """
        import pydantic

        class MyInput(pydantic.BaseModel):
            value: int
            pass
    """
    deep_types = code_block(deep_types)
    program = """
        from typing import List
        import my_module.types as my_types

        def foo(xs: List[my_types.MyInput], mul, mul_2: int = 4) -> int:
            return sum(i.value for i in xs) * mul * mul_2
    """
    program = code_block(program)

    with tempfile.TemporaryDirectory() as tmpdir:
        with LangRT.python(sandbox_root=pathlib.Path(tmpdir), parallelism=1) as rt:
            p_types = rt.parse(module=("my_module", "types"), code=deep_types)
            p_main = rt.parse(module=(), code=program)
            p_solution = LrtSolution(modules=[p_types, p_main])
            result = await rt.run_solution(
                solution=p_solution,
                from_module=(),
                func_name="foo",
                args=[
                    [{"value": 1}, {"value": 2}, {"value": 3}],
                ],
                kwargs={"mul": 5},
            )
            print(result.error)
            assert result.ok
            assert result.result == (1 + 2 + 3) * 5 * 4
    return


@async_test_case
async def test_langrt_wrapper_catches_exceptions():
    program = """
        def foo(x: int) -> int:
            raise ValueError("this is a test")
    """
    program = code_block(program)

    with tempfile.TemporaryDirectory() as tmpdir:
        with LangRT.python(sandbox_root=pathlib.Path(tmpdir), parallelism=1) as rt:
            prog = rt.parse(module=(), code=program)
            result = await rt.run_program(program=prog, func_name="foo", args=[1], kwargs={})
            assert not result.ok
            assert "this is a test" in result.error
    return
