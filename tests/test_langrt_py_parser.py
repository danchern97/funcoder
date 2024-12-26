import ast
from typing import cast

import pytest

from funcoder.langrt.py_parse.parser import PyParser
from funcoder.langrt.py_parse.utils import ppy_get_python_packages
from funcoder.langrt.types import (
    CodeBlock,
    LrtConstantDef,
    LrtEnumDef,
    LrtFunctionDef,
    LrtImport,
    LrtStructDef,
    SymbolName,
)
from funcoder.utils.strings import code_block


@pytest.fixture
def pyp():
    return PyParser()


def test_langrt_py_parser_get_python_packages(pyp: PyParser):
    pkgs_1p = ppy_get_python_packages(incl_std=True, incl_3p=False)
    assert "os" in pkgs_1p
    assert "pytest" not in pkgs_1p
    pkgs_3p = ppy_get_python_packages(incl_std=False, incl_3p=True)
    assert "os" not in pkgs_3p
    assert "pytest" in pkgs_3p
    pkgs_all = ppy_get_python_packages(incl_std=True, incl_3p=True)
    assert "os" in pkgs_all
    assert "pytest" in pkgs_all


def test_langrt_py_parser_parse_import(pyp: PyParser):
    def _test(
        code: CodeBlock,
        module: tuple[SymbolName, ...],
        symbols: list[tuple[SymbolName, SymbolName]],
        level: int,
        alt_code: str | None = None,
    ):
        node = ast.parse(code_block(code))
        expected = LrtImport(kind="import", code=code, module=module, symbols=symbols, level=level)
        parsed = pyp.parse_import(code, cast(ast.Import, node.body[0]))
        assert parsed == expected

    _test("import pydantic\n", module=("pydantic",), symbols=[], level=0)
    # _test("import numpy, scipy\n", module=(), symbols=[("numpy", "numpy"), ("scipy", "scipy")], level=0)
    _test("import pathlib.path\n", module=("pathlib", "path"), symbols=[], level=0)
    # _test("import numpy as np, pandas as pd", module=(), symbols=[("numpy", "np"), ("pandas", "pd")], level=0)
    _test(
        "import pathlib.any.path as p\n",
        module=("pathlib", "any"),
        symbols=[("path", "p")],
        level=0,
        alt_code="from pathlib.any import path as p\n",
    )
    _test("from pathlib import Path\n", module=("pathlib",), symbols=[("Path", "Path")], level=0)
    _test("from pathlib import Path as P\n", module=("pathlib",), symbols=[("Path", "P")], level=0)
    _test("from . import foo\n", module=(), symbols=[("foo", "foo")], level=1)
    _test(
        "from ...utils.strings import bar, foobar as fb\n",
        module=("utils", "strings"),
        symbols=[("bar", "bar"), ("foobar", "fb")],
        level=3,
    )


def test_langrt_py_parser_parse_enum_def(pyp: PyParser):
    code = """
        class Foo(enum.Enum):
            a="alpha"
            b= 'beta'
            c =3
            x: 233 = 4
            pass
    """
    node = ast.parse(code_block(code))
    parsed = pyp.parse_enum_def(code, cast(ast.ClassDef, node.body[0]))

    formatted_code = """
        class Foo(enum.Enum):
            a = "alpha"
            b = "beta"
            c = 3
            x: 233 = 4
            pass
    """
    expected = LrtEnumDef(
        kind="enum",
        code=code_block(formatted_code),
        name="Foo",
        docstring=None,
        options={
            "a": "alpha",
            "b": "beta",
            "c": 3,
            "x": 4,
        },
    )
    assert parsed == expected


def test_langrt_py_parser_parse_struct_def(pyp: PyParser):
    code = '''
        class Foo(pydantic.BaseModel):
            """Recursive definition of a link list."""

            bar: int
            child:list["Foo"]= None
            pass
    '''
    node = ast.parse(code_block(code))
    parsed = pyp.parse_struct_def(code, cast(ast.ClassDef, node.body[0]))

    formatted_code = '''
        class Foo(pydantic.BaseModel):
            """Recursive definition of a link list."""

            bar: int
            child: list["Foo"] = None
            pass
    '''
    expected = LrtStructDef(
        kind="struct",
        code=code_block(formatted_code),
        name="Foo",
        docstring="Recursive definition of a link list.",
        fields={
            "bar": ("int", None),
            "child": ("list['Foo']", "None"),
        },
    )
    assert parsed == expected


def test_langrt_py_parser_parse_function_def(pyp: PyParser):
    code = """
        def foo(a:int,b,c:list[int],d:bool=False) -> dict[str,list[str]]:

            # comment is not consumed
            if a  and  b:
                # second comment
                return {  "x": []  }   # third
            return {}
    """
    node = ast.parse(code_block(code))
    parsed = pyp.parse_function_def(code, cast(ast.FunctionDef, node.body[0]))

    formatted_code = """
        def foo(a: int, b, c: list[int], d: bool = False) -> dict[str, list[str]]:

            # comment is not consumed
            if a and b:
                # second comment
                return {"x": []}  # third
            return {}
    """
    expected_body = """    # comment is not consumed
    if a and b:
        # second comment
        return {"x": []}  # third
    return {}"""
    expected = LrtFunctionDef(
        kind="function",
        code=code_block(formatted_code),
        name="foo",
        docstring=None,
        args=[
            ("a", "int", None),
            ("b", None, None),
            ("c", "list[int]", None),
            ("d", "bool", "False"),
        ],
        ret="dict[str, list[str]]",
        implemented=True,
        body=expected_body,
    )
    assert parsed == expected


def test_langrt_py_parser_parse_constant(pyp: PyParser):
    def _assert(code: CodeBlock, name: str, type: str | None, value: str) -> None:
        node = ast.parse(code_block(code))
        parsed = pyp.parse_constant_def(code, cast(ast.Assign | ast.AnnAssign, node.body[0]))
        expected = LrtConstantDef(
            kind="constant",
            code=code.strip() + "\n",
            name=name,
            type=type,
            value=value,
        )
        assert parsed == expected

    _assert("MOD = 998244353 * 97", "MOD", None, "998244353 * 97")
    _assert("variable: int | None = 23333", "variable", "int | None", "23333")


def test_langrt_py_parser_parse_code(pyp: PyParser):
    c_import_1 = "import enum"
    c_import_2 = "from pydantic import BaseModel"
    c_import_3 = "from ..util.types import Foo as TypeFoo"

    c_const = """
        DISABLE_CACHE: bool = False
    """
    c_const = code_block(c_const)

    c_enum = """
        class Bar(enum.Enum):
            ich = 1
            ni = 2
            san = "san"
    """
    c_enum = code_block(c_enum)

    c_struct = '''
        class Struct(BaseModel):
            """This is a struct."""

            args: list[Bar] = []
            pass
    '''
    c_struct = code_block(c_struct)

    c_func = """
        def fn(arg: Struct, kw: dict[str, TypeFoo] = {}):
            r'''
            Very complex docstring with
            multiple lines, \"\"\"

            and other ''
            quotes.'''

            if arg:
                return arg
            # rest is not done yet

            raise NotImplementedError()
    """
    c_func = code_block(c_func)

    c_main = """
        if __name__ == "__main__":
            main()
            print("Hello, World!")
    """
    c_main = code_block(c_main)

    code = "\n".join(
        [c_import_1, "", c_import_2, c_import_3, "", c_const, "", c_enum, "", c_struct, "", c_main, "", c_func]
    )

    e_imp_1 = LrtImport(kind="import", code=c_import_1 + "\n", module=("enum",), symbols=[], level=0)
    e_imp_2 = LrtImport(
        kind="import",
        code=c_import_2 + "\n",
        module=("pydantic",),
        symbols=[("BaseModel", "BaseModel")],
        level=0,
    )
    e_imp_3 = LrtImport(
        kind="import",
        code=c_import_3 + "\n",
        module=("util", "types"),
        symbols=[("Foo", "TypeFoo")],
        level=2,
    )
    e_const = LrtConstantDef(
        kind="constant",
        code=c_const.strip() + "\n",
        name="DISABLE_CACHE",
        type="bool",
        value="False",
    )
    e_enum = LrtEnumDef(
        kind="enum",
        code=c_enum,
        name="Bar",
        docstring=None,
        options={"ich": 1, "ni": 2, "san": "san"},
    )
    e_struct = LrtStructDef(
        kind="struct",
        code=c_struct,
        name="Struct",
        docstring="This is a struct.",
        fields={"args": ("list[Bar]", "[]")},
    )
    e_docstring = """
    Very complex docstring with
    multiple lines, \"\"\"

    and other ''
    quotes."""
    e_func_body = """    if arg:
        return arg
    # rest is not done yet

    raise NotImplementedError()"""
    e_func = LrtFunctionDef(
        kind="function",
        code=c_func,
        name="fn",
        docstring=e_docstring,
        args=[("arg", "Struct", None), ("kw", "dict[str, TypeFoo]", "{}")],
        ret=None,
        implemented=False,  # depends on last expression
        body=e_func_body,
    )
    expected = [e_imp_1, e_imp_2, e_imp_3, e_const, e_enum, e_struct, e_func]

    parsed = pyp.parse_code(code)
    assert parsed == expected


def test_langrt_py_parser_sort_import(pyp: PyParser):
    prog = """
        import typing
        from typing import List, Tuple
        from typing import Any as any
        import math
    """
    prog = code_block(prog)
    prog_nodes = cast(list[LrtImport], pyp.parse_code(prog))
    expected = """
        import math
        import typing
        from typing import Any as any, List, Tuple
    """
    expected = code_block(expected)
    p_prog = pyp.fmt_imports(prog_nodes)
    p_prog = p_prog.strip()
    p_expected = expected.strip()
    assert p_prog == p_expected


def test_langrt_py_parser_deduplicate_nodes(pyp: PyParser):
    prog = """
        import typing
        import abc

        import math
        from math import factorial


        def foo() -> int:
            return math.pow(1, 3) * factorial(233)

        def foo() -> None:
            raise NotImplementedError()
    """
    prog = code_block(prog)
    expected = """
        import math
        from math import factorial


        def foo() -> int:
            return math.pow(1, 3) * factorial(233)
    """
    expected = code_block(expected)
    p_prog = pyp.parse_code(prog)
    p_prog = pyp.deduplicate_nodes(p_prog)
    p_prog = pyp.fmt_nodes(p_prog, organize_imports=True)
    p_prog = p_prog.strip()
    p_expected = expected.strip()
    assert p_prog == p_expected
