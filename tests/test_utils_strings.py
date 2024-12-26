from funcoder.utils.strings import code_block, compare_strings_cf, extract_md_code, wrap_string_as_triple_quotes


def test_utils_strings_code_block():
    x = '''
        def foo(a, b):
            """this is docstring"""
            return a + b
    '''
    y = "\n".join(
        [
            "def foo(a, b):",
            '    """this is docstring"""',
            "    return a + b",
            "",
        ]
    )
    assert code_block(x) == y
    assert code_block([x, x, x]) == [y, y, y]
    assert code_block({"a": x, "b": x}) == {"a": y, "b": y}


def test_utils_strings_extract_md_code():
    x = """
This is a sample Markdown file.

```py
def foo(a: int, b: str) -> None:
    ...
```

Line interrupting the capture.

```
Plain block quote.
```

 ```nested_py
def main() -> None:
    '''Question as follows:
    ```
    this block quote should not be captured
    ```'''
    pass
```"""
    y_0 = "def foo(a: int, b: str) -> None:\n    ..."
    y_1 = "Plain block quote."
    y_2 = "def main() -> None:\n    '''Question as follows:\n    ```\n    this block quote should not be captured\n    ```'''\n    pass"
    y = [("py", y_0), ("", y_1), ("nested_py", y_2)]
    assert extract_md_code(x) == y


def test_utils_strings_compare_strings_cf():
    gt = ["a b c", "a b d"]
    hyp = "a\nb  c"
    assert compare_strings_cf(gt, hyp) is None
    hyp = "a b\td"
    assert compare_strings_cf(gt, hyp) is None
    hyp = "a b e"
    assert compare_strings_cf(gt, hyp) == "expected: 'd', found: 'e' [3th token]"
    hyp = "a b"
    assert compare_strings_cf(gt, hyp) == "expected: 'd', found: nothing [3th token]"
    hyp = "a b   d  e"
    assert compare_strings_cf(gt, hyp) == "expected: nothing, found: 'e' [4th token]"
    hyp = "a b d e f"
    assert compare_strings_cf(gt, hyp) == "expected: nothing, found: 'e' [4th token]"

    gt = ["1.00000000"]
    hyp = "1.00000001"
    assert compare_strings_cf(gt, hyp) is None

    gt = ["Yes"]
    hyp = "yes"
    assert compare_strings_cf(gt, hyp) is None


def test_utils_strings_wrap_string_as_triple_quotes():
    proc = lambda s: s.replace("_", "'").replace("*", '"').replace("/", "\\")

    def check(x, y):
        assert wrap_string_as_triple_quotes(proc(x)) == proc(y)

    check("simple string", "***simple string***")
    check("with _single quotes_", "***with _single quotes_***")
    check("with *double quotes*", "___with *double quotes*___")
    check("with ___many single quotes___", "***with ___many single quotes___***")
    check("with ***many double quotes***", "___with ***many double quotes***___")
    check("with _both quotes*", "***with _both quotes/****")
    check("___ mixed ***", "***___ mixed /*/*/****")

    check_esc = lambda x, y: wrap_string_as_triple_quotes(x) == y
    check_esc(r"abc\f \ndef", r'r"""abc\f \ndef"""')
    check_esc('\\frac{1}{2} "\\not"', r"r'''\frac{1}{2} " + '"' + r"\not" + '"' + r"'''")
