from ....langrt import LrtFunctionDef, LrtProgram
from ....llm import ChatMessage
from ..gen_once import GenOncePrompt


class FuncoderMathsDividePrompt(GenOncePrompt):
    """Segregate requirement into multiple sub-tasks."""

    def get_few_shot_ids(self):
        # special process to 2 shot, use `solution()` and `is_common()`
        return [[0, 1, 2, 3], [0, 1, 2], [0, 3], [0]]

    get_few_shots = [
        {
            "role": "system",
            "content": """You are a programming copilot, you can solve a math problem by writing Python functions. Your task is to:

  - For every turn, you need to write a Python function that returns the answer, based on current code (not code in chat history) and problem description.
  - The entry function is `solution`.
  - Do not modify function name, arg names, docstring in given functions.
  - You can import libraries to better solve the problem.
  - If a single function is too hard to solve, you can decompose it into multiple smaller functions.
  - You can leave new function unimplemented for now, but write the function at the end of the code and comment what the function does.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": f'''Current Code:
```python
def solution():
    """Compute the sum of all common prime factors of $231$ and $3135$"""
    raise NotImplementedError()
```

Let's think step by step and complete the following Python function `solution` that solves:
"Compute the sum of all common prime factors of $231$ and $3135$"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''
First, I need to get the prime factors of $a$ and $b$.
Second, I can use `for` loop to find common element in two factors list.
Finally, sum the common factor list and return the answer.
Here is the `solution` function:

```python
def solution():
    """Compute the sum of all common prime factors of $231$ and $3135$"""
    factors_a = prime_factor(231)
    factors_b = prime_factor(3135)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    raise NotImplementedError()

def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    raise NotImplementedError()
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": f'''Current Code:
```python
def solution():
    """Compute the sum of all common prime factors of $231$ and $3135$"""
    factors_a = prime_factor(231)
    factors_b = prime_factor(3135)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""

    raise NotImplementedError()
```

Let's think step by step and complete the following Python function `prime_factor` that solves:
"get a list of prime factors of number $x$"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''First, factor number is a positive integer that divides the number exactly, leaving no remainder.
Second, I need to check whether a factor is prime number
Here is the `prime_factor` function:

```python
def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    ret = []
    i = 1
    while i * i <= x:
        i += 1
        if x % i == 0 and is_prime(i):
            ret.append(i)
    return ret

def is_prime(x: int) -> bool:
    """determine $x$ is a prime number or not"""
    raise NotImplementedError()
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": f'''Current Code:
```python
def solution():
    """Compute the sum of all common prime factors of $231$ and $3135$"""
    factors_a = prime_factor(231)
    factors_b = prime_factor(3135)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

def prime_factor(x: int) -> list:
    """get a list of prime factors of number $x$"""
    ret = []
    i = 1
    while i * i <= x:
        i += 1
        if x % i == 0 and is_prime(i):
            ret.append(i)
    return ret

def is_prime(x: int) -> bool:
    """determine $x$ is a prime number or not"""

    raise NotImplementedError()
```

Let's think step by step and complete the following Python function `is_prime` that solves:
"determine $x$ is a prime number or not"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''
Here is the `is_prime` function:

```python
def is_prime(x: int) -> bool:
    """determine $x$ is a prime number or not"""
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": f'''Current Code:
```python
def solution():
    """Compute the sum of all common prime factors of $231$ and $3135$"""
    factors_a = prime_factor(231)
    factors_b = prime_factor(3135)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)

def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    raise NotImplementedError()
```

Let's think step by step and complete the following Python function `get_common` that solves:
"get common element in two list $a$ and $b$"''',
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''
Here is the `get_common` function:

```python
def get_common(a: list, b: list) -> list:
    """get common element in two list $a$ and $b$"""
    ret = []
    for item in a:
        if item in b:
            ret.append(item)
    return ret
```''',
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        func = ctx.lrt.parse(module=(), code=code).find(LrtFunctionDef, func_name)
        func_docstring = (func.docstring if func else None) or ""
        return {
            "role": "user",
            "content": f'''Below chats are examples, these code are not in your workspace, now you need to solve a new problem:
Current Code:
```{lang}
{code}
```

Let's think step by step and complete the following Python function `{func_name}` that solves:
"{func_docstring}"''',
        }

    pass


class FuncoderMathsConquerPrompt(GenOncePrompt):
    """Re-generate original problem based on already completed sub-tasks."""

    @staticmethod
    def _format_user_request(lang: str, prev_code: str, cur_func_name: str, cur_func_doc: str | None) -> str:
        prev_code = prev_code.strip("\n")
        return f'''Current Code:

```python
{prev_code}
```

Let's think step by step and implemented the following method `{cur_func_name}` using existing functions that solves:
"{cur_func_doc}"'''

    get_few_shots = [
        {
            "role": "system",
            "content": """You are a programming copilot, you can solve a math problem by writing Python functions. Your task is to:

  - For every turn, you need to write a Python function that returns the answer, based on current code (not code in chat history) and problem description.
  - Do not modify function name, arg names, docstring in given functions.
  - Consider reusing existing functions that are already implemented.
  - You can import libraries to better solve the problem.""",
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code="""
def prime_factor(x: int) -> list:
    \"\"\"get a list of prime factors of number $x$\"\"\"
    ret = []
    i = 1
    while i * i <= x:
        i += 1
        if x % i == 0 and is_prime(i):
            ret.append(i)
    return ret

def is_prime(x: int) -> bool:
    \"\"\"determine $x$ is a prime number or not\"\"\"
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True

def get_common(a: list, b: list) -> list:
    \"\"\"get common element in two list $a$ and $b$\"\"\"
    ret = []
    for item in a:
        if item in b:
            ret.append(item)
    return ret

def solution():
    \"\"\"Compute the sum of all common prime factors of $231$ and $3135$\"\"\"
    raise NotImplementedError()
""",
                cur_func_name="sum_common_factors",
                cur_func_doc="Compute the sum of all common prime factors of $231$ and $3135$",
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''First, I need to get the prime factors of $231$ and $3135$.
Second, I can use `for` loop to find common element in two factors list.
Finally, sum the common factor list and return the answer.
Here is the `sum_common_factors` function:

```python
def solution():
    """Compute the sum of all common prime factors of $231$ and $3135$"""
    factors_a = prime_factor(231)
    factors_b = prime_factor(3135)
    common_factors = get_common(factors_a, factors_b)
    return sum(common_factors)
```''',
        },
        {
            "role": "system",
            "name": "user",
            "content": _format_user_request(
                lang="python",
                prev_code="""
def reverse(s: str) -> str:
    \"\"\"Return reversed input string\"\"\"

    raise NotImplementedError()
""",
                cur_func_name="reverse",
                cur_func_doc="Return reversed input string",
            ),
        },
        {
            "role": "system",
            "name": "assistant",
            "content": '''Here is the code for `reverse` function:
```python
def reverse(s: str) -> str:
    """Return reversed input string"""
    return s[::-1]
```''',
        },
    ]

    def get_next_turn(self, ctx, func_name, lang, code) -> ChatMessage:
        func = ctx.lrt.parse(module=(), code=code).find(LrtFunctionDef, func_name)
        func_docstring = (func.docstring if func else None) or ""
        return {
            "role": "user",
            "content": self._format_user_request(lang, code, func_name, func_docstring or ""),
        }

    def wrap_prompt_iter(self, ctx, ancestors, func, descendants, keep_ancestors):
        # overriding impl here is necessary because:
        #   - function body must be hidden (becoming a stub)
        #   - the to-be-conquered func must be the last one
        func_stub = ctx.lrt._parse.make_stub_function_def_from_func(func)
        program = LrtProgram(module=(), nodes=[*(ancestors if keep_ancestors else []), *descendants, func_stub])
        code = ctx.lrt.pretty_fmt(program)

        p_system, *p_few_shots = self.get_few_shots
        p_next_turn = self.get_next_turn(ctx, func.name, ctx.lrt.lang, code)
        for nshots in self.get_few_shot_ids():
            few_shots = sum(([p_few_shots[2 * i], p_few_shots[2 * i + 1]] for i in nshots), [])
            yield [p_system, *few_shots, p_next_turn]
        return

    pass
