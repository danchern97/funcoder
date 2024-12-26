<div align="center">
<h2>
Divide-and-Conquer Meets Consensus:

Unleashing the Power of Functions in Code Generation
</h2>
</div>

<div align="center">
    <a href="https://doi.org/10.48550/arXiv.2405.20092"><img src="https://img.shields.io/badge/NeurIPS%202024-Oral-906ba2.svg" alt="Paper"></a>
    <a href="https://github.com/cometeme/funcoder/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/cometeme/funcoder?color=green"> </a>
</div>

This repository contains the official implementation for **NeurIPS 2024** paper **_Divide-and-Conquer Meets Consensus: Unleashing the Power of Functions in Code Generation_**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/method_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./assets/method_light.png">
  <img alt="FunCoder method diagram." src="./assets/method_light.png">
</picture>

In this work, we propose FunCoder, a code generation framework utilizing a divide-and-conquer strategy and a novel functional consensus mechanism on functions to decompose complex problems.

- **Divide**: Starting from the main problem, FunCoder introduces new functions to cope with certain sub-problems. The new functions will be decomposed recursively, eventually forming a tree of functions.
- **Conquer**: FunCoder then combines functions bottom-up to achieve increasingly complicated objectives.
- **Functional Consensus**: By dividing-and-conquering tasks into simpler sub-functions, complexity can be gradually reduced.
However, errors in sub-functions may propagate to the whole program, thereby damaging overall reliability.
We propose functional consensus that samples multiple functions and selects the one demonstrating consensus, measured by the aggregated similarity among candidates. 
By reaching a consensus, we reduce the discrepancies in code behavior and thus alleviate cascading errors.

## 🔧 Setup

To run experiments concerning FunCoder (and/or other LLM-based code generation methods), you need to set up the config file at `funcoder/eval/config.toml` by duplicating `funcoder/eval/config.template.toml` and change the relevant settings first. We keep this file from being tracked by Git, so you won't accidentally commit your API keys or other sensitive information.

### Environment

```bash
conda create -y -n funcoder python=3.10
conda activate funcoder
python -m pip install -r requirements.txt
```

### Datasets

Run the following command under repo root and and we'll download and preprocess the datasets (HumanEval, MATH, MBPP, xCodeEval) automatically.

```bash
python -m funcoder.eval download-datasets
```

### Configuring OpenAI models

Configure your LLM settings by adding a new section for the model you want to use, in `funcoder/eval/config.toml`. We've included a big list of models in the `funcoder/eval/config.template.toml` file which is probably going to come in handy, but you can also add your own models by reading the docs in `funcoder/llm/config.py`. Here's an example for OpenAI GPT-4o mini:

```toml
[llm.gpt_4o_mini_240718]
kind = "gpt"
endpoint = "https://api.openai.com/v1"
key = "********************************"
api_type = "open_ai"
api_dialect = "chat_completions"
model = "gpt-4o-mini-2024-07-18"
```

### Configuring Local Models

You can also utilize local models by deploying an OpenAI-compatible API server powered by [vLLM](https://github.com/vllm-project/vllm).

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model/stable-code-instruct-3b \
    --dtype auto \
    --api-key token-stablecode_3b_instruct \
    --port 28101
```

And add the following config to `dqllm/config/config.toml` to have it recognizable by the `eval` module. In this way, local models can be used in the same way like OpenAI models, provided that the vLLM server is already running. It must be noted that different open source models have different configurations (esp. mixins), so careful configuration is needed.

```toml
[llm.stablecode_3b_instruct]
kind = "gpt"
endpoint = "http://localhost:28105/v1"
key = "token-stablecode_3b_instruct"
model = "/home/share/models/stable-code-instruct-3b"
api_type = "open_ai"
api_dialect = "chat_completions"
mixin_mock_few_shot_prompt = true
```

## 🧪 Experiments

For each experiment involving a dataset and a certain method (with options included), you need to set up an experiment config inside a `.hparams.json` file under `/your/experiment/dir/`. We've already added a few in the `/experiments` directory so you can get started quickly (they don't necessarily have to be the same as that in the paper). Here's a quick example for FunCoder:

```json
{
  "$schema": "../../funcoder/eval/hparams.schema.json",
  "task": {
    "task_name": "HumanEval",
    "task_samples": null
  },
  "langrt": "py3",
  "llm_engine": "gpt_4o_mini_240718",
  "method": {
    "method_name": "funcoder",
    "dfs_max_depth": 6,
    "divide_gen_prompt": "humaneval_divide",
    "divide_temperature": 0.2,
    "divide_retries": 3,
    "fc_root_test_prompt": "humaneval_funccall",
    "fc_root_sys_test_prompt": "sys_test_args",
    "fc_branch_test_prompt": "humaneval_funccall",
    "fc_branch_sys_test_prompt": null,
    "fc_temperature": 0.2,
    "fc_retries": 3,
    "conquer_gen_prompt": "humaneval_conquer",
    "conquer_temperature": 0.8,
    "conquer_samples": 10,
    "conquer_min_samples": 5,
    "conquer_retries": 3
  },
  "results_dir": null,
  "wandb_run_id": null
}
```

To start generating and evaluating, you need to run the following commands one by one. These commands are prone to crashes and machine restarts, such that they resume from where they left off -- disabling `--skip-done` will force the evaluation utility to re-run all tasks, which is not recommended.

```bash
python -m funcoder.eval draft --results-dir /your/experiment/dir/ --parallelism 12 --skip-done
python -m funcoder.eval judge --results-dir /your/experiment/dir/ --skip-done
```

After finishing, the human-readable results can be found in `.results.txt` under that directory. If automation of multiple experiments is sought for, use `.results.jsonl` which contains raw results for each task and consequently more data.

## 📦 Using as a Module

FunCoder comes as a module or independent Python package that can be imported from any Python script.

```bash
python -m pip install --editable .
```

Provided that the problem's input and output are given in the form of a function stub, relevant types are included, and that additional utility functions are declared to be leveraged (or not) by the final generated code, FunCoder can called to generate the complete code for that function with a simple `await` call.

```python
import asyncio
import pathlib
import sys

import pydantic

from funcoder import FuncoderCherryGen, LangRT, LLMConfig, create_code_gen_context


async def main() -> None:
    # our problem
    my_code = '''
from typing import List

StringCollection = List[str]

def longest_palindrome_substring(s: StringCollection) -> str:
    """Given a list of strings `s`, return the longest palindromic substring in
    any of `s`. If there are multiple such substrings, return the first one.

    Sample input: ['hello', 'world'] -> 'll'"""

    raise NotImplementedError()

def is_palindrome(s: str) -> bool:
    return s == s[::-1]
'''

    # set up the execution environment
    ctx = create_code_gen_context(
        llm_config=LLMConfig(
            kind="gpt",
            endpoint="https://api.openai.com/v1",
            key=pydantic.SecretStr("......"),
            api_type="open_ai",
            api_dialect="chat_completions",
            model="gpt-4o-mini-2024-07-18",
            mixin_mock_system_role=False,
        ),
        lrt=LangRT.python(
            sandbox_root=pathlib.Path("./.sandbox/"),
            parallelism=8,
        ),
        silent=False,
    )

    # and use the simplest way to start generating code
    funcoder = FuncoderCherryGen(task="humaneval")
    completion, _ = await funcoder.gen_simple(ctx, my_code, "longest_palindrome_substring")
    if completion is not None:
        # shows the prettified, formatted output
        print(ctx.lrt.pretty_fmt(completion))
    return


if __name__ == "__main__":
    asyncio.run(main())
```

There are lots of changeable attributes and properties, e.g. the maximum depth of the divide-and-conquer tree, the number of samples for each branch, and the number of retries for each step. You can check out the class documentations for more details.

## 📝 Citation

If you find our work helpful, you can cite this paper as:

```bibtex
@inproceedings{
    chen2024funcoder,
    title={Divide-and-Conquer Meets Consensus: Unleashing the Power of Functions in Code Generation},
    author={Jingchang Chen and Hongxuan Tang and Zheng Chu and Qianglong Chen and Zekun Wang and Ming Liu and Bing Qin},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=cFqAANINgW}
}
```
