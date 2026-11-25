# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parser/registry checks against an actual local Raon tokenizer."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoConfig, AutoTokenizer

    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
    from vllm.transformers_utils.configs.raon1_mini import Raon1MiniConfig

    AutoConfig.register("raon1_mini", Raon1MiniConfig, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    request = ChatCompletionRequest(
        model="raon",
        messages=[{"role": "user", "content": "test"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )
    reasoning_class = ReasoningParserManager.get_reasoning_parser("raon_1")
    reasoning = reasoning_class(tokenizer)
    thought, content = reasoning.extract_reasoning("Check.</think>5입니다.", request)
    assert thought == "Check." and content.strip() == "5입니다."
    off = reasoning_class(tokenizer, chat_template_kwargs={"enable_thinking": False})
    assert off.extract_reasoning("5입니다.", request) == (None, "5입니다.")
    streaming = reasoning_class(tokenizer)
    previous, thoughts, contents = "", [], []
    closing_tag = "</think>"
    for chunk in ("Che", "ck.", closing_tag[:5], closing_tag[5:], "5입니다."):
        current = previous + chunk
        delta = streaming.extract_reasoning_streaming(
            previous, current, chunk, [], [], []
        )
        if delta is not None:
            thoughts.append(delta.reasoning or "")
            contents.append(delta.content or "")
        previous = current
    assert "".join(thoughts) == "Check." and "".join(contents) == "5입니다."
    tool = ToolParserManager.get_tool_parser("raon_1")(tokenizer, request.tools)
    plain = tool.extract_tool_calls("5입니다.", request)
    assert not plain.tools_called and plain.content == "5입니다."
    markup = (
        "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n"
        "<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>"
    )
    call = tool.extract_tool_calls(markup, request)
    assert call.tools_called and call.tool_calls[0].function.name == "add"
    assert json.loads(call.tool_calls[0].function.arguments) == {"a": 2, "b": 3}
    tag = get_model_structural_tag(
        "raon_1", request.tools, "required", reasoning=False, token_suffix=""
    )
    assert tag is not None
    Path(args.output).write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": [
                    "registry resolution",
                    "reasoning split",
                    "thinking off",
                    "split-delimiter reasoning stream",
                    "plain content",
                    "typed tool arguments",
                    "five-argument structural-tag dispatch",
                ],
                "scope": "CPU parser contracts; no model generation or GPU benchmark",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
