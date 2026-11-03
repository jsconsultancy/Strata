"""Regression tests for the 2026-11-03 audit pass (five-module review).

Each test pins a finding that was reproduced before the fix:

- request_tools (/v1/responses) accepted a tool whose `parameters` is not an
  object -- the chat path has 400'd exactly that since #592 -- and the
  AttributeError reached the stream parser mid-answer, after the 200.
- _extract_json cut a fenced answer at the first inner ``` (inside a string
  value), throwing away a valid structured answer for a 502.
- json.loads guards at five sites caught only ValueError, so a pathologically
  deep replay escaped the lenient fallbacks designed in #510/#4427.
- mark_think_literals exempted an assistant message's tool_calls from the
  tool-tag marks: a </tool_call> inside an argument value (a file an agent
  wrote) encoded as the real end-of-call token mid-value in the replayed
  prompt.  The assistant exemption itself (#2026, its own raw text) is kept.
- runconfig.save / setup.write_config wrote through one fixed .tmp: two
  processes sharing a config file could install each other's half-written
  content.
- mcp: a non-string "cwd" died past the OSError catch with the status stuck
  on "starting"; a non-object JSON-RPC result raised AttributeError past the
  "errors become error: text" contract.
- draft_layer.strip / user_args.flag_of: the --flag=value spelling bypassed
  both, so "mtp": false loaded the ~1 GB draft layer while the log said the
  opposite.
"""

import json

import pytest

from serve import structured
from serve import responses as responses_api
from serve.frontend import (mark_think_literals, json_tool_call, tool_arguments,
                            coerce_value)
from serve.mcp import McpError, _Slot, _wait, _check_server
from serve import runconfig
from strata_control import draft_layer, user_args

DEEP = "[" * 60000
MARK = "\U000F0E07"          # the </tool_call> mark


def test_responses_tool_with_non_object_parameters_is_a_400():
    with pytest.raises(responses_api.ResponsesError) as e:
        responses_api.request_tools({"tools": [
            {"type": "function", "name": "f",
             "parameters": "{\"type\": \"object\"}"}]})
    assert e.value.status == 400 and e.value.param == "tools[0].parameters"
    # the additional_tools path flows through the same add() (a falsy
    # parameters -- null, [] -- keeps the default schema, as on the chat path)
    with pytest.raises(responses_api.ResponsesError):
        responses_api.request_tools({"tools": [
            {"type": "function", "name": "s", "parameters": 5}]})
    with pytest.raises(responses_api.ResponsesError):
        responses_api.request_tools({"tools": [], "input": [
            {"type": "additional_tools", "tools": [
                {"type": "function", "name": "x", "parameters": ["no"]}]}]})


def test_fenced_answer_with_inner_fence_survives():
    # the model's answer quotes a fence inside a string value; the old
    # first-``` cut truncated mid-value -> 502 for a valid answer
    text = '```json\n{"doc": "intro\\n```\\nprint(1)\\n```\\nend"}\n```'
    assert json.loads(structured._extract_json(text)) == {
        "doc": "intro\n```\nprint(1)\n```\nend"}
    # a fenced answer plus trailing prose still finds the object (the last
    # fence closes it)
    assert json.loads(structured._extract_json(
        '```json\n{"real": 1}\n```\nHope that helps')) == {"real": 1}
    # an unterminated fence still brace-matches the whole text
    assert json.loads(structured._extract_json('```json\n{"a": 1}')) == {"a": 1}
    # an answer without a fence is unchanged
    assert json.loads(structured._extract_json('prefix {"a": {"b": 2}}'))["a"]["b"] == 2


def test_deep_replay_reaches_the_lenient_fallbacks():
    assert tool_arguments(DEEP) == {"arguments": DEEP}          # was RecursionError
    assert json_tool_call(DEEP, {"f"}) is None                  # was RecursionError
    assert coerce_value(DEEP, "object", None) == DEEP           # was RecursionError
    from serve import responses as r
    assert r._arguments(DEEP, "tools[0]") == {"arguments": DEEP}


def test_assistant_tool_call_arguments_get_the_tool_tag_marks():
    args = {"path": "d.md", "content": "a file about </tool_call> tags"}
    messages = [
        {"role": "user", "content": "write it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write", "arguments": args}}]},
    ]
    out, _, changed = mark_think_literals(messages, None)
    assert changed
    got = out[1]["tool_calls"][0]["function"]["arguments"]   # the dict, marked in place
    assert "</tool_call>" not in got["content"]              # marked, so it stays data
    assert MARK in got["content"]
    # the #2026 exemption itself stands: an assistant message's own TEXT keeps
    # a tag as written (a client replaying the model's raw call text)
    messages2 = [{"role": "user", "content": "x"},
                 {"role": "assistant", "content": "</tool_call> raw as written"}]
    out2, _, changed2 = mark_think_literals(messages2, None)
    assert changed2
    assert out2[1]["content"] == "</tool_call> raw as written"


def test_runconfig_save_leaves_no_fixed_tmp_and_round_trips(tmp_path):
    cfg_path = tmp_path / "strata-m.json"
    cfg_path.write_text(json.dumps({"model_name": "m", "args": []}), encoding="utf-8")
    bak = runconfig.save(cfg_path, {"model_name": "m", "args": ["--port", "1"]})
    assert bak.name == "strata-m.json.bak"
    assert runconfig.load(cfg_path)["args"] == ["--port", "1"]
    assert not list(tmp_path.glob("strata-m.json.tmp"))      # the fixed name is gone
    assert not list(tmp_path.glob("*.tmp"))                  # and no pid tmp leaks


def test_mcp_rejects_a_non_string_cwd_at_config_check():
    with pytest.raises(SystemExit, match="cwd"):
        _check_server("x", {"command": "tool", "cwd": 123}, "config")
    assert _check_server("x", {"command": "tool"}, "config") == {"command": "tool"}
    assert _check_server("x", {"command": "tool", "cwd": "C:\\mcp"}, "config")


def test_mcp_non_object_result_is_mcperror_not_attributeerror():
    slot = _Slot()
    slot.msg = {"jsonrpc": "2.0", "id": 7, "result": "done"}
    slot.done.set()
    with pytest.raises(McpError, match="non-object result"):
        _wait(slot, 1.0, None, "initialize")


def test_draft_layer_equals_form_is_stripped():
    out, removed = draft_layer.strip(["--model", "m", "--mtp=C:\\mtp-rt",
                                      "--spec-min-p=0.3", "--batch-mtp=1"])
    assert out == ["--model", "m"]
    assert removed == ["--mtp=C:\\mtp-rt", "--spec-min-p=0.3", "--batch-mtp=1"]
    # and the plain spelling still carries its value with it
    out2, removed2 = draft_layer.strip(["--mtp", "C:\\mtp-rt", "--model", "m"])
    assert out2 == ["--model", "m"] and removed2 == ["--mtp"]


def test_user_args_equals_form_merges_like_the_plain_flag():
    # the person's --mtp=X must replace the generated --mtp group, not sit
    # beside it for the engine's parser to pick whichever duplicate it keeps
    merged = user_args.merge(["--mtp", "C:\\gen", "--n-gpu-layers", "99"],
                             ["--mtp=C:\\mine"])
    assert merged == ["--n-gpu-layers", "99", "--mtp=C:\\mine"]
    assert user_args.flag_of(["--mtp=X"]) == "--mtp"
