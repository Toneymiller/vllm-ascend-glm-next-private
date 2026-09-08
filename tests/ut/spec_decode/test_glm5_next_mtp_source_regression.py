# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight cross-file contract checks for GLM-5 Next MTP glue."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_PROPOSER = ROOT / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"


def _class(name: str) -> ast.ClassDef:
    tree = ast.parse(BASE_PROPOSER.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {BASE_PROPOSER}")


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    for node in _class(class_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {class_name}.{method_name} not found")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def test_glm5_multimodal_target_uses_language_model_sharing_contract() -> None:
    load_model = _src(_method("AscendSpecDecodeBaseProposer", "load_model"))

    assert "Glm5NextForConditionalGeneration" in load_model
    assert "self.model.config.image_token_index = model.config.image_token_id" in load_model
    assert "self._maybe_share_embeddings(target_language_model)" in load_model
    assert "self._maybe_share_lm_head(target_language_model)" in load_model
    assert "self._maybe_share_lm_head(model)" not in load_model


def test_glm5_mtp_shared_head_reuses_target_lm_head_without_weight_copy() -> None:
    share_lm_head = _src(
        _method("AscendSpecDecodeBaseProposer", "_maybe_share_lm_head")
    )

    assert "layer_module.shared_head.head = target_language_model.lm_head" in share_lm_head
    assert "torch.equal" not in share_lm_head
