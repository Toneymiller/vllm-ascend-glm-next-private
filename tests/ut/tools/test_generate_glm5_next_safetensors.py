# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "generate_glm5_next_safetensors.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_glm5_next_safetensors",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


def _write_config(path: Path, *, layer_types=None) -> Path:
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "layer_types": layer_types
            or ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
        },
        "vision_config": {"model_type": "glm_ocr_vision", "hidden_size": 8},
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _spec(name: str, size: int):
    return generator.TensorSpec(name, (size,), "fake", size)


def test_config_keeps_conditional_architecture_and_nested_config(tmp_path):
    config_path = _write_config(tmp_path / "config.json")
    config, _ = generator.load_and_validate_config(config_path)

    assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
    assert config["text_config"]["model_type"] == "glm5_next_text"
    assert config["vision_config"]["model_type"] == "glm_ocr_vision"


def test_rejects_layer_layout_length_mismatch(tmp_path):
    config_path = _write_config(
        tmp_path / "config.json",
        layer_types=["linear_attention"],
    )

    with pytest.raises(ValueError, match="layer_types"):
        generator.load_and_validate_config(config_path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2GiB", 2 * 1024**3), ("5GB", 5 * 1000**3), ("512 MiB", 512 * 1024**2)],
)
def test_parse_size(value, expected):
    assert generator.parse_size(value) == expected


def test_plan_shards_never_splits_a_tensor():
    specs = [
        _spec("a", 6),
        _spec("b", 7),
        _spec("large", 15),
        _spec("c", 2),
    ]

    shards = generator.plan_shards(specs, max_shard_size=10)

    assert [[spec.name for spec in shard] for shard in shards] == [
        ["a"],
        ["b"],
        ["large"],
        ["c"],
    ]
    assert [spec.size_bytes for shard in shards for spec in shard] == [6, 7, 15, 2]


def test_copy_metadata_excludes_existing_weights_and_config(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    _write_config(source_dir / "config.json")
    (source_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source_dir / "model-00001.safetensors").write_bytes(b"old-weight")

    generator.copy_non_weight_metadata(source_dir, output_dir)

    assert (output_dir / "tokenizer.json").is_file()
    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "model-00001.safetensors").exists()


def test_prepare_output_copies_config_byte_for_byte(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    config_path = source_dir / "config.json"
    original = b'{\n  "architectures": ["Glm5NextForConditionalGeneration"]\n}\n'
    config_path.write_bytes(original)

    generator.prepare_output(config_path, output_dir, copy_metadata=False)

    assert (output_dir / "config.json").read_bytes() == original


def test_standard_hugging_face_shard_names():
    assert generator.shard_filename(1, 1) == "model.safetensors"
    assert generator.shard_filename(2, 12) == "model-00002-of-00012.safetensors"
