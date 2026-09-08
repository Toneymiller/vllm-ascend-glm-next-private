# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""按 GLM5 Next Transformers 模型结构生成完整合成 safetensors 权重。

这里的“分片”仅表示 Hugging Face checkpoint 的磁盘文件分卷。每个参数
tensor 都保持完整，绝不会按 TP rank 切分；Transformers/vLLM 在加载完整
tensor 后，各自通过 weight loader 完成 TP 切分。

生成的是用于验证模型加载和启动流程的合成权重，不具备推理精度。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

SUPPORTED_ARCHITECTURES = {
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
}
WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
)


class TensorSpec(NamedTuple):
    name: str
    shape: tuple[int, ...]
    dtype: Any
    size_bytes: int


def _is_weight_file(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in WEIGHT_PATTERNS)


def load_and_validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config.json 顶层必须是 JSON object。")

    architectures = config.get("architectures") or []
    if not set(architectures) & SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "config.json 不是 GLM5 Next：architectures=" f"{architectures!r}。"
        )

    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError("text_config 必须是 JSON object。")

    required_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    missing = tuple(field for field in required_fields if field not in text_config)
    if missing:
        raise ValueError(f"GLM5 Next text_config 缺少字段：{missing}。")

    num_hidden_layers = int(text_config["num_hidden_layers"])
    for field in ("layer_types", "mlp_layer_types"):
        values = text_config.get(field)
        if values is not None and len(values) != num_hidden_layers:
            raise ValueError(
                f"text_config.{field} 长度必须等于 num_hidden_layers="
                f"{num_hidden_layers}，实际为 {len(values)}。"
            )
    return config, text_config


def copy_non_weight_metadata(source_dir: Path, output_dir: Path) -> None:
    """复制 tokenizer 等元数据，但不复制旧 checkpoint 和原 config。"""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    for source in source_dir.iterdir():
        if source.resolve() == output_dir or source.name in {
            ".git",
            "__pycache__",
            "config.json",
        }:
            continue
        if source.is_file():
            if not _is_weight_file(source):
                shutil.copy2(source, output_dir / source.name)
            continue
        if source.is_dir():
            shutil.copytree(
                source,
                output_dir / source.name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "__pycache__",
                    *WEIGHT_PATTERNS,
                ),
            )


def parse_size(value: str) -> int:
    normalized = value.strip().upper().replace(" ", "")
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
    }
    for unit in sorted(units, key=len, reverse=True):
        if normalized.endswith(unit):
            number = normalized[: -len(unit)]
            try:
                size = int(float(number) * units[unit])
            except ValueError as error:
                raise ValueError(f"非法容量：{value!r}") from error
            if size <= 0:
                raise ValueError("容量必须大于 0。")
            return size
    raise ValueError(f"容量必须带单位，例如 2GiB 或 5GB：{value!r}")


def discover_tensor_specs(
    config_dir: Path,
    trust_remote_code: bool,
) -> list[TensorSpec]:
    """在 meta device 构造 HF 模型，只收集完整参数结构，不分配权重内存。"""
    import torch
    import transformers
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        config_dir,
        trust_remote_code=trust_remote_code,
    )
    architectures = getattr(hf_config, "architectures", None) or []
    if len(architectures) != 1:
        raise ValueError(
            "config.architectures 必须且只能指定一个模型类，实际为："
            f"{architectures!r}"
        )
    architecture = architectures[0]
    model_class = getattr(transformers, architecture, None)
    if model_class is None:
        raise ImportError(
            f"当前 Transformers 没有导出 {architecture}；请在包含该 GLM5 Next "
            "模型实现的环境中运行脚本。"
        )

    # 严格按源 config 的 architectures 构造 ConditionalGeneration/CausalLM，
    # 不替换架构，也不删除视觉模块。
    with torch.device("meta"):
        model = model_class(hf_config)

    specs: list[TensorSpec] = []
    for name, tensor in model.state_dict().items():
        item_size = torch.empty((), dtype=tensor.dtype).element_size()
        specs.append(
            TensorSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                size_bytes=tensor.numel() * item_size,
            )
        )
    del model
    if not specs:
        raise RuntimeError("Transformers 模型没有可保存的 state_dict tensor。")
    return specs


def plan_shards(
    specs: list[TensorSpec],
    max_shard_size: int,
) -> list[list[TensorSpec]]:
    """按文件容量分卷；任何一个完整 tensor 都不会跨文件拆开。"""
    shards: list[list[TensorSpec]] = []
    current: list[TensorSpec] = []
    current_size = 0
    for spec in specs:
        if current and current_size + spec.size_bytes > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(spec)
        current_size += spec.size_bytes
    if current:
        shards.append(current)
    return shards


def shard_filename(index: int, count: int) -> str:
    if count == 1:
        return "model.safetensors"
    width = max(5, len(str(count)))
    return f"model-{index:0{width}d}-of-{count:0{width}d}.safetensors"


def make_tensor(spec: TensorSpec, initialization: str, seed: int):
    import torch

    if initialization == "normal" and (
        spec.dtype.is_floating_point or spec.dtype.is_complex
    ):
        generator = torch.Generator(device="cpu")
        # 使用稳定的参数级 seed，重跑时不受 shard 布局影响。
        name_seed = sum(
            (index + 1) * byte
            for index, byte in enumerate(spec.name.encode())
        )
        generator.manual_seed((seed + name_seed) % (2**63 - 1))
        tensor = torch.empty(spec.shape, dtype=spec.dtype, device="cpu")
        tensor.normal_(mean=0.0, std=0.02, generator=generator)
        return tensor

    # 零权重模式下将 norm scale 初始化为 1，避免归一化层把结构验证
    # 变成 NaN；其余参数和 buffer 为 0。
    if spec.name.endswith("norm.weight") or spec.name.endswith("layernorm.weight"):
        return torch.ones(spec.shape, dtype=spec.dtype, device="cpu")
    return torch.zeros(spec.shape, dtype=spec.dtype, device="cpu")


def write_checkpoint(
    output_dir: Path,
    shards: list[list[TensorSpec]],
    initialization: str,
    seed: int,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    weight_map: dict[str, str] = {}
    total_size = sum(spec.size_bytes for shard in shards for spec in shard)
    for shard_index, shard_specs in enumerate(shards, start=1):
        filename = shard_filename(shard_index, len(shards))
        tensors = OrderedDict(
            (spec.name, make_tensor(spec, initialization, seed))
            for spec in shard_specs
        )
        save_file(tensors, output_dir / filename, metadata={"format": "pt"})
        weight_map.update((name, filename) for name in tensors)
        del tensors
        print(f"已写入 {filename}（{shard_index}/{len(shards)}）", flush=True)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    if len(shards) > 1:
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return index


def prepare_output(
    config_path: Path,
    output_dir: Path,
    copy_metadata: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = next(
        (path for path in output_dir.iterdir() if _is_weight_file(path)),
        None,
    )
    if existing is not None:
        raise FileExistsError(f"输出目录已有 checkpoint，拒绝覆盖：{existing}")
    if copy_metadata:
        copy_non_weight_metadata(config_path.parent, output_dir)
    # 字节级原样复制用户给出的 config.json，不重排、不增删任何字段。
    shutil.copy2(config_path, output_dir / "config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="源 config.json。")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出模型目录。")
    parser.add_argument(
        "--max-shard-size",
        default="2GiB",
        help="HF 文件分卷上限，例如 2GiB；大于该值的单个 tensor 仍保持完整。",
    )
    parser.add_argument(
        "--initialization",
        choices=("zeros", "normal"),
        default="zeros",
        help="合成权重初始化方式；都不具备模型精度。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-copy-metadata", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="构造 meta 模型并显示参数/容量规划，但不写权重。",
    )
    return parser


def generate(args: argparse.Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(f"找不到 config.json：{args.config}")
    max_shard_size = parse_size(args.max_shard_size)
    config, text_config = load_and_validate_config(args.config)
    # 输出 config 必须与用户提供的源文件语义和内容完全一致。
    # 直接读取源目录中的原始 config；meta 模型不分配真实权重内存。
    specs = discover_tensor_specs(args.config.parent, args.trust_remote_code)
    shards = plan_shards(specs, max_shard_size)
    total_size = sum(spec.size_bytes for spec in specs)
    plan = {
        "source_architectures": config.get("architectures"),
        "output_architectures": config.get("architectures"),
        "num_hidden_layers": text_config["num_hidden_layers"],
        "tensor_count": len(specs),
        "total_size_bytes": total_size,
        "estimated_size_gib": round(total_size / 1024**3, 2),
        "file_shard_count": len(shards),
        "tensor_parallel_sharding": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    prepare_output(
        args.config,
        args.output_dir,
        copy_metadata=not args.no_copy_metadata,
    )
    free_bytes = shutil.disk_usage(args.output_dir).free
    if free_bytes < total_size:
        raise OSError(
            "磁盘空间不足：需要至少 "
            f"{math.ceil(total_size / 1024**3)} GiB，当前可用 "
            f"{free_bytes / 1024**3:.1f} GiB。"
        )
    write_checkpoint(
        args.output_dir,
        shards,
        initialization=args.initialization,
        seed=args.seed,
    )
    print(f"生成完成：{args.output_dir.resolve()}")


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
