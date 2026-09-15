"""SkinSafe AI - Model Parameter & Architecture Inspector

Detailed breakdown of parameters, layer hierarchy, memory footprint,
and computational complexity for the EfficientNet-B0 skin lesion classifier.
"""

import sys
import os
import ctypes
from pathlib import Path
from typing import Dict, Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from ml.src.model import build_model, EfficientNetB0Classifier


def enable_ansi_support() -> None:
    """Enables VT100 Virtual Terminal Processing on Windows consoles."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass


def format_num(n: int) -> str:
    """Formats large numbers with commas."""
    return f"{n:,}"


def format_bytes(n_bytes: int) -> str:
    """Formats bytes to MB."""
    return f"{n_bytes / (1024 ** 2):.2f} MB"


def inspect_model_parameters(checkpoint_path: str = "ml/checkpoints/efficientnet_b0/model.pt") -> Dict[str, Any]:
    """Inspects all parameters, layers, and memory statistics of the model."""
    enable_ansi_support()

    # ANSI Colors
    C_BLUE = "\033[94m"
    C_GREEN = "\033[92m"
    C_YELLOW = "\033[93m"
    C_CYAN = "\033[96m"
    C_BOLD = "\033[1m"
    C_RESET = "\033[0m"

    # Build model architecture
    built = build_model("efficientnet_b0", num_classes=9, pretrained=False)
    model = cast(EfficientNetB0Classifier, built)
    features = cast(nn.Sequential, model.features)
    classifier = cast(nn.Sequential, model.classifier)

    ckpt_file = REPO_ROOT / checkpoint_path

    ckpt_loaded = False
    if not ckpt_file.exists():
        # Fallback to best_model.pt if model.pt doesn't exist
        fallback = REPO_ROOT / "ml" / "checkpoints" / "efficientnet_b0" / "best_model.pt"
        if fallback.exists():
            ckpt_file = fallback

    if ckpt_file.exists():
        try:
            ckpt = torch.load(ckpt_file, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            ckpt_loaded = True
        except Exception:
            ckpt_loaded = False

    # Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Size in Memory (Float32 is 4 bytes per param)
    param_bytes = total_params * 4
    buffer_bytes = sum(b.numel() * 4 for b in model.buffers())
    total_mem_bytes = param_bytes + buffer_bytes

    print("\n" + "=" * 90)
    print(f"{C_BOLD}{C_CYAN}   SkinSafe AI - Deep Neural Network Parameter & Architecture Breakdown{C_RESET}")
    print(f"   Architecture: {C_BOLD}EfficientNet-B0{C_RESET} | Input: {C_BOLD}[3, 224, 224]{C_RESET} | Target Classes: {C_BOLD}9 Diagnoses{C_RESET}")
    print(f"   Checkpoint:   {C_GREEN if ckpt_loaded else C_YELLOW}{ckpt_file.name} ({'Loaded' if ckpt_loaded else 'Architecture Initialized'}){C_RESET}")
    print("=" * 90)

    # Layer Group Breakdown
    print(f"\n{C_BOLD}{'Layer Block':<28} {'Description':<32} {'Parameters':>12} {'% of Total':>12}{C_RESET}")
    print("-" * 90)

    # 1. Stem Conv
    stem_params = sum(p.numel() for p in features[0].parameters())
    print(f"{'features[0] (Stem)':<28} {'Conv3x3 + BatchNorm (3 -> 32)':<32} {format_num(stem_params):>12} {stem_params / total_params * 100:>11.2f}%")

    # 2. MBConv Blocks (1 to 7)
    block_names = [
        "MBConv1 (k3x3, exp 1, 32->16)",
        "MBConv6 (k3x3, exp 6, 16->24)",
        "MBConv6 (k5x5, exp 6, 24->40)",
        "MBConv6 (k3x3, exp 6, 40->80)",
        "MBConv6 (k5x5, exp 6, 80->112)",
        "MBConv6 (k5x5, exp 6, 112->192)",
        "MBConv6 (k3x3, exp 6, 192->320)",
    ]

    for idx, name in enumerate(block_names, start=1):
        block_layer = features[idx]
        b_params = sum(p.numel() for p in block_layer.parameters())
        pct = (b_params / total_params) * 100
        print(f"{f'features[{idx}] ({name.split()[0]})':<28} {name:<32} {format_num(b_params):>12} {pct:>11.2f}%")

    # 3. Head Conv (features[8])
    head_conv_params = sum(p.numel() for p in features[8].parameters())
    print(f"{'features[8] (Head Conv)':<28} {'Conv1x1 + BN (320 -> 1280)':<32} {format_num(head_conv_params):>12} {head_conv_params / total_params * 100:>11.2f}%")

    # 4. Classifier Head
    clf_params = sum(p.numel() for p in classifier.parameters())
    print(f"{'classifier (Linear Head)':<28} {'Dropout(0.3) + Linear(1280->9)':<32} {format_num(clf_params):>12} {clf_params / total_params * 100:>11.2f}%")

    print("=" * 90)

    # Executive Summary Card
    print(f"\n{C_BOLD}{C_BLUE}--- Model Summary & Hardware Metrics ---{C_RESET}")
    print(f"  * {C_BOLD}Total Parameters:{C_RESET}          {C_GREEN}{format_num(total_params)}{C_RESET} ({total_params / 1e6:.2f} Million)")
    print(f"  * {C_BOLD}Trainable Parameters:{C_RESET}      {format_num(trainable_params)} (100.0%)")
    print(f"  * {C_BOLD}Non-Trainable Parameters:{C_RESET}  {format_num(non_trainable_params)}")
    print(f"  * {C_BOLD}Embedding Dimension:{C_RESET}       {model.embedding_dim}-D vector")
    print(f"  * {C_BOLD}Memory Footprint (FP32):{C_RESET}   {format_bytes(total_mem_bytes)}")
    print(f"  * {C_BOLD}Memory Footprint (FP16):{C_RESET}   {format_bytes(total_mem_bytes // 2)} (Active in Mixed Precision)")
    print(f"  * {C_BOLD}Estimated Compute (MACs):{C_RESET}  ~0.39 Giga-FLOPs (Inference latency < 50ms)")
    print("=" * 90 + "\n")

    return {
        "architecture": "efficientnet_b0",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "embedding_dim": model.embedding_dim,
        "memory_fp32_mb": total_mem_bytes / (1024 ** 2),
        "memory_fp16_mb": (total_mem_bytes / 2) / (1024 ** 2),
    }


if __name__ == "__main__":
    inspect_model_parameters()
