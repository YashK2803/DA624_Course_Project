"""
Generates clean, publication-grade architectural diagrams for Token-Level Fusion (LLaVA)
and embedding into the DA624 course presentation slides.
Author: Vaishnavi Agarwal (Roll No: 230150028)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_llava_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Styles
    box_blue = dict(boxstyle="round,pad=0.5", fc="#EBF3FB", ec="#3182CE", lw=1.8)
    box_frozen = dict(boxstyle="round,pad=0.5", fc="#F0F4F8", ec="#718096", lw=1.8)
    box_green = dict(boxstyle="round,pad=0.5", fc="#E6FFFA", ec="#319795", lw=1.8)
    box_purple = dict(boxstyle="round,pad=0.5", fc="#FAF5FF", ec="#805AD5", lw=1.8)
    box_red = dict(boxstyle="round,pad=0.5", fc="#FFF5F5", ec="#E53E3E", lw=1.8)
    box_orange = dict(boxstyle="round,pad=0.5", fc="#FFFAF0", ec="#DD6B20", lw=1.8)

    # 1. Input Image
    ax.text(1.1, 4.3, "Input Image\n(336 × 336 px)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=box_blue)
    
    # Grid of patches
    rect = patches.Rectangle((0.6, 1.4), 1.0, 1.0, linewidth=1.5, edgecolor="#4A5568", facecolor="#E2E8F0", linestyle="--")
    ax.add_patch(rect)
    for i in range(1, 4):
        ax.plot([0.6 + i*0.25, 0.6 + i*0.25], [1.4, 2.4], color="#718096", lw=1)
        ax.plot([0.6, 1.6], [1.4 + i*0.25, 1.4 + i*0.25], color="#718096", lw=1)
    ax.text(1.1, 0.95, "24 × 24 = 576\nPatches (14×14)", ha="center", va="center", fontsize=9.5, color="#4A5568")

    # Arrow to Vision Encoder
    ax.annotate("", xy=(2.4, 3.4), xytext=(1.9, 4.0), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))
    ax.annotate("", xy=(2.4, 3.0), xytext=(1.8, 2.0), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))

    # 2. Vision Encoder (Frozen)
    ax.text(3.3, 3.2, "Vision Encoder\nCLIP ViT-L/14\n(Frozen)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=box_frozen)
    ax.text(3.3, 2.1, "[★ Weights Frozen\nNo compute cost]", ha="center", va="center", fontsize=8.5, color="#718096", fontstyle="italic")

    # Arrow to Visual Tokens
    ax.annotate("", xy=(4.9, 3.2), xytext=(4.2, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))

    # 3. Dense Visual Tokens
    ax.text(5.6, 3.2, "576 Visual Tokens\nDimension = 1024\n(Dense 24×24 Grid)", ha="center", va="center", fontsize=10.5, fontweight="bold", bbox=box_green)

    # Arrow to Multimodal Projector
    ax.annotate("", xy=(7.0, 3.2), xytext=(6.4, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color="#319795"))

    # 4. Multimodal Projector (Trained)
    ax.text(7.9, 3.2, "Multimodal Projector\n2-Layer MLP (GELU)\n(1024 → 4096)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=box_purple)
    ax.text(7.9, 2.1, "[✓ Trainable in QLoRA\nlr = 2e-5]", ha="center", va="center", fontsize=8.5, color="#805AD5", fontweight="bold")

    # Arrow to LLM
    ax.annotate("", xy=(9.4, 3.2), xytext=(8.8, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color="#805AD5"))

    # 5. Text Instruction Input
    ax.text(7.9, 5.0, "Prompt: 'Is this image blurry?'\nText Tokens (dim 4096)", ha="center", va="center", fontsize=10, fontweight="bold", bbox=box_orange)
    ax.annotate("", xy=(9.5, 3.6), xytext=(8.9, 4.8), arrowprops=dict(arrowstyle="->", lw=2, color="#DD6B20"))

    # 6. Language Model (Vicuna-7B)
    ax.text(10.5, 3.2, "Language Model\nVicuna-7B-v1.5\n(4-bit NF4 + LoRA)", ha="center", va="center", fontsize=11, fontweight="bold", bbox=box_red)
    ax.text(10.5, 2.1, "[✓ Rank-16 Adapters\nlr = 2e-4, 33.5M Params]", ha="center", va="center", fontsize=8.5, color="#E53E3E", fontweight="bold")

    # Arrow to Output
    ax.annotate("", xy=(11.8, 3.2), xytext=(11.3, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color="#E53E3E"))
    ax.text(11.4, 1.2, "Diagnostic Answer:\n'Yes, severe motion blur'\nAcc: 67.1% (Img Gap: 21.9%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2D3748")

    # Title Banner inside diagram
    ax.text(6.0, 5.7, "LLaVA Token-Level Fusion Architecture for Quality Diagnostics", ha="center", va="center", fontsize=14, fontweight="bold", color="#1A202C")

    diag_path = os.path.join(OUTPUT_DIR, "llava_architecture.png")
    plt.tight_layout()
    plt.savefig(diag_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully generated LLaVA Architecture diagram: {diag_path}")

def create_bottleneck_comparison_diagram():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    for ax in (ax1, ax2):
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 2.6)
        ax.axis("off")
        ax.set_facecolor('#FFFFFF')

    box_eva = dict(boxstyle="round,pad=0.4", fc="#EDF2F7", ec="#4A5568", lw=1.5)
    box_qformer = dict(boxstyle="round,pad=0.4", fc="#FFF5F5", ec="#E53E3E", lw=1.8)
    box_opt = dict(boxstyle="round,pad=0.4", fc="#FAF5FF", ec="#805AD5", lw=1.5)

    box_clip = dict(boxstyle="round,pad=0.4", fc="#EDF2F7", ec="#4A5568", lw=1.5)
    box_mlp = dict(boxstyle="round,pad=0.4", fc="#E6FFFA", ec="#319795", lw=1.8)
    box_vicuna = dict(boxstyle="round,pad=0.4", fc="#FAF5FF", ec="#805AD5", lw=1.5)

    # Subplot 1: Mid-Fusion (BLIP-2)
    ax1.text(0.5, 2.2, "(A) Mid-Fusion (BLIP-2): The 14.7× Information Bottleneck", fontsize=11, fontweight="bold", color="#C53030")
    ax1.text(1.5, 1.2, "EVA ViT-g\n257 tokens\n(dim 1408)", ha="center", va="center", fontsize=9.5, bbox=box_eva)
    ax1.annotate("", xy=(3.0, 1.2), xytext=(2.2, 1.2), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))

    ax1.text(4.4, 1.2, "Q-Former Bottleneck\n32 Query Tokens\n(14.7× Compression)", ha="center", va="center", fontsize=9.5, fontweight="bold", bbox=box_qformer)
    ax1.annotate("", xy=(6.5, 1.2), xytext=(5.8, 1.2), arrowprops=dict(arrowstyle="->", lw=2, color="#E53E3E"))

    ax1.text(7.6, 1.2, "OPT-2.7B LLM\n(32 visual tokens)", ha="center", va="center", fontsize=9.5, bbox=box_opt)
    ax1.annotate("", xy=(9.2, 1.2), xytext=(8.4, 1.2), arrowprops=dict(arrowstyle="->", lw=2, color="#805AD5"))

    ax1.text(9.9, 1.2, "Result:\nQ-Bench: 55.3%\nFine texture smoothed", ha="center", va="center", fontsize=8.5, color="#742A2A", fontweight="bold")

    # Subplot 2: Token-Level Fusion (LLaVA)
    ax2.text(0.5, 2.2, "(B) Token-Level Fusion (LLaVA): Uncompressed Spatial Fidelity", fontsize=11, fontweight="bold", color="#234E52")
    ax2.text(1.5, 1.2, "CLIP ViT-L/14\n576 tokens\n(dim 1024)", ha="center", va="center", fontsize=9.5, bbox=box_clip)
    ax2.annotate("", xy=(3.0, 1.2), xytext=(2.2, 1.2), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))

    ax2.text(4.4, 1.2, "2-Layer MLP Projector\n576 tokens preserved\n(No Bottleneck!)", ha="center", va="center", fontsize=9.5, fontweight="bold", bbox=box_mlp)
    ax2.annotate("", xy=(6.5, 1.2), xytext=(5.8, 1.2), arrowprops=dict(arrowstyle="->", lw=2, color="#319795"))

    ax2.text(7.6, 1.2, "Vicuna-7B LLM\n(576 visual tokens)", ha="center", va="center", fontsize=9.5, bbox=box_vicuna)
    ax2.annotate("", xy=(9.2, 1.2), xytext=(8.4, 1.2), arrowprops=dict(arrowstyle="->", lw=2, color="#805AD5"))

    ax2.text(9.9, 1.2, "Result:\nQ-Bench: 67.1%\nHigh-frequency preserved", ha="center", va="center", fontsize=8.5, color="#1D4044", fontweight="bold")

    diag_path = os.path.join(OUTPUT_DIR, "fusion_bottleneck_comparison.png")
    plt.tight_layout()
    plt.savefig(diag_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully generated Bottleneck Comparison diagram: {diag_path}")

if __name__ == "__main__":
    create_llava_architecture_diagram()
    create_bottleneck_comparison_diagram()
