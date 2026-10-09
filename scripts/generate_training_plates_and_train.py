"""
Generates visual inspection plates of synthetic scribal kinematic batches and trains ScribalKinematicFlowNet and ScribalContrastiveEmbeddingNet.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from voynich_ductus.models.medieval_trainer import MedievalScribalTrainer
from voynich_ductus.embeddings.scribal_neural_net import (
    ScribalKinematicFlowNet,
    ScribalContrastiveEmbeddingNet
)


def render_synthetic_kinematic_plate(output_dir: Path):
    """
    Renders a comprehensive high-res visualization plate showing synthetic scribal training batches:
    1. Raw Calligraphic Ink Patches
    2. Tangent Flow Field Quiver / HSV Vector Map
    3. Touchdown Map P(t=0)
    4. Pen-Lift Map P(t=1)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = MedievalScribalTrainer(device="cpu")
    images, flows, touchdowns, penlifts = trainer.generate_synthetic_kinematic_batch(batch_size=8, patch_size=64)

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.patch.set_facecolor("#16181d")

    archetype_names = ["Minim 'i'", "Loop 'o'", "Ascender 'l'", "Gallows 't'", "Minim 'i'", "Loop 'o'", "Ascender 'l'", "Gallows 't'"]

    for col in range(8):
        # 1. Ink image
        ax_img = axes[0, col]
        ax_img.imshow(images[col, 0], cmap="copper", origin="upper")
        ax_img.set_title(f"{archetype_names[col]}\n(Ink Patch)", color="#e2e8f0", fontsize=10)
        ax_img.axis("off")

        # 2. Flow Vector Quiver
        ax_flow = axes[1, col]
        u = flows[col, 0]  # dx
        v = -flows[col, 1] # -dy for matplotlib coords
        y, x = np.mgrid[0:64:4, 0:64:4]
        u_sub = u[::4, ::4]
        v_sub = v[::4, ::4]
        ax_flow.imshow(images[col, 0], cmap="gray", alpha=0.3, origin="upper")
        ax_flow.quiver(x, y, u_sub, v_sub, color="#38bdf8", scale=15, width=0.008)
        ax_flow.set_title("Flow Field u(x,y)", color="#38bdf8", fontsize=10)
        ax_flow.axis("off")

        # 3. Touchdown Map
        ax_td = axes[2, col]
        ax_td.imshow(touchdowns[col, 0], cmap="magma", origin="upper")
        ax_td.set_title("Touchdown P(t=0)", color="#f43f5e", fontsize=10)
        ax_td.axis("off")

        # 4. Pen-lift Map
        ax_pl = axes[3, col]
        ax_pl.imshow(penlifts[col, 0], cmap="viridis", origin="upper")
        ax_pl.set_title("Pen-Lift P(t=1)", color="#10b981", fontsize=10)
        ax_pl.axis("off")

    plt.suptitle("Synthetic Scribal Kinematic Training Batches (Planches de Calibration Cinématique)", color="#f8fafc", fontsize=16, y=0.98)
    plt.tight_layout()
    plate_path = output_dir / "synthetic_kinematic_batch_plate.png"
    plt.savefig(plate_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Synthetic Kinematic Batch Plate to: {plate_path}")


def render_contrastive_triplet_plate(output_dir: Path):
    """
    Renders Anchor / Positive (augmented with angle & parchment noise) / Negative triplet batches.
    """
    trainer = MedievalScribalTrainer(device="cpu")
    anchors, positives, negatives = trainer.generate_contrastive_triplet_batch(batch_size=6, patch_size=64)

    fig, axes = plt.subplots(6, 3, figsize=(9, 14))
    fig.patch.set_facecolor("#16181d")

    col_titles = ["Anchor Glyph (A)", "Positive Sample (P+)", "Negative Sample (N-)"]

    for row in range(6):
        axes[row, 0].imshow(anchors[row, 0], cmap="copper", origin="upper")
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title(col_titles[0], color="#38bdf8", fontsize=12, pad=8)

        axes[row, 1].imshow(positives[row, 0], cmap="copper", origin="upper")
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title(col_titles[1], color="#10b981", fontsize=12, pad=8)

        axes[row, 2].imshow(negatives[row, 0], cmap="copper", origin="upper")
        axes[row, 2].axis("off")
        if row == 0:
            axes[row, 2].set_title(col_titles[2], color="#f43f5e", fontsize=12, pad=8)

    plt.suptitle("Scribal Triplet Contrastive Batches (Margin Loss D(A,P) + m < D(A,N))", color="#f8fafc", fontsize=15, y=0.98)
    plt.tight_layout()
    plate_path = output_dir / "contrastive_triplet_plate.png"
    plt.savefig(plate_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Contrastive Triplet Plate to: {plate_path}")


def train_and_evaluate_models():
    """
    Executes a real training loop on ScribalKinematicFlowNet and ScribalContrastiveEmbeddingNet.
    """
    print("\n" + "="*60)
    print(" TRAINING SCRIBAL KINEMATIC FLOW NET & CONTRASTIVE EMBEDDING NET")
    print("="*60)

    try:
        import torch
        import torch.optim as optim
    except ImportError:
        print("[!] PyTorch not installed. Skipping live training step.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on device: {device}")

    # 1. Train Flow Net
    flow_model = ScribalKinematicFlowNet()
    flow_model._model.to(device)
    optimizer_flow = optim.Adam(flow_model._model.parameters(), lr=1e-3)
    trainer = MedievalScribalTrainer(device=device)

    print("\n--- Training ScribalKinematicFlowNet (50 Epochs) ---")
    for epoch in range(1, 51):
        loss_dict = trainer.train_flow_net_step(flow_model, optimizer_flow, batch_size=16)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/50: Total Loss = {loss_dict.get('total_loss', 0.0):.4f} | Cosine Loss = {loss_dict.get('flow_cosine_loss', 0.0):.4f} | TD Loss = {loss_dict.get('touchdown_loss', 0.0):.4f} | PL Loss = {loss_dict.get('penlift_loss', 0.0):.4f}")

    # 2. Train Contrastive Embedding Net
    emb_model = ScribalContrastiveEmbeddingNet(embedding_dim=64)
    emb_model._model.to(device)
    optimizer_emb = optim.Adam(emb_model._model.parameters(), lr=1e-3)

    print("\n--- Training ScribalContrastiveEmbeddingNet (50 Epochs) ---")
    for epoch in range(1, 51):
        loss_dict = trainer.train_contrastive_step(emb_model, optimizer_emb, batch_size=16, margin=1.0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/50: Triplet Loss = {loss_dict.get('triplet_loss', 0.0):.4f} | Dist(A, P) = {loss_dict.get('pos_dist', 0.0):.4f} | Dist(A, N) = {loss_dict.get('neg_dist', 0.0):.4f}")

    # Save trained checkpoint
    save_dir = Path("output/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(flow_model._model.state_dict(), save_dir / "scribal_flow_net.pt")
    torch.save(emb_model._model.state_dict(), save_dir / "scribal_contrastive_net.pt")
    print(f"\n[+] Saved trained model weights to: {save_dir}")


if __name__ == "__main__":
    out_dir = Path("output/training_plates")
    render_synthetic_kinematic_plate(out_dir)
    render_contrastive_triplet_plate(out_dir)
    train_and_evaluate_models()
