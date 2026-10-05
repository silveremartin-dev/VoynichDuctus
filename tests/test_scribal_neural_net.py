"""
Unit tests for Scribal Kinematic Flow Net and Contrastive Triplet Embedding Net.
"""

import numpy as np
import pytest
from voynich_ductus.embeddings.scribal_neural_net import (
    ScribalKinematicFlowNet,
    ScribalContrastiveEmbeddingNet
)


def test_scribal_kinematic_flow_net():
    flow_net = ScribalKinematicFlowNet(device="cpu")
    
    # Synthetic stroke patch (vertical stroke with thickness)
    patch = np.zeros((32, 32), dtype=np.uint8)
    patch[5:28, 14:18] = 255

    res = flow_net.predict(patch)
    assert "flow_field" in res
    assert "touchdown_prob" in res
    assert "penlift_prob" in res

    assert res["flow_field"].shape == (32, 32, 2)
    assert res["touchdown_prob"].shape == (32, 32)
    assert res["penlift_prob"].shape == (32, 32)


def test_scribal_contrastive_embedding_net():
    emb_net = ScribalContrastiveEmbeddingNet(embedding_dim=128, device="cpu")
    
    patch_a = np.zeros((32, 32), dtype=np.uint8)
    patch_a[8:24, 12:20] = 255
    
    patch_b = np.zeros((32, 32), dtype=np.uint8)
    patch_b[10:26, 14:22] = 255

    vec_a = emb_net.embed_patch(patch_a)
    vec_b = emb_net.embed_patch(patch_b)

    assert vec_a.shape == (128,)
    assert pytest.approx(float(np.linalg.norm(vec_a)), 0.05) == 1.0


def test_triplet_loss_computation():
    emb_net = ScribalContrastiveEmbeddingNet()
    a = np.array([1.0, 0.0, 0.0])
    p = np.array([0.95, 0.05, 0.0])
    n = np.array([0.0, 1.0, 0.0])

    loss = emb_net.compute_triplet_loss(a, p, n, margin=0.3)
    assert isinstance(loss, float)
    assert loss >= 0.0
