"""
Unit tests for BnF Gallica IIIF client and Medieval Scribal Neural Trainer.
"""

import pytest
import numpy as np
from voynich_ductus.ingestion.gallica_client import GallicaManuscriptClient
from voynich_ductus.models.medieval_trainer import MedievalScribalTrainer
from voynich_ductus.embeddings.scribal_neural_net import (
    ScribalKinematicFlowNet,
    ScribalContrastiveEmbeddingNet
)


def test_gallica_client_ark_normalization():
    client = GallicaManuscriptClient()
    
    ark1 = client.normalize_ark("latin_6823")
    assert ark1 == "ark:/12148/btv1b52501620s"

    ark2 = client.normalize_ark("btv1b8451106z")
    assert ark2 == "ark:/12148/btv1b8451106z"

    url = client.get_page_iiif_url("latin_6823", page_num=15, width=1800)
    assert "https://gallica.bnf.fr/iiif/12148/btv1b52501620s/f15/full/1800,/0/native.jpg" in url


def test_gallica_client_synthetic_fallback():
    client = GallicaManuscriptClient(cache_dir="output/test_gallica_cache")
    # Test downloading synthetic fallback folio
    img = client.load_folio_image("latin_6823", page_num=12, target_width=1000)
    assert img is not None
    assert img.width > 500
    assert img.height > 500


def test_medieval_trainer_synthetic_kinematic_batch():
    trainer = MedievalScribalTrainer(device="cpu")
    images, flows, touchdowns, penlifts = trainer.generate_synthetic_kinematic_batch(batch_size=4, patch_size=32)

    assert images.shape == (4, 1, 32, 32)
    assert flows.shape == (4, 2, 32, 32)
    assert touchdowns.shape == (4, 1, 32, 32)
    assert penlifts.shape == (4, 1, 32, 32)
    assert np.max(images) > 0.5


def test_medieval_trainer_contrastive_triplet_batch():
    trainer = MedievalScribalTrainer(device="cpu")
    anchors, positives, negatives = trainer.generate_contrastive_triplet_batch(batch_size=4, patch_size=32)

    assert anchors.shape == (4, 1, 32, 32)
    assert positives.shape == (4, 1, 32, 32)
    assert negatives.shape == (4, 1, 32, 32)


def test_scribal_flow_net_and_contrastive_training_step():
    trainer = MedievalScribalTrainer(device="cpu")
    
    # 1. Flow Net
    flow_net = ScribalKinematicFlowNet(device="cpu")
    if flow_net._model is not None:
        import torch
        opt = torch.optim.Adam(flow_net._model.parameters(), lr=1e-3)
        res_flow = trainer.train_flow_net_step(flow_net, opt, batch_size=4)
        assert "total_loss" in res_flow
        assert res_flow["total_loss"] >= 0.0

    # 2. Contrastive Net
    emb_net = ScribalContrastiveEmbeddingNet(embedding_dim=64, device="cpu")
    if emb_net._model is not None:
        import torch
        opt2 = torch.optim.Adam(emb_net._model.parameters(), lr=1e-3)
        res_cont = trainer.train_contrastive_step(emb_net, opt2, batch_size=4, margin=0.2)
        assert "triplet_loss" in res_cont
        assert "triplet_accuracy" in res_cont
