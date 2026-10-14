# Google Cloud Platform (GCP) — Batch Execution Architecture

Ce sous-dossier `scripts/gcp/` contient l'infrastructure complète pour déployer et exécuter la vectorisation cinématique et l'induction des alphabets des manuscrits (Voynich MS 408 et Codex Seraphinianus) à l'échelle industrielle sur Google Cloud Platform (projet GCP `ether-509812`).

---

## 1. Vue d'ensemble de l'architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Google Cloud Platform                           │
└────────────────────────────────────────────────────────────────────────┘
                                    │
       ┌────────────────────────────┴───────────────────────────┐
       ▼                                                        ▼
┌───────────────────────────────┐        ┌──────────────────────────────┐
│  Option A: Cloud Run Jobs     │   OU   │  Option B: Compute Engine    │
│  - 8 vCPUs / 16 GiB RAM       │        │  - Spot VM (c2d-standard-16) │
│  - Serverless, zéro config    │        │  - 16 cœurs, ultra-rapide    │
│  - Idéal pour lots réguliers  │        │  - Coût: < 0,15 $ par run    │
└───────────────────────────────┘        └──────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Orchestrateur: scripts/gcp/cloud_batch_runner.py                      │
│  - ProcessPoolExecutor multi-cœurs (1 folio/page par cœur)             │
│  - Inférence DINOv2 par batch (tensorié, 80x plus rapide)             │
│  - Binarisation hystérésis & Seam Carving sans déchet                  │
│  - Génération du catalogue JSON et du Grand Explorateur HTML           │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Google Cloud Storage (GCS)                                            │
│  gs://<VOTRE_BUCKET>/voynich_ductus/atlas_dataset/                     │
│  ├── grand_glyph_explorer.html                                         │
│  ├── voynich/ (catalogue JSON, planches, SVG, PNG)                     │
│  └── seraphinianus/ (catalogue JSON, planches, SVG, PNG)               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Performances et Coûts Comparatifs

| Mode d'exécution | Cœurs CPU | Temps Voynich (220 folios) | Temps Seraphinianus (371 planches) | Durée totale | Coût estimé |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local séquentiel** | 1 cœur | ~2h 20 min | ~10h 40 min | **~13 heures** | 0 € |
| **Local multi-process** | 8 cœurs | ~22 minutes | ~1h 15 min | **~1h 37 min** | 0 € |
| **GCP Spot VM (`c2d-16`)** | **16 cœurs** | **~11 minutes** | **~38 minutes** | **~49 minutes** | **~0,12 $** |

---

## 3. Structure des Fichiers

```
scripts/gcp/
├── Dockerfile              # Image Debian Linux avec Python 3.11, PyTorch & Poppler
├── requirements-gcp.txt    # Dépendances spécifiques (google-cloud-storage, batch)
├── cloud_batch_runner.py   # Script d'orchestration parallèle multi-process
├── deploy.sh               # Script de build et déploiement Bash (Linux / Cloud Shell)
├── deploy.ps1              # Script de build et déploiement PowerShell (Windows)
└── README.md               # Guide d'architecture et de référence
```

---

## 4. Sécurité & Bonnes Pratiques Git (`.gitignore`)

Conformément aux règles du projet, les règles suivantes sont actives dans `.gitignore` :
- Tous les fichiers de clés et comptes de service GCP (`*service_account*.json`, `*credentials*.json`, `gcp/*.json`, `gcp/*.key`, `gcp/credentials/`) sont **strictement ignorés**.
- Les variables d'environnement (`gcp/.env*`) sont ignorées.
- Aucun fichier de scan lourd ou binaire temporaire n'est synchronisé dans le dépôt Git.

---

## 5. Guide de Déploiement Pas-à-Pas

### Pré-requis
Avoir le SDK Google Cloud installé et être authentifié :
```bash
gcloud auth login
gcloud config set project <VOTRE_PROJECT_ID>
```

### Méthode 1 : Lancement avec le script de déploiement (Recommandé)

**Sous Windows (PowerShell) :**
```powershell
# Déploiement en Cloud Run Job serverless (8 vCPUs)
.\gcp\deploy.ps1 -Mode cloud-run

# OU déploiement sur VM Spot 16 cœurs ultra-rapide
.\gcp\deploy.ps1 -Mode spot-vm
```

**Sous Linux / Mac / Google Cloud Shell (Bash) :**
```bash
chmod +x gcp/deploy.sh
# Cloud Run Job :
./gcp/deploy.sh cloud-run

# Spot VM :
./gcp/deploy.sh spot-vm
```

### Méthode 2 : Exécution locale avec Docker
Pour tester l'image avant l'envoi sur GCP :
```bash
docker build -t voynich-ductus-batch -f gcp/Dockerfile .
docker run --rm -v $(pwd)/output:/app/output voynich-ductus-batch --workers 4 --voynich-limit 2
```

### Méthode 3 : Récupération des résultats depuis Google Cloud Storage
Une fois le batch terminé, téléchargez le Grand Explorateur et les catalogues :
```bash
gsutil -m rsync -r gs://<VOTRE_BUCKET>/voynich_ductus/atlas_dataset/ ./output/atlas_dataset/
```
Puis ouvrez `output/atlas_dataset/grand_glyph_explorer.html` dans votre navigateur.
