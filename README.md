# Pakistani Public Figures Image Classification

Production-grade, end-to-end image classification system that identifies
**12 Pakistani public figures** (politicians + military spokespersons) from facial
images. Built as a university semester project (ANN + MLOps, Category A).

The differentiator is the **data pipeline** — multi-source scraping, MTCNN face
validation, perceptual-hash deduplication, blur/resolution filtering, and
cross-class duplicate auto-removal. Models are trained on validated data only.

## Final results

| Model | Test Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| ResNet-50 (ImageNet pretrained) | 59.46% | 0.580 | 0.547 |
| EfficientNet-B3 (ImageNet pretrained) | 59.46% | 0.592 | 0.540 |
| **FaceNet (VGGFace2 pretrained)** | **70.27%** | **0.680** | **0.658** |
| 3-way ensemble (equal weights) | 68.92% | 0.674 | 0.657 |
| **Weighted ensemble** (FaceNet 0.70, EffNet 0.15, ResNet 0.15) | **71.62%** | **0.697** | **0.687** |

**Key finding:** face-pretrained backbones dominate ImageNet-pretrained
backbones on this task. FaceNet (InceptionResnetV1 pretrained on VGGFace2's
3.3M celebrity faces) outperforms ResNet-50 and EfficientNet-B3 by ~11 points
absolute. This is consistent with the transfer-learning literature:
domain-matched pretraining beats generic pretraining when target-task data
is limited.

## Architecture

```
+--------------+    +--------------+    +-----------+    +----------+
|  Scrapers    | -> |  Validation  | -> | Split &   | -> | Training |
|  (Google,    |    | (MTCNN,      |    | Augment   |    | (ResNet, |
|   Bing, DDG, |    |  pHash,      |    | (75/15/10)|    | EffNet,  |
|   Wiki,      |    |  blur, dim,  |    +-----------+    | FaceNet) |
|   News,      |    |  cross-class |          |          +-----+----+
|   Selenium)  |    |  dedup)      |          |                |
+------+-------+    +--------------+          v                v
       |                                +----------+      +----------+
       v                                | Eval +   |      | MLflow   |
   data/raw/  --[DVC]--> data/validated/| GradCAM  |      | tracking |
                                        +----------+      +----------+
                                              |                |
                                              v                v
                                        +----------+      +----------+
                                        | FastAPI  | <--- | Best     |
                                        | + Streamlit     | model    |
                                        +----------+      +----------+
                                              |
                                              v
                                  Docker / EC2 / Airflow retrain DAG
```

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Scraping | icrawler, wikipedia-api, duckduckgo-search, BeautifulSoup4, Selenium |
| Face validation | facenet-pytorch (MTCNN + InceptionResnetV1) |
| Dedup / Quality | imagehash, OpenCV |
| Deep learning | PyTorch, torchvision, facenet-pytorch |
| Augmentation | albumentations |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Orchestration | Apache Airflow |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Deployment | AWS EC2 |
| Testing | pytest |
| Linting | ruff |

## The 12 classes

| # | Class | Display name | Validated images |
|---|---|---|---|
| 1 | ahmed_sharif_chaudhry | Ahmed Sharif Chaudhry (DG ISPR) | 41 |
| 2 | ahsan_iqbal | Ahsan Iqbal | 53 |
| 3 | asif_ali_zardari | Asif Ali Zardari | 88 |
| 4 | benazir_bhutto | Benazir Bhutto | 83 |
| 5 | bilawal_bhutto_zardari | Bilawal Bhutto Zardari | 63 |
| 6 | hamza_shehbaz | Hamza Shehbaz | 54 |
| 7 | imran_khan | Imran Khan | 73 |
| 8 | maryam_nawaz_sharif | Maryam Nawaz Sharif | 64 |
| 9 | murad_ali_shah | Murad Ali Shah | 30 |
| 10 | nawaz_sharif | Nawaz Sharif | 86 |
| 11 | shehbaz_sharif | Shehbaz Sharif | 71 |
| 12 | yousaf_raza_gillani | Yousaf Raza Gillani | 29 |

**Total: 735 validated images** (scraped 3,775 raw, ~80% rejected by the
validation pipeline).

Splits: 75% train (550), 15% val (111), 10% test (74). Stratified, seed 42.

## Per-class F1 (FaceNet)

| Class | F1 | Status |
|---|---|---|
| benazir_bhutto | 0.93 | ✓ Excellent |
| asif_ali_zardari | 0.84 | ✓ Excellent |
| ahsan_iqbal | 0.80 | ✓ Strong |
| nawaz_sharif | 0.80 | ✓ Strong |
| maryam_nawaz_sharif | 0.77 | ✓ Strong |
| yousaf_raza_gillani | 0.75 | ✓ Good |
| bilawal_bhutto_zardari | 0.67 | OK |
| murad_ali_shah | 0.67 | OK (was 0.00 with ImageNet backbones) |
| shehbaz_sharif | 0.67 | OK |
| imran_khan | 0.44 | Weak — data quality issue |
| ahmed_sharif_chaudhry | 0.33 | Weak — only 41 validated |
| hamza_shehbaz | 0.22 | Weak — family resemblance |

**Failure modes:**

1. **imran_khan (F1 0.44)** — dataset contains a mix of cricket-era (1980s-90s)
   and political-era (2010s-2020s) Imran Khan photos. The visual gap between
   these is large enough that the model cannot reconcile both within a single
   class.
2. **hamza_shehbaz (F1 0.22)** — family resemblance with Maryam Nawaz Sharif
   (cousin) and Shehbaz Sharif (father). With only 54 validated training
   images, the model lacks discriminative features.
3. **ahmed_sharif_chaudhry (F1 0.33)** — smallest validated count (41) and
   limited unique-pose variation. Confused with other clean-shaven, mustached
   men in the dataset.
## Pretrained models

The trained checkpoints are hosted on Hugging Face Hub (too large for git):

```bash
pip install huggingface_hub
mkdir -p models/saved
hf download Maarij-Aqeel/pak-faces-classifier \
    --include "*.pt" --local-dir models/saved/
```

Or browse the repo at <https://huggingface.co/Maarij-Aqeel/pak-faces-classifier>.

## Quick start (local)

```bash
git clone https://github.com/Maarij-Aqeel/pak-public-figure-classifier.git
cd pak-public-figure-classifier

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install facenet-pytorch

# Scrape (target ~300 raw per class)
python -m scripts.run_data_collection --target-count 300

# Validate (MTCNN + dedup + quality + cross-class dedup)
python -m scripts.run_validation --skip-outliers --auto-remove-cross-class

# Stratified split
python -m scripts.run_split --force

# Train all 3 models (set MLFLOW_TRACKING_URI for local file backend)
export MLFLOW_TRACKING_URI=file:./mlruns
python -m scripts.run_training

# Evaluate
python -m scripts.run_evaluation --model facenet_vggface2 --skip-gradcam

# Serve
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
streamlit run frontend/app.py
```

## Docker quick start

```bash
docker-compose up --build
# API:        http://localhost:8000
# Frontend:   http://localhost:8501
# MLflow UI:  http://localhost:5000
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + readiness |
| GET | `/classes` | List all 12 classes |
| POST | `/predict` | Single-image top-3 prediction |
| POST | `/predict-batch` | Multi-image prediction |

### Example

```bash
curl -F "file=@portrait.jpg" \
     "http://localhost:8000/predict?model=facenet_vggface2"
```

```json
{
  "predictions": [
    {"class": "imran_khan", "display_name": "Imran Khan", "confidence": 0.87},
    {"class": "shehbaz_sharif", "display_name": "Shehbaz Sharif", "confidence": 0.06},
    {"class": "nawaz_sharif", "display_name": "Nawaz Sharif", "confidence": 0.03}
  ],
  "model_used": "facenet_vggface2",
  "face_detected": true,
  "face_bbox": [120, 80, 360, 320],
  "inference_time_ms": 92.4
}
```

## Path to 90%+ accuracy

The ceiling at 71.62% reflects three structural constraints:

1. **Small dataset** — 735 validated images for 12 classes (~46 train/class average).
2. **Imran Khan data quality** — re-scrape with "Imran Khan PTI" /
   "Imran Khan prime minister" queries to exclude cricket-era images (~+3-5%).
3. **Family resemblance pairs** — Nawaz↔Maryam, Shehbaz↔Hamza, Bhutto family.
   Margin-based losses (ArcFace) would help (~+2-4%).
4. **Full 30-class coverage** — the original project spec was 30 figures; the
   current 12 is what one team member collected. With teammates' contributions,
   total training data ~3-4× larger.

Combined potential: ~85-92% with the full dataset + Imran data cleanup +
ArcFace.

## DVC pipeline

```
collect → validate → split → train
   |          |        |       |
   v          v        v       v
data/raw  validated  splits  models/saved
```

Run end-to-end via `dvc repro`.

## CI / CD

| Workflow | Trigger | Actions |
|---|---|---|
| `.github/workflows/ci.yml` | push, PR | ruff + pytest + contributor distribution check |
| `.github/workflows/cd.yml` | push to `main` | Docker build, smoke test, EC2 deploy |

## Project structure

```
pak-public-figures-classifier/
├── .github/workflows/      # CI + CD
├── airflow/dags/           # Weekly retrain DAG
├── data/                   # raw/, validated/, splits/, metadata/ (DVC)
├── models/saved/           # checkpoints (DVC)
├── src/
│   ├── config.py
│   ├── data_collection/    # all scrapers
│   ├── data_validation/    # face + dedup + quality + outlier
│   ├── data_preprocessing/ # split, augment, dataset
│   ├── models/             # ResNet, EfficientNet, FaceNet, ViT, trainer
│   ├── evaluation/         # metrics, confusion matrix, Grad-CAM
│   ├── api/                # FastAPI
│   └── utils/
├── frontend/app.py         # Streamlit
├── scripts/                # CLI entry points
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

## Team Members + Instructors

| Role | Name | GitHub |
|---|---|---|
| Member | Abdullah Khan Niazi | [@Abdullah-Khan-Niazi](https://github.com/Abdullah-Khan-Niazi) |
| Member | Raza Sherazi | [@RazaSherazi09](https://github.com/RazaSherazi09) |
| Member | Maarij Aqeel | [@Maarij-Aqeel](https://github.com/Maarij-Aqeel) |
| Instructor (ANN + MLOps) | Sir Asif Ameer | [@asif370](https://github.com/asif370) |
| TA (ANN-A1) | Omer Farooq Khan | [@omerrfarooqq](https://github.com/omerrfarooqq) |
| TA (ANN-A2) | Aun Ali | [@Aun-Dev146](https://github.com/Aun-Dev146) |
| TA (MLOps) | Ahsan Butt | [@ahsan608](https://github.com/ahsan608) |

## License

MIT — see [LICENSE](LICENSE).
