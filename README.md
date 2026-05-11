# Pakistani Public Figures Image Classification

Production-grade, end-to-end image classification system that identifies **30
Pakistani public figures** (politicians + military spokespersons) from facial
images. Built as a university semester project (ANN + MLOps, Category A).

The differentiator is the **data pipeline** — multi-source scraping, MTCNN face
validation, perceptual-hash deduplication, blur/resolution filtering, and
FaceNet-based cross-class outlier detection. Models are trained on validated
data only.

## Architecture

```
+--------------+    +--------------+    +-----------+    +----------+
|  Scrapers    | -> |  Validation  | -> | Split &   | -> | Training |
|  (Google,    |    | (MTCNN,      |    | Augment   |    | (ResNet, |
|   Bing, DDG, |    |  pHash,      |    | (75/15/10)|    | EffNet,  |
|   Wiki,      |    |  blur, dim,  |    +-----------+    |   ViT)   |
|   News,      |    |  outlier)    |          |          +-----+----+
|   Selenium)  |    +--------------+          |                |
+------+-------+                              v                v
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
| Deep learning | PyTorch, torchvision |
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

## The 30 classes

| # | Class | Display name |
|---|---|---|
| 1 | imran_khan | Imran Khan |
| 2 | nawaz_sharif | Nawaz Sharif |
| 3 | shehbaz_sharif | Shehbaz Sharif |
| 4 | maryam_nawaz_sharif | Maryam Nawaz Sharif |
| 5 | hamza_shehbaz | Hamza Shehbaz |
| 6 | asif_ali_zardari | Asif Ali Zardari |
| 7 | bilawal_bhutto_zardari | Bilawal Bhutto Zardari |
| 8 | benazir_bhutto | Benazir Bhutto |
| 9 | yousaf_raza_gillani | Yousaf Raza Gillani |
| 10 | murad_ali_shah | Murad Ali Shah |
| 11 | shah_mehmood_qureshi | Shah Mehmood Qureshi |
| 12 | asad_umar | Asad Umar |
| 13 | sheikh_rashid_ahmed | Sheikh Rashid Ahmed |
| 14 | fawad_chaudhry | Fawad Chaudhry |
| 15 | pervez_khattak | Pervez Khattak |
| 16 | ali_amin_gandapur | Ali Amin Gandapur |
| 17 | fazl_ur_rehman | Fazl-ur-Rehman |
| 18 | sirajul_haq | Siraj-ul-Haq |
| 19 | mahmood_khan_achakzai | Mahmood Khan Achakzai |
| 20 | akhtar_mengal | Akhtar Mengal |
| 21 | pervez_musharraf | Pervez Musharraf |
| 22 | pervez_elahi | Pervez Elahi |
| 23 | chaudhry_shujaat_hussain | Chaudhry Shujaat Hussain |
| 24 | khawaja_asif | Khawaja Asif |
| 25 | ahsan_iqbal | Ahsan Iqbal |
| 26 | hina_rabbani_khar | Hina Rabbani Khar |
| 27 | sherry_rehman | Sherry Rehman |
| 28 | asim_munir | Asim Munir (current COAS) |
| 29 | qamar_javed_bajwa | Qamar Javed Bajwa (former COAS) |
| 30 | ahmed_sharif_chaudhry | Ahmed Sharif Chaudhry (current DG ISPR) |

## Quick start (local)

```bash
git clone https://github.com/<USER>/pak-public-figures-classifier.git
cd pak-public-figures-classifier

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

dvc pull                          # if remote configured
python scripts/run_data_collection.py
python scripts/run_validation.py
python scripts/run_split.py --force
python scripts/run_training.py
python scripts/run_evaluation.py --model resnet50
uvicorn src.api.main:app --reload
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
| GET | `/classes` | List all 30 classes |
| POST | `/predict` | Single-image top-3 prediction |
| POST | `/predict-batch` | Multi-image prediction |

### Example

```bash
curl -F "file=@portrait.jpg" \
     "http://localhost:8000/predict?model=resnet50"
```

```json
{
  "predictions": [
    {"class": "imran_khan", "display_name": "Imran Khan", "confidence": 0.94},
    {"class": "shah_mehmood_qureshi", "display_name": "Shah Mehmood Qureshi", "confidence": 0.04},
    {"class": "shehbaz_sharif", "display_name": "Shehbaz Sharif", "confidence": 0.02}
  ],
  "model_used": "resnet50",
  "face_detected": true,
  "face_bbox": [120, 80, 360, 320],
  "inference_time_ms": 87.2
}
```

## Model performance

See `results/model_comparison.csv` after training:

| Model | Val accuracy | Test accuracy | Macro F1 |
|---|---|---|---|
| ResNet-50 | (after training) | (after training) | (after training) |
| EfficientNet-B3 | (after training) | (after training) | (after training) |
| ViT-B/16 (bonus) | (after training) | (after training) | (after training) |

Training curves, confusion matrices, and Grad-CAM artifacts are auto-saved
under `results/` and uploaded to MLflow.

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
│   ├── models/             # ResNet, EfficientNet, ViT, trainer
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

## Member ownership

| Member | Classes | Owns |
|---|---|---|
| Abdullah | 1–10 (PML-N + PPP) | Scrapers, augmentation, DVC dataset |
| Raza     | 11–20 (PTI + religious parties) | Scrapers, ResNet-50 training, evaluation |
| Maarij   | 21–30 (others + military) | Scrapers, EfficientNet-B3, FastAPI, Docker, CI/CD, EC2, frontend |

## License

MIT — see [LICENSE](LICENSE).
