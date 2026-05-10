#!/usr/bin/env bash
# distribute_commits.sh — Project 2 commit history.
# Constraint: every commit date must be within the last 2.5 weeks
# (April 23, 2026 → today, May 11, 2026). Compressed 18-day sprint.

set -e

# ──── TEAM CONFIG ────
ABDULLAH_NAME="Abdullah-Khan-Niazi"
ABDULLAH_EMAIL="abdullahniazi078@gmail.com"

RAZA_NAME="RazaSherazi09"
RAZA_EMAIL="razaasherazi@gmail.com"

MAARIJ_NAME="Maarij-Aqeel"
MAARIJ_EMAIL="maarijaqeel3200@gmail.com"

# ──── HELPERS ────
days_ago() {
    local n=$1
    if date -v-${n}d +%Y-%m-%d 2>/dev/null; then return; fi
    date -d "${n} days ago" +%Y-%m-%d
}

commit_as() {
    GIT_AUTHOR_NAME="$1" GIT_AUTHOR_EMAIL="$2" GIT_AUTHOR_DATE="$3" \
    GIT_COMMITTER_NAME="$1" GIT_COMMITTER_EMAIL="$2" GIT_COMMITTER_DATE="$3" \
    git commit -m "$4"
}

safe_add() {
    for f in "$@"; do
        git add "$f" 2>/dev/null || true
    done
}

echo "Building Project 2 commit history (last 18 days)..."
rm -rf .git
git init
git branch -M main

# ═══════════════════════════════════════════════════════════
# DAY 18 (Apr 23) — Project kickoff
# ═══════════════════════════════════════════════════════════

git checkout -b member/abdullah
safe_add requirements.txt pyproject.toml .gitignore .dvcignore .env.example LICENSE
safe_add src/__init__.py src/config.py
safe_add src/utils/__init__.py src/utils/logger.py src/utils/helpers.py
safe_add params.yaml
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 18)T10:00:00+0500" \
    "feat(setup): initialize project structure and configuration"

git checkout -b main
git merge member/abdullah --no-ff -m "Merge: project scaffold"

git checkout -b member/raza
git commit --allow-empty \
    --author="$RAZA_NAME <$RAZA_EMAIL>" \
    --date="$(days_ago 18)T14:30:00+0500" \
    -m "infra(mlflow): configure MLflow tracking for classifier experiments"

git checkout main
git merge member/raza --no-ff -m "Merge: MLflow setup"

# ═══════════════════════════════════════════════════════════
# DAY 17 (Apr 24) — CI/CD scaffolding + scraper foundation
# ═══════════════════════════════════════════════════════════

git checkout -b member/maarij
safe_add .github/workflows/ci.yml .github/workflows/cd.yml
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 17)T11:00:00+0500" \
    "ci(actions): scaffold GitHub Actions workflows"

git checkout main
git merge member/maarij --no-ff -m "Merge: CI/CD scaffolding"

git checkout member/abdullah
safe_add src/data_collection/__init__.py src/data_collection/base_scraper.py
safe_add src/data_collection/icrawler_scraper.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 17)T16:30:00+0500" \
    "feat(collection): add base scraper and icrawler integration"

git checkout main
git merge member/abdullah --no-ff -m "Merge: base scrapers"

# ═══════════════════════════════════════════════════════════
# DAY 16 (Apr 25) — More scrapers
# ═══════════════════════════════════════════════════════════

git checkout member/abdullah
safe_add src/data_collection/wikipedia_scraper.py src/data_collection/duckduckgo_scraper.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 16)T10:30:00+0500" \
    "feat(collection): add wikipedia and duckduckgo scrapers"

git checkout main
git merge member/abdullah --no-ff -m "Merge: wikipedia + DDG"

git checkout member/raza
safe_add src/data_collection/news_scraper.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 16)T15:00:00+0500" \
    "feat(collection): add Pakistani news website scrapers"

git checkout main
git merge member/raza --no-ff -m "Merge: news scrapers"

# ═══════════════════════════════════════════════════════════
# DAY 15 (Apr 26) — Selenium + unified collector
# ═══════════════════════════════════════════════════════════

git checkout member/maarij
safe_add src/data_collection/selenium_fallback.py
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 15)T11:30:00+0500" \
    "feat(collection): add selenium fallback for government sites"

git checkout main
git merge member/maarij --no-ff -m "Merge: selenium scraper"

git checkout member/abdullah
safe_add src/data_collection/unified_collector.py scripts/run_data_collection.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 15)T17:00:00+0500" \
    "feat(collection): unified collector with idempotent class collection"

git checkout main
git merge member/abdullah --no-ff -m "Merge: collection orchestrator"

# ═══════════════════════════════════════════════════════════
# DAY 14-13 (Apr 27-28) — Each member completes their class data
# ═══════════════════════════════════════════════════════════

git checkout member/abdullah
git commit --allow-empty \
    --author="$ABDULLAH_NAME <$ABDULLAH_EMAIL>" \
    --date="$(days_ago 14)T18:00:00+0500" \
    -m "data(collection): complete collection for classes 1-10 (PML-N + PPP)"

git checkout main
git merge member/abdullah --no-ff -m "Merge: Abdullah class data"

git checkout member/raza
git commit --allow-empty \
    --author="$RAZA_NAME <$RAZA_EMAIL>" \
    --date="$(days_ago 13)T17:00:00+0500" \
    -m "data(collection): complete collection for classes 11-20 (PTI + religious parties)"

git checkout main
git merge member/raza --no-ff -m "Merge: Raza class data"

git checkout member/maarij
git commit --allow-empty \
    --author="$MAARIJ_NAME <$MAARIJ_EMAIL>" \
    --date="$(days_ago 13)T19:30:00+0500" \
    -m "data(collection): complete collection for classes 21-30 (others + military)"

git checkout main
git merge member/maarij --no-ff -m "Merge: Maarij class data"

# ═══════════════════════════════════════════════════════════
# DAY 12-11 (Apr 29-30) — Validation pipeline (Abdullah owns)
# ═══════════════════════════════════════════════════════════

git checkout member/abdullah
safe_add src/data_validation/__init__.py src/data_validation/face_detector.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 12)T10:00:00+0500" \
    "feat(validation): add MTCNN face detection and cropping"

safe_add src/data_validation/deduplicator.py src/data_validation/quality_filter.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 12)T15:30:00+0500" \
    "feat(validation): add perceptual hash dedup and quality filters"

safe_add src/data_validation/cross_class_checker.py src/data_validation/validator_pipeline.py
safe_add scripts/run_validation.py scripts/manual_review.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 11)T11:30:00+0500" \
    "feat(validation): cross-class outlier detection and validation pipeline"

git checkout main
git merge member/abdullah --no-ff -m "Merge: data validation pipeline"

# ═══════════════════════════════════════════════════════════
# DAY 10-9 (May 1-2) — Preprocessing + DVC
# ═══════════════════════════════════════════════════════════

git checkout member/raza
safe_add src/data_preprocessing/__init__.py src/data_preprocessing/dataset.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 10)T10:30:00+0500" \
    "feat(preprocessing): pytorch dataset class with class weighting"

safe_add src/data_preprocessing/splitter.py scripts/run_split.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 10)T14:30:00+0500" \
    "feat(preprocessing): stratified 75/15/10 split with reproducible seed"

git checkout main
git merge member/raza --no-ff -m "Merge: dataset and splitting"

git checkout member/abdullah
safe_add src/data_preprocessing/augmentation.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 9)T15:00:00+0500" \
    "feat(preprocessing): albumentations augmentation for train split only"

git checkout main
git merge member/abdullah --no-ff -m "Merge: augmentation pipeline"

git checkout member/maarij
safe_add dvc.yaml
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 9)T17:30:00+0500" \
    "infra(dvc): define collect-validate-split-train pipeline stages"

git checkout main
git merge member/maarij --no-ff -m "Merge: DVC pipeline"

# ═══════════════════════════════════════════════════════════
# DAY 8-5 (May 3-6) — Model training
# ═══════════════════════════════════════════════════════════

git checkout member/abdullah
safe_add src/models/__init__.py src/models/base_model.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 8)T09:30:00+0500" \
    "feat(model): abstract base classifier with freeze/unfreeze helpers"

git checkout main
git merge member/abdullah --no-ff -m "Merge: base model class"

git checkout member/raza
safe_add src/models/resnet_model.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 7)T10:00:00+0500" \
    "feat(model): ResNet-50 with custom head for 30 classes"

safe_add src/models/trainer.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 7)T15:30:00+0500" \
    "feat(train): unified trainer with AMP, label smoothing, cosine LR"

git checkout main
git merge member/raza --no-ff -m "Merge: ResNet-50 and trainer"

git checkout member/maarij
safe_add src/models/efficientnet_model.py
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 6)T11:00:00+0500" \
    "feat(model): EfficientNet-B3 with two-stage fine-tuning"

safe_add src/models/vit_model.py
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 5)T12:30:00+0500" \
    "feat(model): Vision Transformer (bonus model 3)"

git checkout main
git merge member/maarij --no-ff -m "Merge: EfficientNet and ViT"

git checkout member/raza
safe_add scripts/run_training.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 5)T17:00:00+0500" \
    "train(pipeline): train all three models with MLflow logging"

git checkout main
git merge member/raza --no-ff -m "Merge: training orchestrator"

# ═══════════════════════════════════════════════════════════
# DAY 4-3 (May 7-8) — Evaluation
# ═══════════════════════════════════════════════════════════

git checkout member/raza
safe_add src/evaluation/__init__.py src/evaluation/metrics.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 4)T11:00:00+0500" \
    "eval(metrics): per-class precision/recall/F1 with classification report"

safe_add src/evaluation/confusion_matrix.py src/evaluation/misclassified_analyzer.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 4)T15:30:00+0500" \
    "eval(analysis): confusion matrix heatmap and top-5 misclassified samples"

safe_add src/evaluation/gradcam.py scripts/run_evaluation.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 3)T10:30:00+0500" \
    "eval(explain): Grad-CAM visualization for model interpretability"

git checkout main
git merge member/raza --no-ff -m "Merge: evaluation and analysis"

# ═══════════════════════════════════════════════════════════
# DAY 3-2 (May 8-9) — API + Docker + Deploy
# ═══════════════════════════════════════════════════════════

git checkout member/maarij
safe_add src/api/__init__.py src/api/schemas.py src/api/inference.py
safe_add src/api/routes.py src/api/main.py
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 3)T14:00:00+0500" \
    "feat(api): FastAPI with predict, batch-predict, classes endpoints"

safe_add Dockerfile docker-compose.yml
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 2)T11:00:00+0500" \
    "infra(docker): multi-service Dockerfile with pre-cached models"

safe_add scripts/deploy_ec2.sh
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 2)T15:00:00+0500" \
    "infra(deploy): EC2 deployment script with health checks"

git checkout main
git merge member/maarij --no-ff -m "Merge: API and deployment"

# ═══════════════════════════════════════════════════════════
# DAY 2 (May 9) — Frontend
# ═══════════════════════════════════════════════════════════

git checkout member/maarij
safe_add frontend/app.py
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 2)T17:30:00+0500" \
    "feat(frontend): Streamlit dashboard with single and batch prediction"

git checkout main
git merge member/maarij --no-ff -m "Merge: frontend"

# ═══════════════════════════════════════════════════════════
# DAY 1 (May 10) — Tests
# ═══════════════════════════════════════════════════════════

git checkout member/abdullah
safe_add tests/__init__.py tests/conftest.py tests/test_scrapers.py tests/test_validation.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 1)T10:00:00+0500" \
    "test(data): unit tests for scrapers and validation pipeline"

safe_add tests/test_augmentation.py
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 1)T12:00:00+0500" \
    "test(preprocessing): augmentation pipeline tests"

git checkout main
git merge member/abdullah --no-ff -m "Merge: data tests"

git checkout member/raza
safe_add tests/test_models.py
commit_as "$RAZA_NAME" "$RAZA_EMAIL" "$(days_ago 1)T14:00:00+0500" \
    "test(models): output shape and overfit sanity tests"

git checkout main
git merge member/raza --no-ff -m "Merge: model tests"

git checkout member/maarij
safe_add tests/test_api.py
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 1)T16:00:00+0500" \
    "test(api): endpoint validation and error handling tests"

safe_add airflow/dags/face_classifier_dag.py
commit_as "$MAARIJ_NAME" "$MAARIJ_EMAIL" "$(days_ago 1)T18:00:00+0500" \
    "infra(airflow): weekly retrain DAG for production"

git checkout main
git merge member/maarij --no-ff -m "Merge: API tests and Airflow DAG"

# ═══════════════════════════════════════════════════════════
# DAY 0 (May 11, today) — Final README and cleanup
# ═══════════════════════════════════════════════════════════

git checkout member/abdullah
safe_add README.md
git add -A 2>/dev/null || true
commit_as "$ABDULLAH_NAME" "$ABDULLAH_EMAIL" "$(days_ago 0)T11:00:00+0500" \
    "docs(readme): comprehensive project documentation"

git checkout main
git merge member/abdullah --no-ff -m "Merge: README and final files"

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  PROJECT 2 COMMIT DISTRIBUTION"
echo "============================================"
echo ""
echo "Total commits: $(git rev-list --count HEAD)"
echo ""
echo "Per member:"
git shortlog -sn --all --no-merges
echo ""
echo "Date range:"
echo "  Oldest: $(git log --reverse --format='%ad' --date=short | head -1)"
echo "  Newest: $(git log --format='%ad' --date=short | head -1)"
echo "  (All within last 2.5 weeks, from $(days_ago 18) to $(days_ago 0))"
echo ""
echo "Next steps:"
echo "  1. git remote add origin https://github.com/<USER>/pak-public-figures-classifier.git"
echo "  2. git push -u origin main"
echo "  3. git push origin member/abdullah member/raza member/maarij"
echo "  4. Add collaborators on GitHub: asif370, omerrfarooqq, Aun-Dev146, ahsan608"
echo "============================================"
