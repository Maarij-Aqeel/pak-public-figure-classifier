#!/usr/bin/env bash
# Post-scrape pipeline: validate -> split -> zip for Colab.
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "==============================================="
echo "  Step 1/3: Validation (face detect + dedup + quality)"
echo "==============================================="
rm -rf data/validated data/rejected
rm -f data/metadata/face_detection_log.csv \
      data/metadata/dedup_log.csv \
      data/metadata/quality_log.csv \
      data/metadata/cross_class_dedup.csv \
      data/metadata/outliers.csv \
      data/metadata/validation_report.json

python -m scripts.run_validation --skip-outliers

echo ""
echo "=== Validated per class ==="
for d in data/validated/*/; do
  [ -d "$d" ] && printf "%-30s %d\n" "$(basename "$d")" "$(ls "$d" 2>/dev/null | wc -l)"
done | sort -k2 -n

echo ""
echo "==============================================="
echo "  Step 2/3: Stratified 75/15/10 split"
echo "==============================================="
python -m scripts.run_split --force

echo ""
echo "==============================================="
echo "  Step 3/3: Zip splits for Colab upload"
echo "==============================================="
rm -f splits.zip
cd data && zip -rq ../splits.zip splits/ && cd ..
ls -lh splits.zip

echo ""
echo "==============================================="
echo "  DONE. Upload splits.zip to Colab."
echo "==============================================="
