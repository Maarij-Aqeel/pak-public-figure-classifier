#!/usr/bin/env bash
# EC2 deployment helper.
set -euo pipefail

EC2_HOST="${EC2_HOST:?EC2_HOST required}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_KEY="${EC2_KEY:-$HOME/.ssh/pak-faces.pem}"
REMOTE_DIR="/opt/pak-faces"

echo "Deploying to ${EC2_USER}@${EC2_HOST}..."

ssh -i "${EC2_KEY}" -o StrictHostKeyChecking=no \
    "${EC2_USER}@${EC2_HOST}" \
    "sudo mkdir -p ${REMOTE_DIR} && sudo chown ${EC2_USER}:${EC2_USER} ${REMOTE_DIR}"

rsync -avz --delete \
    --exclude '.git/' \
    --exclude 'data/' \
    --exclude 'models/saved/*' \
    --exclude '__pycache__/' \
    --exclude '*.log' \
    --exclude 'venv/' \
    -e "ssh -i ${EC2_KEY} -o StrictHostKeyChecking=no" \
    ./ "${EC2_USER}@${EC2_HOST}:${REMOTE_DIR}/"

ssh -i "${EC2_KEY}" -o StrictHostKeyChecking=no \
    "${EC2_USER}@${EC2_HOST}" \
    "cd ${REMOTE_DIR} && docker-compose pull && docker-compose up -d --build --remove-orphans"

echo "Waiting for API health..."
for i in $(seq 1 30); do
    if curl -fs "http://${EC2_HOST}:8000/health" > /dev/null; then
        echo "API is healthy."
        exit 0
    fi
    sleep 5
done
echo "API did not become healthy within 150s."
exit 1
