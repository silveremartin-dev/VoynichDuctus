#!/usr/bin/env bash
# ==============================================================================
# VoynichDuctus — Google Cloud Platform Deployment Script
# Supports: Google Cloud Build, Cloud Run Jobs, and Compute Engine Spot VMs
# ==============================================================================
set -euo pipefail

# Configuration — set via environment or adjust defaults below
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo 'ether-509812')}"
REGION="${GCP_REGION:-europe-west1}"
IMAGE_NAME="voynich-ductus-batch"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"
BUCKET_NAME="${GCS_BUCKET:-${PROJECT_ID}-voynich-atlas}"

if [[ -z "${PROJECT_ID}" ]]; then
    PROJECT_ID="ether-509812"
fi

echo "=================================================================="
echo " VoynichDuctus GCP Batch Deployment"
echo " Project: ${PROJECT_ID} | Region: ${REGION}"
echo " Image:   ${IMAGE_TAG}"
echo " Bucket:  gs://${BUCKET_NAME}"
echo "=================================================================="

# 1. Build and push container via Cloud Build
echo -e "\n[*] Step 1: Submitting build to Google Cloud Build..."
gcloud builds submit --tag "${IMAGE_TAG}" -f scripts/gcp/Dockerfile .

# 2. Ensure GCS bucket exists
echo -e "\n[*] Step 2: Verifying Google Cloud Storage bucket..."
if ! gsutil ls -b "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
    echo "  [+] Creating bucket gs://${BUCKET_NAME} in ${REGION}..."
    gsutil mb -p "${PROJECT_ID}" -c STANDARD -l "${REGION}" "gs://${BUCKET_NAME}"
fi

# 3. Choose execution target
MODE="${1:-cloud-run}"
case "${MODE}" in
    "cloud-run")
        echo -e "\n[*] Step 3: Deploying Cloud Run Job (8 vCPUs, 16 GiB RAM)..."
        gcloud run jobs create voynich-batch-job \
            --image "${IMAGE_TAG}" \
            --region "${REGION}" \
            --cpu 8 \
            --memory 16Gi \
            --task-timeout 3600 \
            --max-retries 1 \
            --args="--workers=8,--gcs-bucket=${BUCKET_NAME}" \
            || gcloud run jobs update voynich-batch-job \
                --image "${IMAGE_TAG}" \
                --region "${REGION}" \
                --cpu 8 \
                --memory 16Gi \
                --args="--workers=8,--gcs-bucket=${BUCKET_NAME}"

        echo -e "\n[+] To execute job now, run:"
        echo "    gcloud run jobs execute voynich-batch-job --region ${REGION}"
        ;;

    "spot-vm")
        echo -e "\n[*] Step 3: Launching high-throughput Compute Engine Spot VM (c2d-standard-16)..."
        VM_NAME="voynich-batch-worker-$(date +%s)"
        gcloud compute instances create-with-container "${VM_NAME}" \
            --project="${PROJECT_ID}" \
            --zone="${REGION}-b" \
            --machine-type=c2d-standard-16 \
            --preemptible \
            --container-image="${IMAGE_TAG}" \
            --container-arg="--workers=16" \
            --container-arg="--gcs-bucket=${BUCKET_NAME}" \
            --scopes=cloud-platform \
            --tags=voynich-batch

        echo -e "\n[+] Spot VM launched: ${VM_NAME}. Monitor logs with:"
        echo "    gcloud compute instances get-serial-port-output ${VM_NAME} --zone=${REGION}-b"
        ;;

    *)
        echo "Unknown mode: ${MODE}. Available options: cloud-run, spot-vm"
        exit 1
        ;;
esac

echo -e "\n[+] Deployment configuration complete."
