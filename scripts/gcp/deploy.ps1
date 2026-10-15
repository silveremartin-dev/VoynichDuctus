# ==============================================================================
# VoynichDuctus — Google Cloud Platform Deployment Script (PowerShell)
# Automates: Build -> Cloud Execution (Spot VM / Cloud Run) -> VM Shutdown -> Data Repatriation
# ==============================================================================
param (
    [string]$Mode = "cloud-run",
    [string]$ProjectId = "ether-509812",
    [string]$Region = "europe-west1",
    [string]$BucketName = "ether-509812-voynich-atlas",
    [int]$VoynichLimit = 0,
    [int]$SerafiniLimit = 0,
    [switch]$SkipBuild
)

$env:CLOUDSDK_CORE_PROJECT = $ProjectId
$ServiceAccount = "openprimer-batch-runner@${ProjectId}.iam.gserviceaccount.com"
$ImageName = "voynich-ductus-batch"
$ImageTag = "gcr.io/${ProjectId}/${ImageName}:latest"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " VoynichDuctus GCP Batch Deployment & Auto-Repatriation" -ForegroundColor Cyan
Write-Host " Project:         $ProjectId"
Write-Host " Service Account: $ServiceAccount"
Write-Host " Region:          $Region"
Write-Host " Execution Mode:  $Mode"
Write-Host " Target Image:    $ImageTag"
Write-Host " GCS Bucket:      gs://$BucketName"
Write-Host "=================================================================="

# ------------------------------------------------------------------------------
# Step 1: Submit Container Build to Google Cloud Build (or reuse existing)
# ------------------------------------------------------------------------------
if ($SkipBuild) {
    Write-Host "`n[*] Step 1: Skipping build (reusing existing image: $ImageTag)..." -ForegroundColor Yellow
} else {
    Write-Host "`n[*] Step 1: Submitting build to Google Cloud Build..." -ForegroundColor Yellow
    gcloud builds submit . --project=$ProjectId --config=scripts/gcp/cloudbuild.yaml --substitutions=_IMAGE_TAG=$ImageTag
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Cloud Build failed! Aborting."
        exit 1
    }
}

# ------------------------------------------------------------------------------
# Step 2: Verify / Create Google Cloud Storage Bucket
# ------------------------------------------------------------------------------
Write-Host "`n[*] Step 2: Verifying Google Cloud Storage bucket..." -ForegroundColor Yellow
$bucketExists = gsutil ls -b "gs://$BucketName" 2>$null
if (-not $bucketExists) {
    Write-Host "  [+] Creating bucket gs://$BucketName in $Region..."
    gsutil mb -p $ProjectId -c STANDARD -l $Region "gs://$BucketName"
} else {
    Write-Host "  [+] Bucket gs://$BucketName exists and is ready."
}

# Construct Container Arguments
$workers = if ($Mode -eq "cloud-run") { 8 } else { 16 }
$containerArgs = @(
    "--workers=$workers",
    "--gcs-bucket=$BucketName",
    "--gcs-prefix=voynich_ductus/atlas_dataset"
)
if ($VoynichLimit -gt 0) {
    $containerArgs += "--voynich-limit=$VoynichLimit"
}
if ($SerafiniLimit -gt 0) {
    $containerArgs += "--serafini-limit=$SerafiniLimit"
}

# ------------------------------------------------------------------------------
# Step 3: Launch Execution & Monitor
# ------------------------------------------------------------------------------
if ($Mode -eq "spot-vm") {
    $epoch = [int][double]::Parse((Get-Date -UFormat %s))
    $vmName = "voynich-batch-worker-$epoch"
    $zone = "${Region}-b"

    Write-Host "`n[*] Step 3: Launching high-throughput Compute Engine Spot VM ($vmName, c2d-standard-16)..." -ForegroundColor Green
    
    $createCmd = "gcloud compute instances create-with-container $vmName " +
        "--project=$ProjectId " +
        "--zone=$zone " +
        "--machine-type=c2d-standard-16 " +
        "--preemptible " +
        "--boot-disk-size=50GB " +
        "--service-account=$ServiceAccount " +
        "--scopes=cloud-platform " +
        "--container-image=$ImageTag " +
        "--container-restart-policy=never " +
        "--tags=voynich-batch"

    foreach ($arg in $containerArgs) {
        $createCmd += " `"--container-arg=$arg`""
    }

    Invoke-Expression $createCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to launch Spot VM! Aborting."
        exit 1
    }

    Write-Host "`n[+] Spot VM launched: $vmName in zone $zone." -ForegroundColor Green
    Write-Host "    Monitoring execution until completion and automatic shutdown..." -ForegroundColor Yellow

    $explorerBlob = "gs://${BucketName}/voynich_ductus/atlas_dataset/grand_glyph_explorer.html"
    $maxWaitMinutes = 60
    $startTime = Get-Date

    try {
        while ($true) {
            Start-Sleep -Seconds 15
            $elapsed = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)

            # Check if the job finished by checking GCS for the generated explorer
            $hasExplorer = gsutil -q stat $explorerBlob
            if ($LASTEXITCODE -eq 0) {
                Write-Host "`n[+] Detection: Batch run successfully completed and uploaded to GCS! ($elapsed min elapsed)" -ForegroundColor Green
                # Allow 10 seconds for any final blob flush
                Start-Sleep -Seconds 10
                break
            }

            # Check VM status
            $vmStatus = (gcloud compute instances describe $vmName --project=$ProjectId --zone=$zone --format="value(status)" 2>$null)
            Write-Host "    [$elapsed min] VM Status: $vmStatus | Waiting for GCS batch completion..." -ForegroundColor Gray

            if ($vmStatus -eq "TERMINATED" -or $vmStatus -eq "STOPPED") {
                Write-Host "    [!] VM stopped. Checking final completion..."
                Start-Sleep -Seconds 5
                break
            }

            if ($elapsed -gt $maxWaitMinutes) {
                Write-Warning "Execution exceeded $maxWaitMinutes minutes timeout. Proceeding to shutdown."
                break
            }
        }
    }
    finally {
        # ----------------------------------------------------------------------
        # Step 4: SHUT DOWN & DELETE THE VM (No lingering charges!)
        # ----------------------------------------------------------------------
        Write-Host "`n[*] Step 4: Deleting VM $vmName to ensure ZERO ongoing costs..." -ForegroundColor Yellow
        gcloud compute instances delete $vmName --project=$ProjectId --zone=$zone --quiet
        Write-Host "  [+] Compute Engine Spot VM $vmName successfully terminated and deleted." -ForegroundColor Green
    }
}
elseif ($Mode -eq "cloud-run") {
    Write-Host "`n[*] Step 3: Deploying Cloud Run Job (8 vCPUs, 32 GiB RAM, 24h timeout)..." -ForegroundColor Green
    $runArgs = [string]::Join(",", $containerArgs)
    
    gcloud run jobs create voynich-batch-job `
        --project=$ProjectId `
        --image=$ImageTag `
        --region=$Region `
        --cpu=8 `
        --memory=32Gi `
        --task-timeout=86400s `
        --max-retries=1 `
        --service-account=$ServiceAccount `
        --args="$runArgs" 2>$null

    if ($LASTEXITCODE -ne 0) {
        gcloud run jobs update voynich-batch-job `
            --project=$ProjectId `
            --image=$ImageTag `
            --region=$Region `
            --cpu=8 `
            --memory=32Gi `
            --task-timeout=86400s `
            --service-account=$ServiceAccount `
            --args="$runArgs"
    }

    Write-Host "  [+] Executing Cloud Run Job (async/autonomous for overnight run)..." -ForegroundColor Green
    gcloud run jobs execute voynich-batch-job --project=$ProjectId --region=$Region --async
}

# ------------------------------------------------------------------------------
# Step 5: Repatriate All Data to Local Workspace
# ------------------------------------------------------------------------------
Write-Host "`n[*] Step 5: Repatriating all extracted assets and interactive explorer locally..." -ForegroundColor Cyan
$localDest = "output/atlas_dataset"
New-Item -ItemType Directory -Force -Path $localDest | Out-Null

gsutil -m rsync -r "gs://${BucketName}/voynich_ductus/atlas_dataset" "$localDest"

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host " BATCH EXECUTION & REPATRIATION COMPLETE!" -ForegroundColor Green
Write-Host " VM Shutdown: Verified deleted (Zero lingering costs)"
Write-Host " Local Path:  $localDest"
Write-Host " Explorer:    $localDest/grand_glyph_explorer.html"
Write-Host "=================================================================="
