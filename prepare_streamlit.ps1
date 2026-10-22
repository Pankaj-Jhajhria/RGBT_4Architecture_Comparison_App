$ErrorActionPreference = "Stop"
$appRoot = (Get-Location).Path
$parentRoot = Split-Path -Parent $appRoot

Write-Host "App root: $appRoot" -ForegroundColor Cyan

$requiredPaths = @(
    "app.py",
    "inference\model_loaders.py",
    "inference\preprocessing.py",
    "inference\visualization.py",
    "architectures\mfnet\model.py",
    "architectures\csfnet\models\csfnet.py",
    "architectures\cmaf\my_model_v2.py",
    "architectures\rtfnet18v2\model19_v2.py",
    "requirements.txt",
    ".gitattributes"
)
$missing = @($requiredPaths | Where-Object { -not (Test-Path (Join-Path $appRoot $_)) })
if ($missing.Count -gt 0) {
    Write-Host "Missing required app files:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw "Copy the latest app/model patches into the app root before deployment."
}

$weightsDir = Join-Path $appRoot "weights"
New-Item -ItemType Directory -Force -Path $weightsDir | Out-Null
$cmafTarget = Join-Path $weightsDir "cmaf_resnet18_cmaf_best_checkpoint.pt"
$cmafSource = Join-Path $parentRoot "CMAF_ResNet18_Upstream\checkpoints\resnet18_crossmodal_attention_v2_1_final\best_checkpoint.pt"
$copyCmaf = -not (Test-Path $cmafTarget)
if (-not $copyCmaf) {
    $targetSize = (Get-Item $cmafTarget).Length
    if ($targetSize -lt 300MB) { $copyCmaf = $true }
    else {
        $stream = [System.IO.File]::OpenRead($cmafTarget)
        try {
            $buffer = New-Object byte[] 80
            $readCount = $stream.Read($buffer, 0, $buffer.Length)
            $header = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $readCount)
            if ($header.StartsWith("version https://git-lfs.github.com/spec/v1")) { $copyCmaf = $true }
        } finally { $stream.Dispose() }
    }
}
if ($copyCmaf) {
    if (Test-Path $cmafSource) {
        Copy-Item $cmafSource $cmafTarget -Force
        Write-Host "Copied verified CMAF v2.1 checkpoint into app weights folder." -ForegroundColor Green
    } else {
        throw "New CMAF checkpoint missing or invalid. Expected sibling file: $cmafSource"
    }
}

$expected = @(
    @{ Name = "mfnet_best_checkpoint.pt"; MinMB = 5 },
    @{ Name = "csfnet2_best_model.pth"; MinMB = 50 },
    @{ Name = "cmaf_resnet18_cmaf_best_checkpoint.pt"; MinMB = 300 },
    @{ Name = "rtfnet18v2_best_fp16.pt"; MinMB = 25 }
)
foreach ($item in $expected) {
    $path = Join-Path $weightsDir $item.Name
    if (-not (Test-Path $path)) { throw "Missing active checkpoint: $path" }
    $file = Get-Item $path
    $sizeMB = [math]::Round($file.Length / 1MB, 1)
    if ($sizeMB -lt $item.MinMB) {
        throw "Checkpoint $($item.Name) is only $sizeMB MB. This may be a Git LFS pointer or incomplete file."
    }
    $stream = [System.IO.File]::OpenRead($path)
    try {
        $buffer = New-Object byte[] 80
        $readCount = $stream.Read($buffer, 0, $buffer.Length)
        $header = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $readCount)
        if ($header.StartsWith("version https://git-lfs.github.com/spec/v1")) {
            throw "Checkpoint $($item.Name) is still a Git LFS pointer. Ensure the actual binary exists locally before staging."
        }
    } finally { $stream.Dispose() }
    Write-Host ("  OK {0,-42} {1,7} MB" -f $item.Name, $sizeMB) -ForegroundColor Green
}

$gitLfs = Get-Command git -ErrorAction Stop
& git lfs version | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Git LFS is not installed. Install Git LFS, then rerun this script." }
& git lfs install | Out-Host
& git lfs track "weights/*.pt" "weights/*.pth" | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Could not configure Git LFS tracking." }

Write-Host "`nDeployment assets checked. No commit or push was performed." -ForegroundColor Cyan
Write-Host "Review the remote and current branch before staging files:" -ForegroundColor Yellow
& git remote -v
& git branch --show-current
& git status --short
