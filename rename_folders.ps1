# rename_folders.ps1
# Run this script AFTER stopping all uvicorn and Streamlit servers.
# It renames all 10 project folders to their professional names,
# then stages the rename in git.
#
# Usage (from the portfolio root in PowerShell):
#   .\rename_folders.ps1

$base = $PSScriptRoot

$renames = @(
    @{ From = 'Loan Approval Prediction';              To = 'Fair Mortgage Decisioning Platform' },
    @{ From = 'Credit Card Default Prediction';        To = 'Real-Time Fraud Detection' },
    @{ From = 'Employee Attrition Prediction';         To = 'People Analytics Platform' },
    @{ From = 'Parkinsons Disease Detection';          To = 'Parkinsons Biomarker Detection' },
    @{ From = 'Bankruptcy Prediction';                 To = 'Supply Chain Risk Intelligence' },
    @{ From = 'Customer Churn';                        To = 'CLV Retention Platform' },
    @{ From = 'AirBnb reviews Sentimental Analysis';   To = 'Brand Intelligence Platform' },
    @{ From = 'Object Detection';                      To = 'Retail Operations Intelligence' },
    @{ From = 'Pose Estimation';                       To = 'Workplace Ergonomics AI' },
    @{ From = 'Face Detection';                        To = 'PPE Safety Compliance' }
)

$successCount = 0
$failCount = 0

foreach ($r in $renames) {
    $src = Join-Path $base $r.From
    $dst = Join-Path $base $r.To

    if (Test-Path $dst) {
        Write-Host "ALREADY RENAMED: $($r.To)" -ForegroundColor Green
        $successCount++
        continue
    }

    if (-not (Test-Path $src)) {
        Write-Host "NOT FOUND: $($r.From)" -ForegroundColor Yellow
        continue
    }

    try {
        Rename-Item -Path $src -NewName $r.To -ErrorAction Stop
        Write-Host "OK: $($r.From)  ->  $($r.To)" -ForegroundColor Green
        $successCount++
    } catch {
        Write-Host "LOCKED: $($r.From) — make sure all servers are stopped" -ForegroundColor Red
        $failCount++
    }
}

Write-Host ""
Write-Host "$successCount renamed successfully, $failCount still locked." -ForegroundColor Cyan

if ($failCount -eq 0) {
    Write-Host ""
    Write-Host "All folders renamed. Staging rename in git..." -ForegroundColor Cyan
    Set-Location $base
    git add -A
    git commit -m "rename: project folders to professional names"
    Write-Host "Done. Push with: git push origin main" -ForegroundColor Green
}
