[CmdletBinding()]
param(
    [switch]$SkipRoboMasterPc,
    [switch]$ForceRoboMasterPc,
    [switch]$LocalOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $SkipRoboMasterPc -and -not $LocalOnly) {
    & (Join-Path $PSScriptRoot 'setup_robomaster_pc.ps1') -Force:$ForceRoboMasterPc
}

if (-not $LocalOnly) {
    & (Join-Path $PSScriptRoot 'setup_robot_sdk.ps1')
    Write-Host 'Setup complete.'
    Write-Host 'Start the robot-control environment with:'
    Write-Host '  .\.venv-robot\Scripts\Activate.ps1'
    Write-Host '  python robomaster_yolo_ai.py'
    return
}

$LocalVirtualEnvironment = Join-Path $PSScriptRoot '.venv'
$LocalVirtualEnvironmentPython = Join-Path $LocalVirtualEnvironment 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $LocalVirtualEnvironmentPython -PathType Leaf)) {
    $SystemPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $SystemPython) {
        throw 'Python was not found on PATH. Install Python, then run this setup script again.'
    }

    Write-Host 'Creating the project virtual environment...'
    & $SystemPython.Source -m venv $LocalVirtualEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed to create the virtual environment (exit code $LASTEXITCODE)."
    }
}

Write-Host 'Installing the project Python dependencies...'
& $LocalVirtualEnvironmentPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    throw "pip failed to install the project dependencies (exit code $LASTEXITCODE)."
}

Write-Host 'Setup complete.'
Write-Host 'Start the project with:'
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  python robomaster_yolo_ai.py'
