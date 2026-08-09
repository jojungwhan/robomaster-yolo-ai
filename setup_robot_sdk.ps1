[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PythonVersion = '3.8.10'
$PythonInstallerSha256 = '7628244CB53408B50639D2C1287C659F4E29D3DFDB9084B11AED5870C0C6A48A'
$PythonSignerThumbprint = 'C91DCECB3A92A17B063059200B20F5CE251B5A95'
$SdkWheelSha256 = '90A0EF0E5A95198FCDE0A37D5346E683A76814E0F61B0FAFFA9CD1FD7F109942'

$VendorDirectory = Join-Path $PSScriptRoot 'vendor\robomaster-sdk\windows'
$PythonInstaller = Join-Path $VendorDirectory 'python-3.8.10-amd64.exe'
$SdkWheel = Join-Path $VendorDirectory 'robomaster-0.1.1.68-cp38-cp38-win_amd64.whl'
$VirtualEnvironment = Join-Path $PSScriptRoot '.venv-robot'
$VirtualEnvironmentPython = Join-Path $VirtualEnvironment 'Scripts\python.exe'

function Assert-FileHash {
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string]$ExpectedSha256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required offline setup file is missing: $Path"
    }
    $ActualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "SHA256 verification failed for $Path. Got $ActualSha256"
    }
}

function Test-CompatiblePython {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    & $Path -c "import struct, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 8) and struct.calcsize('P') == 8 else 1)"
    return $LASTEXITCODE -eq 0
}

function Find-CompatiblePython {
    $PythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($PythonLauncher) {
        $LauncherCandidate = & $PythonLauncher.Source -3.8 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $LauncherCandidate) {
            $LauncherCandidate = [string]($LauncherCandidate | Select-Object -Last 1)
            if (Test-CompatiblePython -Path $LauncherCandidate) {
                return $LauncherCandidate
            }
        }
    }

    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python38\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python38-64\python.exe'),
        (Join-Path $env:ProgramFiles 'Python38\python.exe')
    )
    foreach ($Candidate in $Candidates) {
        if (Test-CompatiblePython -Path $Candidate) {
            return $Candidate
        }
    }
    return $null
}

if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'The bundled RoboMaster SDK requires 64-bit Windows.'
}

Assert-FileHash -Path $SdkWheel -ExpectedSha256 $SdkWheelSha256

if (Test-Path -LiteralPath $VirtualEnvironmentPython -PathType Leaf) {
    if (-not (Test-CompatiblePython -Path $VirtualEnvironmentPython)) {
        throw "Existing $VirtualEnvironment was not created with 64-bit Python 3.8. Remove or rename it, then rerun setup."
    }
}
else {
    $BasePython = Find-CompatiblePython
    if (-not $BasePython) {
        Assert-FileHash -Path $PythonInstaller -ExpectedSha256 $PythonInstallerSha256
        $Signature = Get-AuthenticodeSignature -LiteralPath $PythonInstaller
        if ($Signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
            -not $Signature.SignerCertificate -or
            $Signature.SignerCertificate.Thumbprint -ne $PythonSignerThumbprint) {
            throw 'The bundled Python 3.8.10 installer publisher signature is not valid.'
        }

        $PythonInstallDirectory = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python38'
        $PythonArguments = @(
            '/quiet',
            'InstallAllUsers=0',
            "TargetDir=`"$PythonInstallDirectory`"",
            'PrependPath=0',
            'Include_launcher=0',
            'Include_pip=1',
            'Include_test=0',
            'Include_doc=0',
            'Include_tcltk=0',
            'Shortcuts=0'
        )
        Write-Host "Installing the bundled 64-bit Python $PythonVersion runtime for the RoboMaster SDK..."
        $PythonInstallProcess = Start-Process -FilePath $PythonInstaller `
            -ArgumentList $PythonArguments -Wait -PassThru
        if ($PythonInstallProcess.ExitCode -ne 0) {
            throw "Python $PythonVersion installation failed with exit code $($PythonInstallProcess.ExitCode)."
        }
        $BasePython = Join-Path $PythonInstallDirectory 'python.exe'
        if (-not (Test-CompatiblePython -Path $BasePython)) {
            throw "Python $PythonVersion installed but could not be validated at $BasePython"
        }
    }

    Write-Host 'Creating the Python 3.8 RoboMaster virtual environment...'
    & $BasePython -m venv $VirtualEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed to create $VirtualEnvironment (exit code $LASTEXITCODE)."
    }
}

& $VirtualEnvironmentPython -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Bootstrapping pip in the RoboMaster virtual environment...'
    & $VirtualEnvironmentPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        throw "ensurepip failed with exit code $LASTEXITCODE."
    }
}

Write-Host 'Updating Python 3.8 packaging tools...'
& $VirtualEnvironmentPython -m pip install --upgrade 'pip<25.1' 'setuptools<76' wheel
if ($LASTEXITCODE -ne 0) {
    throw "pip bootstrap failed with exit code $LASTEXITCODE."
}

Write-Host 'Installing application and RoboMaster robot-control dependencies...'
Push-Location $PSScriptRoot
try {
    & $VirtualEnvironmentPython -m pip install -r requirements-robot.txt
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed to install requirements-robot.txt (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}

& $VirtualEnvironmentPython -c "import struct, sys; from robomaster import robot, version; assert struct.calcsize('P') == 8; print('RoboMaster SDK', version.__version__, 'ready on', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) {
    throw 'RoboMaster SDK import verification failed.'
}

Write-Host 'Robot-control environment setup complete.'
