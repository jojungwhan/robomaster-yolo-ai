[CmdletBinding()]
param(
    [string]$InstallerPath,
    [switch]$DownloadOnly,
    [switch]$Force,
    [switch]$KeepInstaller
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$DriveFileId = '1KaB71nUmsWfCn3udnFZWmD_qKw3eBObs'
$InstallerFileName = 'RoboMaster_x64_Installer_v1.1.5.exe'
$DownloadUrl = "https://drive.usercontent.google.com/download?id=$DriveFileId&export=download&confirm=t"
$ExpectedSha256 = 'A6B837257556BCC8BB128F0A5BDD642FAEE56B49B47CB9EAB719E44FC199F9A1'
$ExpectedSignerThumbprint = '5349D1B6BDB1C2B4F2BF9AD848AA110D374899A1'
$ExpectedVersion = [version]'1.1.5'

function Get-InstalledRoboMaster {
    $RegistryPaths = @(
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )

    Get-ItemProperty -Path $RegistryPaths -ErrorAction SilentlyContinue |
        Where-Object {
            $DisplayNameProperty = $_.PSObject.Properties['DisplayName']
            $DisplayNameProperty -and (
                $DisplayNameProperty.Value -eq 'RoboMaster' -or
                $DisplayNameProperty.Value -like 'RoboMaster *'
            )
        }
}

if ($env:OS -ne 'Windows_NT') {
    throw 'The RoboMaster PC application can only be installed on Windows.'
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'RoboMaster_x64_Installer_v1.1.5.exe requires 64-bit Windows.'
}

$InstalledRoboMaster = @(Get-InstalledRoboMaster)
$CurrentInstall = $InstalledRoboMaster | Where-Object {
    $DisplayVersionProperty = $_.PSObject.Properties['DisplayVersion']
    if (-not $DisplayVersionProperty) {
        return $false
    }
    $ParsedVersion = $null
    [version]::TryParse([string]$DisplayVersionProperty.Value, [ref]$ParsedVersion) -and
        $ParsedVersion -ge $ExpectedVersion
} | Select-Object -First 1

if ($CurrentInstall -and -not $Force -and -not $DownloadOnly) {
    Write-Host "RoboMaster $($CurrentInstall.DisplayVersion) is already installed."
    return
}

$TemporaryDirectory = $null
$DownloadedInstaller = $false

try {
    if ([string]::IsNullOrWhiteSpace($InstallerPath)) {
        $TemporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) (
            'robomaster-pc-setup-' + [guid]::NewGuid().ToString('N')
        )
        New-Item -ItemType Directory -Path $TemporaryDirectory | Out-Null
        $InstallerPath = Join-Path $TemporaryDirectory $InstallerFileName
        $DownloadedInstaller = $true

        Write-Host "Downloading RoboMaster PC $ExpectedVersion from the pinned Google Drive file..."
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $InstallerPath -UseBasicParsing
    }
    else {
        if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) {
            throw "Installer not found: $InstallerPath"
        }
        $InstallerPath = (Resolve-Path -LiteralPath $InstallerPath).Path
    }

    $ActualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $InstallerPath).Hash
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "RoboMaster installer SHA256 verification failed. Got $ActualSha256"
    }

    $Signature = Get-AuthenticodeSignature -LiteralPath $InstallerPath
    if ($Signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        throw "RoboMaster installer signature is not valid: $($Signature.StatusMessage)"
    }
    if (-not $Signature.SignerCertificate -or
        $Signature.SignerCertificate.Thumbprint -ne $ExpectedSignerThumbprint) {
        $ActualThumbprint = if ($Signature.SignerCertificate) {
            $Signature.SignerCertificate.Thumbprint
        }
        else {
            '<none>'
        }
        throw "RoboMaster installer signer verification failed. Got $ActualThumbprint"
    }

    Write-Host "Verified SHA256 and publisher signature: $InstallerPath"
    if ($DownloadOnly) {
        Write-Host 'Download-only mode selected; the installer was not executed.'
        return
    }

    $InstallLog = Join-Path ([IO.Path]::GetTempPath()) (
        'robomaster-pc-install-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log'
    )
    $InstallerArguments = @(
        '/SP-',
        '/VERYSILENT',
        '/SUPPRESSMSGBOXES',
        '/NORESTART',
        "/LOG=`"$InstallLog`""
    )

    Write-Host 'Installing RoboMaster PC silently. Windows may request administrator approval...'
    $InstallerProcess = Start-Process -FilePath $InstallerPath `
        -ArgumentList $InstallerArguments -Wait -PassThru
    if ($InstallerProcess.ExitCode -ne 0) {
        throw "RoboMaster PC installation failed with exit code $($InstallerProcess.ExitCode). Log: $InstallLog"
    }

    Write-Host "RoboMaster PC $ExpectedVersion installed successfully. Log: $InstallLog"
}
finally {
    $PreserveDownload = $KeepInstaller -or $DownloadOnly -or -not $DownloadedInstaller
    if (-not $PreserveDownload -and
        $TemporaryDirectory -and
        (Test-Path -LiteralPath $TemporaryDirectory)) {
        $ResolvedTemporaryDirectory = (Resolve-Path -LiteralPath $TemporaryDirectory).Path
        $ResolvedSystemTemp = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
        if (-not $ResolvedTemporaryDirectory.StartsWith(
            $ResolvedSystemTemp,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refusing to remove a temporary directory outside the system temp path: $ResolvedTemporaryDirectory"
        }
        Remove-Item -LiteralPath $ResolvedTemporaryDirectory -Recurse -Force
    }
}
