[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$PinnedCommit = 'ddd54ae077a8fed243065a1104ee14eb4aa5f5e2'
$ExpectedSha256 = '87591257D011CC7409CFF14BABF28A1D15402AB521E75F3D10BF5F7A1E013CF6'
$ModelDirectory = Join-Path $PSScriptRoot 'models\lego-identification'
$Destination = Join-Path $ModelDirectory 'FinalCoShSi.pt'
$DownloadUrl = "https://raw.githubusercontent.com/vsmidhun21/Lego-Identification/$PinnedCommit/FinalCoShSi.pt"

New-Item -ItemType Directory -Path $ModelDirectory -Force | Out-Null

if (Test-Path -LiteralPath $Destination) {
    $ExistingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash
    if ($ExistingHash -eq $ExpectedSha256) {
        Write-Host "Lego-Identification model is already installed and verified: $Destination"
        exit 0
    }
    throw "Existing model has an unexpected SHA256. Remove it manually before reinstalling: $Destination"
}

$TemporaryFile = Join-Path $ModelDirectory ("FinalCoShSi.pt.download." + [guid]::NewGuid().ToString('N'))
try {
    Write-Host "Downloading the pinned Lego-Identification checkpoint..."
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $TemporaryFile
    $DownloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $TemporaryFile).Hash
    if ($DownloadedHash -ne $ExpectedSha256) {
        throw "Downloaded checkpoint failed SHA256 verification. Got $DownloadedHash"
    }
    Move-Item -LiteralPath $TemporaryFile -Destination $Destination
    Write-Host "Installed verified model: $Destination"
}
finally {
    if (Test-Path -LiteralPath $TemporaryFile) {
        Remove-Item -LiteralPath $TemporaryFile -Force
    }
}
