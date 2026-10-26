# Builds dist\downloader\downloader.exe and a zip for a GitHub release.
# Build output goes outside the repo; set DOWNLOADER_BUILD to choose where (default: .\build).
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$out = if ($env:DOWNLOADER_BUILD) { $env:DOWNLOADER_BUILD } else { Join-Path $repo "build" }
$version = (Select-String -Path (Join-Path $repo "pyproject.toml") -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

Push-Location $repo
try {
    # Absolute paths: PyInstaller resolves relative ones from --specpath.
    $assets = Join-Path $repo "downloader\ui\assets"
    & $python -m PyInstaller --noconfirm --clean --windowed --name downloader `
        --icon (Join-Path $assets "app.ico") `
        --add-data "$assets;downloader\ui\assets" `
        --collect-binaries imageio_ffmpeg `
        --collect-submodules yt_dlp `
        --distpath (Join-Path $out "dist") `
        --workpath (Join-Path $out "work") `
        --specpath (Join-Path $out "work") `
        (Join-Path $repo "scripts\launcher.py")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

    $zip = Join-Path $out "dist\downloader-$version-windows.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path (Join-Path $out "dist\downloader") -DestinationPath $zip
    Write-Host "Built $zip"
}
finally {
    Pop-Location
}
