param(
    [string]$MirrorUrl = "https://hf-mirror.com",
    [string]$OutputRoot = "data/cache",
    [int]$EnglishFiles = 14,
    [int]$ChineseFiles = 80,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Get-ParquetList {
    param(
        [string]$Dataset,
        [string]$Prefix,
        [int]$MaxFiles
    )

    $apiUrl = "$($MirrorUrl.TrimEnd('/'))/api/datasets/$Dataset/tree/main/$($Prefix.Trim('/'))?recursive=true&expand=false"
    $rows = Invoke-RestMethod -Uri $apiUrl -TimeoutSec 60
    $files = @(
        $rows |
            Where-Object { $_.type -eq "file" -and $_.path.EndsWith(".parquet") } |
            Sort-Object path |
            Select-Object -First $MaxFiles
    )

    if ($files.Count -eq 0) {
        throw "No parquet files found for $Dataset/$Prefix"
    }

    return $files | ForEach-Object {
        [PSCustomObject]@{
            dataset = $Dataset
            path = $_.path
            size = $_.size
            url = "$($MirrorUrl.TrimEnd('/'))/datasets/$Dataset/resolve/main/$($_.path)"
        }
    }
}

function Save-Manifest {
    param(
        [array]$Files,
        [string]$Path
    )

    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force $parent | Out-Null
    $Files | ConvertTo-Json -Depth 5 | Set-Content -Path $Path -Encoding UTF8
}

function Download-File {
    param(
        [string]$Url,
        [string]$OutputPath,
        [Nullable[Int64]]$ExpectedSize
    )

    $parent = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Force $parent | Out-Null

    if ((Test-Path $OutputPath) -and $ExpectedSize) {
        $currentSize = (Get-Item $OutputPath).Length
        if ($currentSize -eq $ExpectedSize) {
            Write-Host "already complete: $OutputPath"
            return
        }
    }

    Write-Host "download: $Url"
    Write-Host "      -> $OutputPath"
    curl.exe -L -C - --retry 20 --retry-delay 5 --connect-timeout 30 --speed-limit 1024 --speed-time 120 -o $OutputPath $Url

    if ($ExpectedSize) {
        $actualSize = (Get-Item $OutputPath).Length
        if ($actualSize -ne $ExpectedSize) {
            throw "Size mismatch for $OutputPath. expected=$ExpectedSize actual=$actualSize"
        }
    }
}

$englishOut = Join-Path $OutputRoot "fineweb_edu_10bt"
$chineseOut = Join-Path $OutputRoot "chinesewebtext2_hq"

$english = @(Get-ParquetList -Dataset "HuggingFaceFW/fineweb-edu" -Prefix "sample/10BT" -MaxFiles $EnglishFiles)
$chinese = @(Get-ParquetList -Dataset "Morton-Li/ChineseWebText2.0-HighQuality" -Prefix "data" -MaxFiles $ChineseFiles)

Save-Manifest -Files $english -Path (Join-Path $englishOut "manifest.json")
Save-Manifest -Files $chinese -Path (Join-Path $chineseOut "manifest.json")

$englishBytes = ($english | Measure-Object -Property size -Sum).Sum
$chineseBytes = ($chinese | Measure-Object -Property size -Sum).Sum
$totalBytes = $englishBytes + $chineseBytes

Write-Host "English files: $($english.Count), bytes: $englishBytes"
Write-Host "Chinese files: $($chinese.Count), bytes: $chineseBytes"
Write-Host "Total bytes: $totalBytes"
Write-Host ("Estimated total GiB: {0:N2}" -f ($totalBytes / 1GB))

if ($DryRun) {
    Write-Host "Dry run only. No parquet files downloaded."
    exit 0
}

foreach ($file in $english) {
    $output = Join-Path $englishOut $file.path
    Download-File -Url $file.url -OutputPath $output -ExpectedSize $file.size
}

foreach ($file in $chinese) {
    $output = Join-Path $chineseOut $file.path
    Download-File -Url $file.url -OutputPath $output -ExpectedSize $file.size
}

Write-Host "Download complete."
