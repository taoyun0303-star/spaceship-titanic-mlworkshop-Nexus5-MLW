$ErrorActionPreference = "Stop"

function Find-RequiredScript {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FileName
    )

    $matches = @(Get-ChildItem -LiteralPath . -Recurse -File -Filter $FileName)
    if ($matches.Count -eq 0) {
        throw "Cannot find required script: $FileName"
    }
    if ($matches.Count -gt 1) {
        $paths = ($matches | ForEach-Object { $_.FullName }) -join "`n"
        throw "Found multiple matches for ${FileName}:`n$paths"
    }
    return $matches[0].FullName
}

$v110 = Find-RequiredScript "pipeline_v110_clean_public_model_nohardcode.py"
$v124 = Find-RequiredScript "build_v124_v120_refinement_nohardcode.py"

python $v110 --fast
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python $v124
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
