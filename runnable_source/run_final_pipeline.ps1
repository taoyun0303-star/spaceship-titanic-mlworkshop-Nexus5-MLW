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

$finalEnsemble = Find-RequiredScript "build_final_model_level_ensemble.py"

python $finalEnsemble
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
