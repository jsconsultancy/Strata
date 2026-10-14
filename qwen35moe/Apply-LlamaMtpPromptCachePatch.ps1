[CmdletBinding()]
param(
    [string]$LlamaCppDir = 'C:\Models\Strata\runtimes\qwen36-35b-a3b\llama.cpp',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$ExpectedRevision = '79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6'
$PatchPath = Join-Path $PSScriptRoot 'patches\llama-mtp-prompt-cache-state.patch'
$Target = Join-Path $LlamaCppDir 'common\speculative.cpp'
if (-not (Test-Path -LiteralPath $PatchPath)) { throw "Patch not found: $PatchPath" }
if (-not (Test-Path -LiteralPath $Target)) { throw "llama.cpp speculative source not found: $Target" }
$Revision = (& git -C $LlamaCppDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to read the llama.cpp revision at $LlamaCppDir" }
if ($Revision -ne $ExpectedRevision) {
    throw "Refusing to apply the MTP prompt-cache patch to $Revision; expected $ExpectedRevision. Review and rebase the patch first."
}
if (Select-String -LiteralPath $Target -SimpleMatch 'kMtpPromptCacheStateMagic' -Quiet) {
    Write-Output "MTP prompt-cache state patch is already present in $LlamaCppDir."
    exit 0
}
& git -C $LlamaCppDir apply --check $PatchPath
if ($LASTEXITCODE -ne 0) { throw 'The MTP prompt-cache patch does not apply cleanly.' }
if ($CheckOnly) {
    Write-Output "MTP prompt-cache state patch applies cleanly to $LlamaCppDir."
    exit 0
}
& git -C $LlamaCppDir apply $PatchPath
if ($LASTEXITCODE -ne 0) { throw 'git apply failed for the MTP prompt-cache patch.' }
Write-Output "Applied MTP prompt-cache state patch to $LlamaCppDir. Rebuild the native adapter before starting Strata."
