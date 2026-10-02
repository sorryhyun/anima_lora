param(
  [Parameter(Mandatory = $true)][string]$Installer,
  [Parameter(Mandatory = $true)][ValidateSet('cuda', 'rocm')][string]$Backend
)

# Execute the real post-download installer with package/GUI operations stubbed.
# A flagless uv run models its implicit sync to the default CUDA group (GH #105).
$ErrorActionPreference = 'Stop'
$script:Calls = @()
$script:InstalledBackend = 'uninstalled'
$CondaActive = $false
$Dir = (Get-Location).Path

function Say {}
function Write-Host {}
function Die($message) { throw $message }

function Invoke-TestUv([string]$Kind, [string[]]$CommandArgs) {
  # Native executables ignore the $null produced by an empty $SyncArgs branch.
  $CommandArgs = @($CommandArgs | Where-Object { $null -ne $_ })
  if ($CommandArgs[0] -notin @('sync', 'run')) {
    throw "unexpected uv command: $CommandArgs"
  }
  if ($CommandArgs -notcontains '--no-sync') {
    $script:InstalledBackend = if ($CommandArgs -contains 'rocm-windows') { 'rocm' } else { 'cuda' }
  }
  $script:Calls += [pscustomobject]@{
    kind = $Kind
    argv = @($CommandArgs)
    backend = $script:InstalledBackend
  }
  $global:LASTEXITCODE = 0
}

function uv { Invoke-TestUv 'uv' @($args) }

function Start-Process {
  param([string]$FilePath, [string[]]$ArgumentList, [string]$WorkingDirectory, [string]$WindowStyle)
  if ($FilePath -ne 'uv') { throw "unexpected launch: $FilePath" }
  Invoke-TestUv 'launch' $ArgumentList
}

$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
  $Installer, [ref]$tokens, [ref]$parseErrors
)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$syncAssignment = @($ast.EndBlock.Statements | Where-Object {
  $_ -is [System.Management.Automation.Language.AssignmentStatementAst] -and
  $_.Left.Extent.Text -eq '$SyncArgs'
})
if ($syncAssignment.Count -ne 1) { throw 'expected one backend sync argument assignment' }
$source = [System.IO.File]::ReadAllText($Installer)
& ([scriptblock]::Create($source.Substring($syncAssignment[0].Extent.StartOffset)))
ConvertTo-Json -InputObject @($script:Calls) -Depth 5 -Compress
