# Open an SSH local-forward tunnel from this machine's :LocalPort
# to the Grafilab GPU host's :RemotePort, where the FastAPI inference
# service is bound to 127.0.0.1.
#
# Usage:
#   .\scripts\tunnel.ps1 -User root -RemoteHost 118.107.222.200 -Port 39225
#   .\scripts\tunnel.ps1                  # uses $env:GRAFILAB_USER + $env:GRAFILAB_HOST [+ $env:GRAFILAB_PORT]

param(
  [string]$User       = $env:GRAFILAB_USER,
  [string]$RemoteHost = $env:GRAFILAB_HOST,
  [int]$Port          = $(if ($env:GRAFILAB_PORT) { [int]$env:GRAFILAB_PORT } else { 22 }),
  [int]$LocalPort     = 8000,
  [int]$RemotePort    = 8000,
  [string]$IdentityFile = ""
)

if (-not $User -or -not $RemoteHost) {
  Write-Host "Usage: .\scripts\tunnel.ps1 -User <name> -RemoteHost <host> [-Port <sshPort>] [-LocalPort 8000] [-RemotePort 8000] [-IdentityFile path]" -ForegroundColor Yellow
  Write-Host "Or set `$env:GRAFILAB_USER, `$env:GRAFILAB_HOST, and (if non-22) `$env:GRAFILAB_PORT."
  exit 1
}

$sshArgs = @(
  "-N",
  "-p", "$Port",
  "-L", "${LocalPort}:127.0.0.1:${RemotePort}",
  "-o", "ServerAliveInterval=30",
  "-o", "ServerAliveCountMax=3",
  "-o", "ExitOnForwardFailure=yes"
)
if ($IdentityFile) { $sshArgs += @("-i", $IdentityFile) }
$sshArgs += "$User@$RemoteHost"

Write-Host "Opening tunnel: localhost:$LocalPort  ->  $User@${RemoteHost}:$Port (remote 127.0.0.1:$RemotePort)" -ForegroundColor Cyan
Write-Host "Ctrl-C to close." -ForegroundColor DarkGray
& ssh @sshArgs
