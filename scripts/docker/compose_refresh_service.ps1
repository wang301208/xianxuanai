$ErrorActionPreference = "Stop"

param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectDir,

  [string]$ComposeFile = "docker-compose.yml",

  [Parameter(Mandatory = $true)]
  [string]$Service
)

# Pull the latest image(s) and recreate a docker compose service (automation helper).
#
# Examples:
#   powershell -NoProfile -File .\compose_refresh_service.ps1 -ProjectDir .\deploy -ComposeFile docker-compose.yml -Service web

Set-Location $ProjectDir

docker compose -f $ComposeFile pull $Service
docker compose -f $ComposeFile up -d --no-deps --force-recreate --pull always $Service

