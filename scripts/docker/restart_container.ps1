$ErrorActionPreference = "Stop"

param(
  [Parameter(Mandatory = $true)]
  [string]$Container,

  [int]$TimeoutSeconds = 10
)

# Restart a Docker container (simple wrapper for automation).
#
# Examples:
#   powershell -NoProfile -File .\restart_container.ps1 -Container my_container
#   powershell -NoProfile -File .\restart_container.ps1 -Container my_container -TimeoutSeconds 30

docker restart -t $TimeoutSeconds $Container

