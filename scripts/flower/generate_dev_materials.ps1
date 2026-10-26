# Development-only TLS + SuperNode authentication keys for Flower 1.30.
# Never commit outputs (see deployment/flower/dev/.gitignore).

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Out = Join-Path $Root "deployment\flower\dev"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location $Out

Write-Host "Generating CA and server certificate in $Out"

openssl req -x509 -newkey rsa:2048 -days 3650 -nodes `
  -keyout ca.key -out ca.pem `
  -subj "/CN=MedShield-FL-Dev-CA"

openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr `
  -subj "/CN=127.0.0.1"

@"
subjectAltName=DNS:localhost,IP:127.0.0.1
"@ | Set-Content -Encoding ascii san.cnf

openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial `
  -out server.pem -days 825 -extfile san.cnf

foreach ($id in 0, 1) {
  $name = "client$id"
  if (Test-Path "$name") { Remove-Item -Force "$name", "$name.pub" -ErrorAction SilentlyContinue }
  ssh-keygen -t ecdsa -b 384 -f $name -N '""' -q
}

Write-Host "Done. Files: ca.pem, server.pem, server.key, client0/client1 (+ .pub for registration)"
