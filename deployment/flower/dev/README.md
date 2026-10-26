# Flower development TLS material (never commit)

Generate keys and certificates locally:

```powershell
Set-Location C:\Federated-Learning-Project
.\scripts\flower\generate_dev_materials.ps1
```

Outputs land in this directory (`ca.pem`, `server.pem`, `server.key`, `client0.key`, `client1.key`, public keys for registration).
