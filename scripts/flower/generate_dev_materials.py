"""Generate dev TLS + SuperNode SSH keys (pure Python fallback when openssl CLI missing)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "deployment" / "flower" / "dev"


def _openssl_available() -> bool:
    try:
        subprocess.run(["openssl", "version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _generate_with_cryptography() -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import datetime

    OUT.mkdir(parents=True, exist_ok=True)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MedShield-FL-Dev-CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .sign(ca_key, hashes.SHA256())
    )
    (OUT / "ca.pem").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    (OUT / "ca.key").write_bytes(
        ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )

    srv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    srv_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(srv_name)
        .issuer_name(ca_name)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    (OUT / "server.pem").write_bytes(srv_cert.public_bytes(serialization.Encoding.PEM))
    (OUT / "server.key").write_bytes(
        srv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )


def _generate_ssh_keys() -> None:
    for idx in (0, 1):
        name = OUT / f"client{idx}"
        if name.exists():
            name.unlink()
        pub = OUT / f"client{idx}.pub"
        if pub.exists():
            pub.unlink()
        subprocess.run(
            ["ssh-keygen", "-t", "ecdsa", "-b", "384", "-f", str(name), "-N", ""],
            check=True,
        )


def main() -> int:
    if _openssl_available():
        ps1 = REPO / "scripts" / "flower" / "generate_dev_materials.ps1"
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
            check=True,
            cwd=str(REPO),
        )
    else:
        _generate_with_cryptography()
        _generate_ssh_keys()
    print(f"Dev Flower material ready under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
