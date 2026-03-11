#!/usr/bin/env bash
# Generate self-signed TLS certificates for local development.
# Usage: ./scripts/generate-dev-certs.sh
set -euo pipefail

CERT_DIR="$(cd "$(dirname "$0")/.." && pwd)/deploy/nginx/certs"
mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/server.crt" ] && [ -f "$CERT_DIR/server.key" ]; then
    echo "Certificates already exist at $CERT_DIR. Delete them first to regenerate."
    exit 0
fi

echo "Generating self-signed TLS certificate for local development..."
openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$CERT_DIR/server.key" \
    -out "$CERT_DIR/server.crt" \
    -subj "/C=US/ST=Dev/L=Local/O=AlphaForge/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

echo "Certificates generated at:"
echo "  Key:  $CERT_DIR/server.key"
echo "  Cert: $CERT_DIR/server.crt"
echo ""
echo "WARNING: These are self-signed certificates for LOCAL DEVELOPMENT ONLY."
echo "Do NOT use them in production."
