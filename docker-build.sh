#!/bin/bash
set -e

echo "🐳 Building PyCodeRL Docker image..."

# Build the Docker image with x86-64 platform
docker build --platform linux/amd64 -t pycode-rl:latest .

echo "✅ Docker image built successfully!"
echo ""
echo "🚀 Next steps:"
echo "   Run: ./docker-run.sh"
echo "   Or:  docker run --rm -it pycode-rl:latest"
echo ""
