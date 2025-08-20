#!/bin/bash
set -e

echo "🔧 Starting PyCodeRL development container..."

# Check if image exists
if ! docker image inspect pycode-rl:latest > /dev/null 2>&1; then
    echo "❌ Docker image not found. Building it first..."
    ./docker-build.sh
fi

# Run interactive development container
echo "🛠️  Starting development environment..."
echo "   (Your local directory is mounted to /workspace)"
echo ""

docker run --rm -it \
    --platform linux/amd64 \
    -v "$(pwd):/workspace" \
    -w /workspace \
    -p 8000:8000 \
    -p 8080:8080 \
    -p 8888:8888 \
    pycode-rl:latest \
    /bin/bash

echo "✅ Development container exited."
