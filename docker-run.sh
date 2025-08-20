#!/bin/bash
set -e

echo "🐳 Running PyCodeRL Docker container..."

# Check if image exists
if ! docker image inspect pycode-rl:latest > /dev/null 2>&1; then
    echo "❌ Docker image not found. Building it first..."
    ./docker-build.sh
fi

# Run the container with x86-64 emulation
echo "🏃 Starting container..."
docker run --rm -it --platform linux/amd64 pycode-rl:latest

echo "✅ Container finished."
