# PyCodeRL Docker Usage

This directory contains Docker configuration and scripts for running PyCodeRL on x86-64 architecture using Docker's platform emulation.

## 🚀 Quick Start

### Option 1: Run Tests (Easiest)
```bash
chmod +x docker-*.sh
./docker-run.sh
```

### Option 2: Development Environment
```bash
./docker-dev.sh
# This gives you a bash shell inside the container
```

### Option 3: Manual Docker Commands
```bash
# Build
docker build --platform linux/amd64 -t pycode-rl:latest .

# Run tests
docker run --rm -it --platform linux/amd64 pycode-rl:latest

# Development shell
docker run --rm -it --platform linux/amd64 -v $(pwd):/workspace -w /workspace pycode-rl:latest /bin/bash
```

## 📁 Docker Files

- `Dockerfile` - Main container configuration
- `docker-build.sh` - Build the Docker image
- `docker-run.sh` - Run PyCodeRL tests in container
- `docker-dev.sh` - Start development environment
- `docker-README.md` - This file

## 🏗️ What's Included

The Docker container provides:
- **Ubuntu 20.04** base with Python 3.11
- **x86-64 architecture** (emulated on ARM64 Mac)
- **GCC toolchain** for assembly compilation
- **All PyCodeRL dependencies** pre-installed
- **Development tools** (gdb, valgrind, vim)

## ⚡ Performance Notes

**On ARM64 Mac:**
- Docker will emulate x86-64 architecture
- Performance will be slower than native
- But assembly compilation/execution will work correctly

**Recommended for:**
- ✅ Testing PyCodeRL functionality
- ✅ Verifying x86-64 compatibility
- ✅ Development and debugging
- ❌ High-performance training (use Codespaces instead)

## 🛠️ Development Workflow

1. **Start development container:**
   ```bash
   ./docker-dev.sh
   ```

2. **Inside the container:**
   ```bash
   # Test PyCodeRL
   python docker_test.py
   
   # Run examples
   python examples/example_simple.py
   
   # Start training
   python examples/example_training.py
   ```

3. **Your local files are mounted** to `/workspace` so changes persist

## 🐛 Troubleshooting

**Docker not installed:**
```bash
# macOS
brew install docker
# Or download Docker Desktop from docker.com
```

**Permission errors:**
```bash
chmod +x docker-*.sh
```

**Build failures:**
- Ensure Docker is running
- Check internet connection for package downloads
- Try: `docker system prune` to clean up

**Slow performance:**
- This is expected due to x86-64 emulation on ARM64
- Use GitHub Codespaces for better performance

## 🆚 Docker vs Codespaces

| Feature | Docker | Codespaces |
|---------|--------|------------|
| **Setup** | Manual install | Browser-based |
| **Performance** | Slower (emulation) | Native x86-64 |
| **Cost** | Free (local) | Free tier available |
| **Use Case** | Local development | Remote development |

## 💡 Tips

- Use `docker-dev.sh` for interactive development
- Use `docker-run.sh` for quick testing
- Mount your workspace for persistent changes
- Use Codespaces for CPU-intensive tasks
