#!/bin/bash
# PyCodeRL Cloud VM Setup Script
# Run this on any x86-64 Linux system (AWS EC2, Google Cloud, etc.)

set -e

echo "=== PyCodeRL Cloud VM Setup ==="
echo "Setting up PyCodeRL on x86-64 Linux system..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    binutils \
    build-essential \
    git \
    curl \
    wget \
    htop \
    vim

# Install Python dependencies
pip3 install --user --upgrade pip
pip3 install --user torch numpy matplotlib seaborn pandas psutil tqdm

# Clone and setup PyCodeRL (if not already present)
if [ ! -d "PyCodeRL" ]; then
    git clone https://github.com/your-username/PyCodeRL.git
    cd PyCodeRL
else
    cd PyCodeRL
    git pull
fi

# Run setup
python3 setup.py

echo "=== Setup Complete ==="
echo "PyCodeRL is now ready on this x86-64 system!"
echo ""
echo "Test the installation:"
echo "  python3 -c 'from pycode_rl_implementation import PyCodeRLCompiler; print(\"Success!\")'"
echo ""
echo "Run training:"
echo "  python3 pycode_rl_training.py"
echo ""
echo "Run evaluation:"
echo "  python3 pycode_rl_evaluation.py"
