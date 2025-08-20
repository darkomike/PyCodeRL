# PyCodeRL GitHub Codespaces Setup

This directory contains the configuration for running PyCodeRL in GitHub Codespaces with proper x86-64 support.

## 🚀 Quick Start

1. **Open in Codespaces:**
   - Go to your PyCodeRL repository on GitHub
   - Click the green "Code" button
   - Select "Codespaces" tab
   - Click "Create codespace on main"

2. **Wait for Setup:**
   - The devcontainer will automatically install all dependencies
   - This takes about 2-3 minutes on first launch

3. **Test PyCodeRL:**
   ```bash
   python test_pycode_rl.py
   ```

4. **Run Examples:**
   ```bash
   python examples/example_simple.py
   python examples/example_training.py
   python examples/example_evaluation.py
   ```

## 🏗️ What's Included

- **x86-64 Linux environment** (Ubuntu-based)
- **Python 3.11** with all PyCodeRL dependencies
- **GCC toolchain** for assembly compilation
- **VS Code extensions** for Python development
- **Jupyter support** for interactive development

## 🎯 Key Advantages

- ✅ **Native x86-64 execution** (no ARM64 compatibility issues)
- ✅ **Pre-configured environment** (no manual setup required)
- ✅ **Free tier available** (120 core hours/month)
- ✅ **VS Code integration** (familiar development environment)
- ✅ **Git integration** (seamless commit/push workflow)

## 📁 Files

- `devcontainer.json` - Main configuration
- `setup.sh` - Post-creation setup script
- `README.md` - This file

## 🔧 Architecture Details

The devcontainer:
- Uses Microsoft's official Python devcontainer image
- Runs on x86-64 architecture (Intel/AMD)
- Includes build tools for assembly compilation
- Sets up proper Python environment with all dependencies
- Configures VS Code with relevant extensions

## ⚡ Performance

Expected performance in Codespaces:
- **2-core machine**: Good for development and testing
- **4-core machine**: Better for training and benchmarks
- **Assembly execution**: Full compatibility (unlike ARM64 Mac)

## 🆘 Troubleshooting

If you encounter issues:

1. **Codespace won't start**: Try rebuilding the container
2. **Assembly errors**: Verify you're on x86-64 (run `uname -m`)
3. **Permission errors**: The container runs as `vscode` user with sudo access
4. **Package issues**: Rebuild container to get latest dependencies

## 💡 Tips

- Use the integrated terminal for running PyCodeRL commands
- Jupyter notebooks are available on port 8888
- The workspace is automatically synced with your GitHub repo
- You can commit and push changes directly from Codespaces
