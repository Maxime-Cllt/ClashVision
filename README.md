<div align="center">
  <h1>ClashVision</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Version-1.0.0-success?style=for-the-badge" alt="Version" />
</div>

## 📖 Overview

**ClashVision** is a computer vision project that leverages advanced deep learning techniques to provide
state-of-the-art object detection like gold storage and elixir storage detection in Clash of Clans game.
Built with PyTorch, ClashVision is designed to be modular, scalable, and easy to use for both research and production
environments.

## ✨ Features

- **Object Detection**: Detects and classifies various objects within images.
- **Pre-trained Models**: Comes with pre-trained models for quick deployment.

## 🔧 Pre-requisites

- **Python 3.13+**
- **pip** or **conda**
- **uv** (package manager, optional but recommended)
- **Virtual Environment** (recommended)
- **GPU** (recommended)

See `pyproject.toml` for a complete list of dependencies.

## 🚀 Installation

1. Run commands to set up the environment and install dependencies:

```bash
./scripts/setup-env.sh
```

2. If you are using a GPU, ensure that you have the appropriate CUDA toolkit installed.

## 🧪 Code quality

### Unit Tests available

To run unit tests and ensure code quality, run the following commands:

```bash
./scripts/run-pytest.sh
```

### Linting available

Linting is done using `ruff`. To check for linting issues, run:

```bash
.scripts/run-ruff.sh
```

### Formatting available

Code formatting is done using `black`. To format the code, run:

```bash
./scripts/run-black.sh
```

## 🤝 Contributing

We welcome contributions to enhance the capabilities of ClashVision.