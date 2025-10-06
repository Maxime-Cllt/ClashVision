<div align="center">
  <h1>ClashVision</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Version-1.0.0-success?style=for-the-badge" alt="Version" />
</div>


<div align="center">
  <img src="assets/val_batch0_labels.png" width="500px" alt="ClashVision"/>
</div>

## 📖 Overview

**ClashVision** is a computer vision project that leverages advanced deep learning techniques to provide
state-of-the-art object detection like gold storage and elixir storage detection in Clash of Clans game.
Built with PyTorch, ClashVision is designed to be modular, scalable, and easy to use for both research and production
environments.

## ✨ Features

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

## 📊 Model Summary

### Model Architecture

72 layers, 3,006,038 parameters, 0 gradients, 8.1 GFLOPs

<div align="center">
    <table>
      <thead>
        <tr>
          <th style="padding:8px 16px;">Layer</th>
          <th style="padding:8px 16px;">Parameters</th>
          <th style="padding:8px 16px;">Gradients</th>
          <th style="padding:8px 16px;">GFLOPs</th>
        </tr>
      </thead>
      <tbody align="center">
        <tr>
          <td>72</td>
            <td>3,006,038</td>
            <td>0</td>
            <td>8.1</td>
        </tr>
      </tbody>
    </table>
</div>

### Training and Validation Curves

<div align="center">
  <img src="assets/results.png" width="500px" alt="Training and Validation Curves"/>
</div>

### Inference Example

<div align="center">
  <img src="assets/val_batch0_pred.png" width="500px" alt="Training and Validation Curves"/>
</div>

## 🤝 Contributing

We welcome contributions to enhance the capabilities of ClashVision.