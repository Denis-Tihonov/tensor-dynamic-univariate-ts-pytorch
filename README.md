# Hybrid Tensor-Neural Dynamic System Reconstruction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A hybrid approach combining tensor decomposition methods with neural networks for dynamic system reconstruction and time series forecasting.

## Overview

This repository implements a novel hybrid method for reconstructing dynamic systems from time series data. The approach integrates:

- **Neural network preprocessing** of input delay vectors
- **Multilinear tensor transformations** for state space reconstruction  
- **Canonical tensor decomposition** for parameter efficiency
- **Linear forecasting** in the reconstructed phase space

The method demonstrates 15-20% improvement in forecasting accuracy compared to traditional approaches on both synthetic (Lorenz attractor) and real-world (accelerometer) data.

## Key Features

- **Hybrid Architecture**: Neural networks + tensor methods
- **Multilinear Modeling**: Captures polynomial dependencies
- **Dimension Reduction**: Efficient state space reconstruction
- **Parameter Efficiency**: Tensor decomposition reduces model complexity
- **Flexible Preprocessing**: Customizable neural network layers
