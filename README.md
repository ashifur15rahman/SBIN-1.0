# TemporalPhysicsNet

This repository contains the implementation of the `TemporalPhysicsNet` model, a physics-informed neural network designed for temporal modeling of time series data. The model integrates physical laws through Lagrangian, Hamiltonian, and Brachistochrone losses to ensure predictions are physically consistent.

## Overview

The `TemporalPhysicsNet` model is a 1D convolutional neural network (CNN) that maps input features and time to a sequence of coordinates (`theta`), mass (`mass`), and potential (`V`). It is trained using a combination of physics-based loss functions to ensure the learned dynamics adhere to physical laws.

### Key Features

- **Physics-Informed Losses**:
  - **Euler-Lagrange Residual**: Ensures the dynamics satisfy the Euler-Lagrange equations.
  - **Hamiltonian Residual**: Enforces conservation of the Hamiltonian (energy).
  - **Brachistochrone Cost**: Optimizes for time-optimal trajectories.
  - **Relation Tensor Constraint**: Enforces geometric consistency using clustering.

- **Model Architecture**:
  - A 1D CNN with ReLU activations for feature extraction.
  - 1x1 convolutions to predict mass and potential.

- **Training**:
  - Uses Adam optimizer with a learning rate of `1e-3`.
  - Trains for 100 epochs by default.

## Usage

1. **Data Preparation**:
   - The model expects a CSV file with columns: `time`, `lat`, `lon`, `vn`, `ve`.
   - Example: `20220712_0_1.csv`

2. **Running the Code**:
   - Ensure all dependencies are installed (see `requirements.txt`).
   - Run the script: `python temporal_physics_net.py`

3. **Customization**:
   - Adjust hyperparameters like `epochs`, `lr`, and `clusters` in the `train` function.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `torch`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Code Structure

- **`temporal_physics_net.py`**: Main Python file containing the model and training loop.
- **`README.md`**: This file, providing an overview and usage instructions.
- **`requirements.txt`**: Lists the required Python packages.

## Notes

- The code has been updated to fix typos and improve readability.
- The relation tensor constraint uses KMeans clustering to enforce geometric consistency.
- The model is designed for temporal data with underlying physical dynamics, making it suitable for applications in physics, engineering, and climate science.
