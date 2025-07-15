# SBIN-1.0
Model Description
The TemporalPhysicsNet is a physics-informed neural network designed to model temporal dynamics in time-series data, ensuring that predictions adhere to physical laws. It leverages a 1D Convolutional Neural Network (CNN) architecture to process sequential input data and incorporates physics-based constraints during training. This makes it particularly suitable for applications in physics, engineering, and navigation, where physical consistency is as important as predictive accuracy.

Key Components
Input: The model accepts a sequence of features (e.g., latitude, longitude, velocities like vn, ve, etc.) along with a time dimension. For the given dataset, these could include columns such as lat, lon, vn, ve, roll, pitch, yaw, etc.
Architecture:
1D CNN: A convolutional network with ReLU activations processes the input sequence to predict a coordinate sequence, denoted as θ, which represents the system's state over time.
Mass and Potential Prediction: Two additional 1x1 convolutional layers predict the mass (diagonal entries) and a potential scalar from θ and time, capturing the physical properties of the system.
Physics-Informed Losses: The model enforces physical consistency through specialized loss functions:
Euler-Lagrange Residual: Ensures the dynamics satisfy the Euler-Lagrange equations, a cornerstone of classical mechanics for describing motion.
Hamiltonian Residual: Enforces conservation of the Hamiltonian, representing the system's total energy.
Brachistochrone Cost: Optimizes for time-optimal trajectories, inspired by the Brachistochrone problem in physics.
Relation Tensor Constraint: Uses clustering to maintain geometric consistency across similar data points.
Training: The model is trained using the Adam optimizer with a default learning rate of 1e-3 over 100 epochs, balancing data-driven learning with physical constraints.
Purpose
The TemporalPhysicsNet is designed to handle time-series data with underlying physical dynamics, such as navigation or motion tracking, as seen in the provided dataset. By integrating physical laws into the learning process, it produces predictions that are both accurate and physically plausible, making it ideal for scenarios with noisy or limited data.
