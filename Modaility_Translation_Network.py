import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the input size, hidden size, and output size for the MLPs
input_size = 9  # (3 x 3) flattened size
hidden_size = 64
output_size = 32  # Latent space size

# Create the MLPs for amplitude and phase
amplitude_mlp = MLP(input_size, hidden_size, output_size)
phase_mlp = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer_amplitude = optim.Adam(amplitude_mlp.parameters(), lr=0.001)
optimizer_phase = optim.Adam(phase_mlp.parameters(), lr=0.001)

# Flatten the amplitude and phase tensors (assuming they are already loaded)
amplitude_flattened = amplitude_tensor.view(-1, input_size)
phase_flattened = phase_tensor.view(-1, input_size)

# Convert the flattened tensors to PyTorch tensors
amplitude_tensor = torch.Tensor(amplitude_flattened)
phase_tensor = torch.Tensor(phase_flattened)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass and compute the features in the CSI latent space
    amplitude_features = amplitude_mlp(amplitude_tensor)
    phase_features = phase_mlp(phase_tensor)

    # Compute the loss between the predicted features and the ground truth (if available)

    # Zero the gradients and perform backpropagation for amplitude MLP
    optimizer_amplitude.zero_grad()
    amplitude_loss = criterion(amplitude_features, ground_truth_amplitude)
    amplitude_loss.backward()
    optimizer_amplitude.step()

    # Zero the gradients and perform backpropagation for phase MLP
    optimizer_phase.zero_grad()
    phase_loss = criterion(phase_features, ground_truth_phase)
    phase_loss.backward()
    optimizer_phase.step()

    # Print the loss for monitoring
    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch+1}, Amplitude Loss: {amplitude_loss.item()}, Phase Loss: {phase_loss.item()}")