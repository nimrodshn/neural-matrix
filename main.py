import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # total neurons
f_E = 0.5  # fraction of excitatory neurons
N_E = int(N * f_E)
N_I = N - N_E

# We would like a "balanced model" where the the following equation holds - 
# mu_E*f_E + mu_I*(1-f_E) = 0 
# meaning their weighted average is zero.
mu_E, sigma_E = 0.5,  1 / np.sqrt(N)  # excitatory mean/variance
mu_I, sigma_I = -1*(f_E*mu_E)/(1-f_E),  1 / np.sqrt(N) # inhibitory mean/variance

# Create an empty synaptic matrix
W = np.zeros((N, N))

# Assign excitatory and inhibitory neurons
# types = np.array([1] * N_E + [-1] * N_I)  # +1 = E, -1 = I
types = np.array([1] * N) # All exicatory
np.random.shuffle(types)

for j in range(N):
    if types[j] == 1:
        # Excitatory column: all positive
        W[:, j] = np.random.normal(mu_E / np.sqrt(N),sigma_E, size=N)
    else:
        # Inhibitory column: all negative
        W[:, j] = np.random.normal(mu_I / np.sqrt(N), sigma_I, size=N)

# Compute eigenvalues
eigvals = np.linalg.eigvals(W)

# Plot eigenvalue spectrum
plt.figure(figsize=(6, 6))
plt.scatter(eigvals.real, eigvals.imag, s=5, color='blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.title("Eigenvalue Spectrum of Synaptic Matrix with Dale's Law")
plt.xlabel("Re")
plt.ylabel("Im")
plt.grid(True)
plt.show()