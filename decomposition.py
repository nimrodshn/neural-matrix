import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 500                  # Number of neurons
f = 0.8                  # Fraction of excitatory neurons
mu_E = 0.1               # Mean synaptic strength (excitatory)
mu_I = -0.1              # Mean synaptic strength (inhibitory)
sigma = 1.0 / np.sqrt(N) # Variance scaling for circular law

# 1. Construct J: Random matrix with zero mean and variance ~ 1/N
J = np.random.normal(loc=0.0, scale=sigma, size=(N, N))

# Optional: Enforce balance condition (row sums = 0)
J = J - J.mean(axis=1, keepdims=True)

# 2. Construct M: Rank-1 matrix
m = np.zeros(N)
num_E = int(f * N)
m[:num_E] = mu_E
m[num_E:] = mu_I
np.random.shuffle(m)  # Randomize neuron order

u = np.ones(N)  # Uniform vector
M = (1 / np.sqrt(N)) * np.outer(np.ones(N), m)  # Each row is (1/sqrt(N)) * m

# 3. Full synaptic matrix: W = J + M
W = J + M

# 4. Compute eigenvalues
eigs_J = np.linalg.eigvals(J)
eigs_W = np.linalg.eigvals(W)

# 5. Plot eigenvalues
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(eigs_J.real, eigs_J.imag, s=5, color='blue')
plt.title("Eigenvalues of J (balanced)")
plt.xlabel("Re(λ)")
plt.ylabel("Im(λ)")
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(eigs_W.real, eigs_W.imag, s=5, color='red')
plt.title("Eigenvalues of W = J + M")
plt.xlabel("Re(λ)")
plt.ylabel("Im(λ)")
plt.axis('equal')

plt.tight_layout()
plt.show()