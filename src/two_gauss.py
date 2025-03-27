import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Define parameters for two Gaussian distributions
# Left distribution (lower mean, lower variance)
mean1 = 2
std1 = 1.0  # standard deviation (square root of variance)

# Right distribution (higher mean, higher variance)
mean2 = 4
std2 = 1.5

# Create x values for plotting
x = np.linspace(-2, 8, 1000)

# Calculate probability densities for both distributions
pdf1 = norm.pdf(x, mean1, std1)
pdf2 = norm.pdf(x, mean2, std2)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the two distributions
plt.plot(x, pdf1, "r-", linewidth=2, label=f"Left Gaussian (μ={mean1}, σ={std1})")
plt.plot(x, pdf2, "g-", linewidth=2, label=f"Right Gaussian (μ={mean2}, σ={std2})")

# Calculate and shade the overlap
overlap = np.minimum(pdf1, pdf2)
plt.fill_between(x, overlap, color="blue", alpha=0.5, label="Overlap")

# Add labels and legend
# plt.xlabel("x", fontsize=12)
# plt.ylabel('Probability Density', fontsize=12)
# plt.title('Two Overlapping Gaussian Distributions', fontsize=14)
# plt.legend(fontsize=10)
# plt.grid(True, alpha=0.3)

# Ensure axes are visible
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
# plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
