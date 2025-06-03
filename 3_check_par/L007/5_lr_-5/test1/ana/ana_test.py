import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

file_path = "../Result/E_test"
data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Experimental", "Calculated", "Other"])

experimental = data["Experimental"].values
calculated = data["Calculated"].values

rmsd = np.sqrt(np.mean((experimental - calculated) ** 2))
correlation, _ = pearsonr(experimental, calculated)

with open("L07_5_test.txt", "w") as f:
    f.write(f"RMSD: {rmsd:.6f}\n")
    f.write(f"Pearson Correlation: {correlation:.6f}\n")

x_min, x_max = np.floor(experimental.min()), np.ceil(experimental.max())
y_min, y_max = x_min, x_max

plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(experimental, calculated, alpha=0.7, color='gray')
plt.xticks(np.arange(-20, 1, 5), fontsize=24)
plt.yticks(np.arange(-20, 1, 5), fontsize=24)
plt.xlim(-21, 1)
plt.ylim(-21, 1)
plt.grid(True, which='major', linestyle='--', linewidth=1)
text = (r"$\#\mathrm{QML}_{\mathrm{unit}}$ = 7"
        f"\nRMSD = {rmsd:.2f}"
        f"\nPearson = {correlation:.3f}")
plt.text(-20, 0, text, fontsize=28, verticalalignment='top')
plt.savefig("L07_5_test.png", dpi=100, bbox_inches='tight')
plt.show()

