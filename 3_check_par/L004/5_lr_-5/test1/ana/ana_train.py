import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取檔案
file_path = "../Result/E_train"  # 替換為您的檔案路徑
data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Experimental", "Calculated", "Other"])

# 提取實驗值和計算值
experimental = data["Experimental"].values
calculated = data["Calculated"].values

# 計算 RMSD
rmsd = np.sqrt(np.mean((experimental - calculated) ** 2))

# 計算 Pearson correlation
from scipy.stats import pearsonr
correlation, _ = pearsonr(experimental, calculated)

# 將 RMSD 和 Pearson correlation 存檔
with open("L04_5_train.txt", "w") as f:
    f.write(f"RMSD: {rmsd:.6f}\n")
    f.write(f"Pearson Correlation: {correlation:.6f}\n")

# 設置 X 和 Y 軸範圍
x_min, x_max = np.floor(experimental.min()), np.ceil(experimental.max())
y_min, y_max = x_min, x_max  # Y 軸範圍與 X 軸一致

# 繪製散佈圖
plt.figure(figsize=(10, 10), dpi=100)  # 增加尺寸並降低 DPI
plt.scatter(experimental, calculated, alpha=0.7, color='gray')

# 設置 X 和 Y 軸標籤字體大小
#plt.xlabel("Exp (Kcal/mol)", fontsize=50)
#plt.ylabel("Cal (Kcal/mol)", fontsize=50)

# 設置標題字體大小
#plt.title(f"RMSD = {rmsd:.2f}, Pearson = {correlation:.3f}", fontsize=36)

# 設置 X 和 Y 軸刻度字體大小
x_ticks = np.arange(-20,1,5)
plt.xticks(x_ticks, fontsize=24)
plt.yticks(x_ticks, fontsize=24)

# 設置 X 和 Y 軸範圍
plt.xlim(-21, 1)
plt.ylim(-21, 1)

# 繪製格線
plt.grid(True, which='major', linestyle='--', linewidth=1)

# 保存圖形，使用 bbox_inches='tight'
plt.savefig("L04_5_train.png", dpi=100, bbox_inches='tight')  # 保存高質量圖形
plt.show()

