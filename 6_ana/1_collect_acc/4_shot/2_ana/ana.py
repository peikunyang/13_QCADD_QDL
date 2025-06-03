import pandas as pd

# 讀入資料
df = pd.read_csv("../1_collect/rmsd", delim_whitespace=True)

# 定義 Level2 的順序
level2_order = ['N_Shot_3', 'N_Shot_4', 'N_Shot_5', 'N_Shot_6']
df['Level2'] = pd.Categorical(df['Level2'], categories=level2_order, ordered=True)
df = df.sort_values(by=['Level1', 'Level2'])

# 將每一組 Level1 資料整理為一行
rows = []
for level1, group in df.groupby("Level1"):
    group_sorted = group.sort_values("Level2")
    row = [level1]
    row += group_sorted["Train_RMSD"].tolist()
    row += group_sorted["Train_Pearson"].tolist()
    row += group_sorted["Test_RMSD"].tolist()
    row += group_sorted["Test_Pearson"].tolist()
    rows.append(row)

# 設定欄寬
col_width = 8

# 第一列標題
header1_parts = [
    "Level1".ljust(col_width),
    "Train_RMSD".center(col_width * 4),
    "Train_Pearson".center(col_width * 4),
    "Test_RMSD".center(col_width * 4),
    "Test_Pearson".center(col_width * 4),
]
header1 = "".join(header1_parts)

# 第二列標題（對應各個 QMLunit 設定）
header2 = "".ljust(col_width) + "".join(str(i).rjust(col_width) for i in [3, 4, 5, 6] * 4)

# 定義資料列格式
def format_row(row):
    level = str(row[0]).ljust(col_width)
    tr_rmsd = ["{:.2f}".format(x) for x in row[1:5]]
    tr_pcc  = ["{:.3f}".format(x) for x in row[5:9]]
    te_rmsd = ["{:.2f}".format(x) for x in row[9:13]]
    te_pcc  = ["{:.3f}".format(x) for x in row[13:17]]
    return level + "".join(x.rjust(col_width) for x in tr_rmsd + tr_pcc + te_rmsd + te_pcc)

# 輸出到檔案
with open("rmsd", "w") as f:
    f.write(header1 + "\n")
    f.write(header2 + "\n")
    for row in rows:
        f.write(format_row(row) + "\n")

