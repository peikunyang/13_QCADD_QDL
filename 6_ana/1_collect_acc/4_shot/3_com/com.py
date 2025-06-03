import pandas as pd

# 定義欄位名稱（已擴充到 4 組）
cols = [
    "Level1",
    "Train_RMSD_1", "Train_RMSD_2", "Train_RMSD_3", "Train_RMSD_4",
    "Train_Pearson_1", "Train_Pearson_2", "Train_Pearson_3", "Train_Pearson_4",
    "Test_RMSD_1", "Test_RMSD_2", "Test_RMSD_3", "Test_RMSD_4",
    "Test_Pearson_1", "Test_Pearson_2", "Test_Pearson_3", "Test_Pearson_4"
]

# 讀取主表與補充表
df1 = pd.read_csv("../2_ana/rmsd", sep="\s+", skiprows=2, header=None, names=cols)
df2 = pd.read_csv("../../3_check_par/2_select/rmsd", sep="\s+")

# 統一 Level1 格式
df1["Level1"] = df1["Level1"].astype(str)
df2["Level1"] = df2["Level1"].astype(str)

# 合併並重新命名 df2 裡的欄位以對應第 4 組
df2 = df2.rename(columns={
    "Train_RMSD": "Train_RMSD_4",
    "Train_Pearson": "Train_Pearson_4",
    "Test_RMSD": "Test_RMSD_4",
    "Test_Pearson": "Test_Pearson_4"
})

df_merged = df1.merge(df2[["Level1", "Train_RMSD_4", "Train_Pearson_4", "Test_RMSD_4", "Test_Pearson_4"]], on="Level1", how="left")

# 設定每欄寬度
col_width = 7

# 自動群組欄位名稱
header_groups = {
    "Train_RMSD": [col for col in df_merged.columns if col.startswith("Train_RMSD")],
    "Train_Pearson": [col for col in df_merged.columns if col.startswith("Train_Pearson")],
    "Test_RMSD": [col for col in df_merged.columns if col.startswith("Test_RMSD")],
    "Test_Pearson": [col for col in df_merged.columns if col.startswith("Test_Pearson")]
}

# 第一列標題（群組名）
header1 = "Level1".ljust(col_width)
for group in header_groups:
    header1 += group.center(col_width * len(header_groups[group]))

# 第二列標題（QMLunit 數字 1~4）
header2 = "".ljust(col_width)
for group in header_groups:
    header2 += "".join(str(i + 1).rjust(col_width) for i in range(len(header_groups[group])))

# 格式化每列資料
def format_row(row):
    values = [str(row['Level1']).ljust(col_width)]
    for group in header_groups:
        for col in header_groups[group]:
            val = row[col]
            if "RMSD" in col:
                values.append(f"{val:.2f}".rjust(col_width))
            else:
                values.append(f"{val:.3f}".rjust(col_width))
    return "".join(values)

# 輸出到檔案
with open("rmsd", "w") as f:
    f.write(header1 + "\n")
    f.write(header2 + "\n")
    for _, row in df_merged.iterrows():
        f.write(format_row(row) + "\n")

