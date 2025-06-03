import glob
import os

def extract_info_from_files():
    train_files = glob.glob('../../../../4*/*/*/*/*/ana/*train.txt')
    test_files = glob.glob('../../../../4*/*/*/*/*/ana/*test.txt')

    results = {}

    for file_path in train_files:
        parts = file_path.split(os.sep)
        if len(parts) >= 9:
            key = (parts[-6], parts[-5], parts[-4], parts[-3])  # Level1~4
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            rmsd, pearson = None, None
            for line in lines:
                if 'RMSD' in line:
                    try:
                        rmsd = float(line.strip().split(':')[1])
                    except:
                        pass
                elif 'Pearson Correlation' in line:
                    try:
                        pearson = float(line.strip().split(':')[1])
                    except:
                        pass
            results[key] = {
                'train_rmsd': rmsd, 'train_pearson': pearson,
                'test_rmsd': None, 'test_pearson': None
            }

    for file_path in test_files:
        parts = file_path.split(os.sep)
        if len(parts) >= 9:
            key = (parts[-6], parts[-5], parts[-4], parts[-3])
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            rmsd, pearson = None, None
            for line in lines:
                if 'RMSD' in line:
                    try:
                        rmsd = float(line.strip().split(':')[1])
                    except:
                        pass
                elif 'Pearson Correlation' in line:
                    try:
                        pearson = float(line.strip().split(':')[1])
                    except:
                        pass
            if key in results:
                results[key]['test_rmsd'] = rmsd
                results[key]['test_pearson'] = pearson
            else:
                results[key] = {
                    'train_rmsd': None, 'train_pearson': None,
                    'test_rmsd': rmsd, 'test_pearson': pearson
                }

    return results

# ================== 輸出 ==================
data = extract_info_from_files()

print(f"{'Level1':<12} {'Level2':<10} {'Level3':<12} {'Level4':<12} {'Train_RMSD':>12} {'Train_Pearson':>14} {'Test_RMSD':>12} {'Test_Pearson':>14}")
for (lv1, lv2, lv3, lv4), values in data.items():
    print(f"{lv1:<12} {lv2:<10} {lv3:<12} {lv4:<12} "
          f"{values['train_rmsd']:12.2f} {values['train_pearson']:14.3f} "
          f"{values['test_rmsd']:12.2f} {values['test_pearson']:14.3f}")

