import glob
import os

def extract_info_from_files():
    train_files = glob.glob('../../../../3*/*/*/*/ana/*train.txt')
    test_files = glob.glob('../../../../3*/*/*/*/ana/*test.txt')

    results = {}

    for file_path in train_files:
        parts = file_path.split(os.sep)
        if len(parts) >= 8:
            key = (parts[-5], parts[-4], parts[-3])  # L001, 5_lr_-5, test1
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            rmsd = pearson = None
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
        if len(parts) >= 8:
            key = (parts[-5], parts[-4], parts[-3])
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            rmsd = pearson = None
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

# ======== 輸出（抓 L001, 5_lr_-5, test1）========
data = extract_info_from_files()

print(f"{'Level1':<10} {'Level2':<12} {'Level3':<10} {'Train_RMSD':>10} {'Train_Pearson':>14} {'Test_RMSD':>10} {'Test_Pearson':>14}")
for (level1, level2, level3), values in data.items():
    print(f"{level1:<10} {level2:<12} {level3:<10} "
          f"{values['train_rmsd']:10.4f} {values['train_pearson']:14.4f} "
          f"{values['test_rmsd']:10.4f} {values['test_pearson']:14.4f}")

