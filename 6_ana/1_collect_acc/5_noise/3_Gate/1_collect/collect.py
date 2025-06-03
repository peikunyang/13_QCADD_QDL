import glob
import os

def extract_info_from_files():
    train_files = glob.glob('../../../../../5*/3*/*/*/*/ana/*train.txt')
    test_files = glob.glob('../../../../../5*/3*/*/*/*/ana/*test.txt')

    results = {}

    for file_path in train_files:
        parts = file_path.split(os.sep)
        if len(parts) >= 8:
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
            results[key] = {
                'train_rmsd': rmsd, 'train_pearson': pearson,
                'test_rmsd': None, 'test_pearson': None
            }

    for file_path in test_files:
        parts = file_path.split(os.sep)
        if len(parts) >= 8:
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

data = extract_info_from_files()

sorted_items = sorted(data.items(), key=lambda x: x[0][1])

print(f"{'Level2':<25} {'Level3':<10} {'Level4':<12} {'Train_RMSD':>10} {'Train_Pearson':>14} {'Test_RMSD':>10} {'Test_Pearson':>14}")

for (_, lv2, lv3, lv4), values in sorted_items:
    print(f"{lv2:<25} {lv3:<10} {lv4:<12} "
          f"{values['train_rmsd']:10.2f} {values['train_pearson']:14.3f} "
          f"{values['test_rmsd']:10.2f} {values['test_pearson']:14.3f}")

