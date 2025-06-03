def extract_best_train_rmsd(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = lines[0]
    data_lines = lines[1:]
    grouped = {}
    for line in data_lines:
        parts = line.strip().split()
        level1 = parts[0]
        train_rmsd = float(parts[3])  # Train_RMSD 欄
        if level1 not in grouped or train_rmsd < float(grouped[level1][3]):
            grouped[level1] = parts
    sorted_keys = sorted(grouped.keys(), key=lambda x: int(x[1:]))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        for k in sorted_keys:
            row = grouped[k]
            f.write("{:<10} {:<12} {:<10} {:>10} {:>14} {:>10} {:>14}\n".format(
                row[0], row[1], row[2],
                row[3], row[4], row[5], row[6]
            ))

extract_best_train_rmsd('../1_collect/rmsd', 'rmsd')

