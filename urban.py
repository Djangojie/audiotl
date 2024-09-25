import os
import random

def create_file_list(data_dir, train_file, test_file, test_ratio=0.2):
    file_paths = []

    # 遍历data_dir中的所有文件
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.relpath(os.path.join(root, file), data_dir)
                file_paths.append(file_path)

    # 打乱文件列表
    random.shuffle(file_paths)

    # 根据比例划分训练和测试集
    split_idx = int(len(file_paths) * (1 - test_ratio))
    train_paths = file_paths[:split_idx]
    test_paths = file_paths[split_idx:]

    # 写入文件
    with open(train_file, 'w') as f:
        for path in train_paths:
            f.write(f"{path}\n")

    with open(test_file, 'w') as f:
        for path in test_paths:
            f.write(f"{path}\n")

    print(f"训练集: {len(train_paths)} 个文件")
    print(f"测试集: {len(test_paths)} 个文件")

if __name__ == "__main__":
    data_dir = "/home/ubuntu/ZX/UrbanSound8K/audio/"
    train_file = "/home/ubuntu/ZX/UrbanSound8K/train_list.txt"
    test_file = "/home/ubuntu/ZX/UrbanSound8K/test_list.txt"

    create_file_list(data_dir, train_file, test_file)
