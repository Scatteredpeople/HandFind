import pandas as pd

# 手势识别数据集的二次清洗与标签重构

# 读取CSV文件
file_path = r".\data\find.csv"
df = pd.read_csv(file_path)

# 删除第2列（索引1）和第127列（索引126）**同时** 为 0 或 NaN 的行
df = df[~((df[df.columns[1]].isna() | (df[df.columns[1]] == 0)) &
          (df[df.columns[126]].isna() | (df[df.columns[126]] == 0)))]

# 修改label列的标签名为Image_ID的前缀
df['label'] = df['Image_ID'].apply(lambda x: str(x).split('_')[0])

# 保存处理后的数据到新文件
new_file_path = r".\data1\change.csv"
df.to_csv(new_file_path, index=False)

print(f"处理完成，数据已保存至 {new_file_path}")
