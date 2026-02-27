import pandas as pd
import json

# 文件路径
file_path = "/aul/homes/zyin007/zb/assignment5-alignment/data/ultrachat_200k/data/train_sft-00000-of-00003-a3ecf92756993583.parquet"

def inspect_parquet(path, num_rows=3):
    print(f"正在加载文件: {path}\n")
    
    # 1. 加载数据
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 2. 打印基本信息
    print(f"数据总行数: {len(df)}")
    print(f"数据列名: {df.columns.tolist()}")
    print("-" * 50)

    # 3. 显示前几条
    # 设置 pandas 显示选项，防止文本被截断
    pd.set_option('display.max_colwidth', None)
    
    for i in range(min(num_rows, len(df))):
        print(f"--- 第 {i} 条数据 ---")
        row = df.iloc[i]
        
        # 遍历每一列并打印
        for col in df.columns:
            content = row[col]
            # 如果内容是列表或字典（UltraChat 常见格式），格式化一下
            if isinstance(content, (list, dict)):
                print(f"[{col}]:\n{json.dumps(content, indent=2, ensure_ascii=False)}")
            else:
                print(f"[{col}]: {content}")
        print("\n")

if __name__ == "__main__":
    inspect_parquet(file_path)