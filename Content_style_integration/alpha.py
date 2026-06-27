import pandas as pd

# 读取txt文件路径
file_path = './alpha_values.txt'

# 初始化存储
data = {}

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 分割字符串：Lora名_safetensors_层名: [[alpha值]]
        lora_full, value_part = line.split(':')
        alpha_value = float(value_part.strip().lstrip('[').lstrip('[').rstrip(']').rstrip(']'))

        # 拆出 Lora 名和 层名
        if '_safetensors_' in lora_full:
            lora_name, layer_name = lora_full.split('_safetensors_', 1)
        else:
            continue  # 跳过不匹配的行

        # 填入数据
        if layer_name not in data:
            data[layer_name] = {}
        data[layer_name][lora_name] = alpha_value

# 提取所有唯一 Lora 名和层名
all_loras = sorted({lora for layer in data.values() for lora in layer.keys()})
all_layers = sorted(data.keys())

# 构建 DataFrame，默认填1
df = pd.DataFrame(1, index=all_layers, columns=all_loras)

# 填入实际的 alpha 值
for layer, lora_values in data.items():
    for lora, alpha in lora_values.items():
        df.at[layer, lora] = alpha
# 输出结果
print(df)
