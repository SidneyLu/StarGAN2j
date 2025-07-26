import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.legend_handler import HandlerLine2D


def load_json_file(file_path):
    """从文件加载JSON数据并处理可能的错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在！")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"错误位置: 行 {e.lineno}, 列 {e.colno}")
        exit(1)
    except Exception as e:
        print(f"未知错误: {e}")
        exit(1)


# 直接指定文件名
file_path1 = 'expr/checkpoints/logj/G.json'
file_path2 = 'expr/checkpoints/logj/D.json'

# 加载两个文件的数据
print(f"正在从 '{file_path1}' 和 '{file_path2}' 加载数据...")
data1 = load_json_file(file_path1)
data2 = load_json_file(file_path2)

# 提取迭代次数和损失类型
iterations1 = list(data1.keys())
iterations2 = list(data2.keys())

if not iterations1 or not iterations2:
    print("错误: JSON数据中没有找到迭代数据！")
    exit(1)

loss_types1 = list(data1[iterations1[0]].keys())
loss_types2 = list(data2[iterations2[0]].keys())


# 提取每个迭代的损失值
def extract_loss_data(data, iterations, loss_types):
    loss_data = {loss_type: [] for loss_type in loss_types}
    for iteration in iterations:
        if iteration in data:
            for loss_type in loss_types:
                if loss_type in data[iteration]:
                    loss_data[loss_type].append(data[iteration][loss_type])
                else:
                    loss_data[loss_type].append(None)  # 缺失值
        else:
            # 迭代不存在，为所有损失类型添加缺失值
            for loss_type in loss_types:
                loss_data[loss_type].append(None)
    return loss_data


loss_data1 = extract_loss_data(data1, iterations1, loss_types1)
loss_data2 = extract_loss_data(data2, iterations2, loss_types2)

# 为每个迭代生成数值索引
iteration_indices1 = list(range(1, len(iterations1) + 1))
iteration_indices2 = list(range(1, len(iterations2) + 1))

# 创建图形和两个子图（上下布局）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(153.6, 86.4), sharex=False)

# 生成颜色映射，覆盖可能的最大损失类型数量
max_loss_types = max(len(loss_types1), len(loss_types2))
colors = plt.cm.tab20(np.linspace(0, 1, max_loss_types))

# 在上部子图绘制G.json的数据
for i, loss_type in enumerate(loss_types1):
    # 过滤掉缺失值
    valid_indices1 = [idx for idx, val in zip(iteration_indices1, loss_data1[loss_type]) if val is not None]
    valid_values1 = [val for val in loss_data1[loss_type] if val is not None]

    ax1.plot(valid_indices1, valid_values1, ':',
             label=loss_type, color=colors[i], linewidth=2)

ax1.set_title(f'Generator Losses', fontsize=192)
ax1.set_ylabel('Loss Values', fontsize=192)
ax1.tick_params(axis='y', labelsize=192)
ax1.set_ylim(0, 4)
ax1.grid(True, linestyle='--', alpha=0.7)
leg1 = ax1.legend(fontsize=108, loc='upper right')

for line in leg1.get_lines():
    line.set_linewidth(48)

# 在下部子图绘制D.json的数据
for i, loss_type in enumerate(loss_types2):
    # 过滤掉缺失值
    valid_indices2 = [idx for idx, val in zip(iteration_indices2, loss_data2[loss_type]) if val is not None]
    valid_values2 = [val for val in loss_data2[loss_type] if val is not None]

    ax2.plot(valid_indices2, valid_values2, ':',
             label=loss_type, color=colors[i], linewidth=2)

ax2.set_title(f'Discriminator Losses', fontsize=192)
ax2.set_xlabel('Iterations', fontsize=192)
ax2.set_ylabel('Loss Values', fontsize=192)
ax2.tick_params(axis='x', labelsize=192, rotation=45)
ax2.tick_params(axis='y', labelsize=192)
ax2.set_ylim(0, 4)
ax2.grid(True, linestyle='--', alpha=0.7)
leg2 = ax2.legend(fontsize=108, loc='upper right')

for line in leg2.get_lines():
    line.set_linewidth(48)
# 显示图表
plt.show()