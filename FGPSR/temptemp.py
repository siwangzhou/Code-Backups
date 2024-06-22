import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 创建新的图表和坐标轴
fig, ax = plt.subplots()

# 添加矩形以示意Dense Block的层
previous_input = Rectangle((0.1, 0.6), 0.2, 0.2, edgecolor='black', facecolor='none')
ax.add_patch(previous_input)
ax.text(0.1, 0.7, 'Input', verticalalignment='center', horizontalalignment='center')

# Dense Block的层
for i in range(1, 5):  # 画4个dense层作为例子
    left = i * 0.2
    block = Rectangle((left, 0.5 + (0.1 * i)), 0.2, 0.2, edgecolor='black', facecolor='none')
    ax.add_patch(block)
    ax.text(left, 0.6 + (0.1 * i), f'Layer {i}', verticalalignment='center', horizontalalignment='center')

    # 连接输入和层
    for j in range(i):
        ax.arrow(j * 0.2 + 0.2, 0.6 + (0.1 * j), (i - j - 1) * 0.2, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')

# 设置标题和展示图表
ax.set_title('Dense Block Schematic')
plt.axis('off')  # 关闭坐标轴
plt.show()