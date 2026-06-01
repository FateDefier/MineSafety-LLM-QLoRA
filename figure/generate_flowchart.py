import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Chinese font setup
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
colors = {
    'header': '#2563EB',
    'data': '#DBEAFE',
    'train': '#FEF3C7',
    'eval': '#D1FAE5',
    'arrow': '#6B7280',
    'text': '#1F2937',
    'header_text': '#FFFFFF',
}

# Layout: 3 columns
col_x = [1.5, 5.5, 9.5]
col_w = 3.2
node_h = 0.65
gap = 0.25

# Column headers
headers = ['数据收集', '训练', '评估']
for i, (x, h) in enumerate(zip(col_x, headers)):
    rect = FancyBboxPatch((x - col_w/2, 7.1), col_w, 0.6,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['header'], edgecolor='none')
    ax.add_patch(rect)
    ax.text(x, 7.4, h, ha='center', va='center',
            fontsize=14, fontweight='bold', color=colors['header_text'])

# Nodes for each column
col_nodes = [
    ['PDF 规程文档', 'MinerU 转 Markdown', 'Easy Dataset 生成 QA', 'AI 质量评估+过滤'],
    ['AutoDL 云 GPU', 'QLoRA 4-bit 训练', 'Loss 曲线分析', '模型导出'],
    ['15 道专业测试题', 'GPT-5.5 盲评', '5 维度评分'],
]

bg_colors = [colors['data'], colors['train'], colors['eval']]

node_positions = {}  # (col, idx) -> (x, y)

for col_idx, (nodes, bg) in enumerate(zip(col_nodes, bg_colors)):
    x = col_x[col_idx]
    n = len(nodes)
    # Center nodes vertically
    total_h = n * node_h + (n - 1) * gap
    start_y = 6.6 - (6.6 - total_h) / 2 - node_h  # leave space below header

    for i, label in enumerate(nodes):
        y = 6.6 - i * (node_h + gap) - node_h / 2 - 0.3
        rect = FancyBboxPatch((x - col_w/2, y - node_h/2), col_w, node_h,
                               boxstyle="round,pad=0.12",
                               facecolor=bg, edgecolor='#94A3B8', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=11, color=colors['text'])
        node_positions[(col_idx, i)] = (x, y)

# Arrows within columns
for col_idx in range(3):
    nodes = col_nodes[col_idx]
    for i in range(len(nodes) - 1):
        x, y1 = node_positions[(col_idx, i)]
        _, y2 = node_positions[(col_idx, i + 1)]
        ax.annotate('', xy=(x, y2 + node_h/2 + 0.05), xytext=(x, y1 - node_h/2 - 0.05),
                     arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# Arrows between columns (col 0 → col 1, col 1 → col 2)
# Last node of col 0 → first node of col 1
for src_col, dst_col in [(0, 1), (1, 2)]:
    sx, sy = node_positions[(src_col, len(col_nodes[src_col]) - 1)]
    dx, dy = node_positions[(dst_col, 0)]
    ax.annotate('', xy=(dx - col_w/2 - 0.05, dy), xytext=(sx + col_w/2 + 0.05, sy),
                 arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.8,
                                connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('D:/VSCode/Project/QLoRA/figure/methodology.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved figure/methodology.png')
