import os
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==================== 参数区域 ====================
img_dir = "./"          # PNG 所在文件夹
pattern = "*.png"           # 匹配所有 png
save_path = "combined.png"  # 输出的组合图文件名
n_cols = 3                  # 每行放几张图，自行修改
dpi = 600                   # 导出分辨率
# ==================================================

png_files = sorted(glob.glob(os.path.join(img_dir, pattern)))
n_imgs = len(png_files)
if n_imgs == 0:
    raise ValueError("指定文件夹下没有 PNG 文件")

n_rows = math.ceil(n_imgs / n_cols)

# 根据行列数设置画布大小（可以按需要调节每张子图尺寸）
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4 * n_cols, 4 * n_rows),
                         dpi=dpi)

# axes 统一拉平成一维方便遍历
axes = axes.flatten() if isinstance(axes, (list, tuple)) or hasattr(axes, "flatten") else [axes]

for ax, img_path in zip(axes, png_files):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis("off")        # 不要坐标轴
    # 不设置 title，这样就没有标题了

# 多出来的空子图隐藏掉
for ax in axes[n_imgs:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
plt.show()

print(f"已保存组合图：{save_path}")
