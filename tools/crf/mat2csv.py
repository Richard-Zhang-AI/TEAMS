import scipy.io as sio
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 1) 基本配置
mat_path = '/mnt/sdb1/leijh/EnergySnake1/18_0crf-image-labeling-master-vertebrae/original_version/rpn_and_detections_train0.mat'
out_dir = Path('/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/tools/crf/')
out_dir.mkdir(parents=True, exist_ok=True)

mat_data = sio.loadmat(mat_path, simplify_cells=True)

# 2) 导出“指定变量”为 CSV/JSON（以 mrcnn_boxes_and_scores_all 为例）
var = 'mrcnn_boxes_and_scores_all'
if var not in mat_data:
    raise KeyError(f'变量 {var} 不在 {mat_path} 中，可先打印 keys 检查。')

arr = np.array(mat_data[var])

# —— 导出为 CSV —— #
if arr.ndim == 1:
    df = pd.DataFrame(arr, columns=[var])
    df.to_csv(out_dir / f'{var}.csv', index=False)
elif arr.ndim == 2:
    df = pd.DataFrame(arr)
    df.to_csv(out_dir / f'{var}.csv', index=False)
elif arr.ndim == 3:
    # 常见做法：将前两维展平为行，最后一维作为列
    n1, n2, nf = arr.shape
    flat = arr.reshape(n1 * n2, nf)
    col_names = [f'feat_{i}' for i in range(nf)]
    df = pd.DataFrame(flat, columns=col_names)
    # 可选：保留原始前两维索引方便回溯
    df.insert(0, 'dim1', np.repeat(np.arange(n1), n2))
    df.insert(1, 'dim2', np.tile(np.arange(n2), n1))
    df.to_csv(out_dir / f'{var}.csv', index=False)
else:
    # 维度更高时建议保存为 JSON 或者 .npy/.npz
    np.save(out_dir / f'{var}.npy', arr)

# —— 导出为 JSON —— #
with open(out_dir / f'{var}.json', 'w', encoding='utf-8') as f:
    json.dump({var: arr.tolist()}, f, ensure_ascii=False)
print(f'已导出: {out_dir}/{var}.csv 和 {out_dir}/{var}.json（如存在）')