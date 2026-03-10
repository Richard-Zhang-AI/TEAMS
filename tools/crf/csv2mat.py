import pandas as pd
import numpy as np
import scipy.io as sio
import re
from pathlib import Path

csv_path = '/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/visual/test111/seq_disc_detections.csv'
save_mat_path = '/mnt/sdb1/leijh/EnergySnake1/EnergeSnake1/visual/test111/seq_disc_detections.mat'
var_name = 'mrcnn_boxes_and_scores_all'  # 变量名可自定义

K = 20  # 每图固定行数（检测数）

def natural_image_id_key(x: str):
    # 试图提取文件名中的数字用于排序，如 "522_image.png" -> 522；失败则回退为字符串
    base = Path(x).name
    m = re.search(r'(\d+)', base)
    return (0, int(m.group(1))) if m else (1, base)

def csv_to_mat_array(csv_file: str, K: int = 20):
    df = pd.read_csv(csv_file)
    # 统一列名（容错）
    cols = [c.strip().lower() for c in df.columns]
    name_map = {}
    for want in ['image_id', 'rank', 'x1n', 'y1n', 'x2n', 'y2n', 'class', 'score']:
        for c in df.columns:
            if c.strip().lower() == want:
                name_map[want] = c
                break
    required = ['image_id', 'rank', 'x1n', 'y1n', 'x2n', 'y2n', 'class', 'score']
    for r in required:
        if r not in name_map:
            raise ValueError(f'CSV缺少列: {r}')

    # 保留必要列
    df = df[[name_map[c] for c in required]].copy()
    df.columns = required

    # 类型转换
    df['rank'] = df['rank'].astype(int)
    for c in ['x1n','y1n','x2n','y2n','score']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype(np.float32)
    df['class'] = pd.to_numeric(df['class'], errors='coerce').fillna(0).astype(np.int32)

    # 唯一图像ID，做稳定排序
    image_ids = sorted(df['image_id'].unique().tolist(), key=natural_image_id_key)
    N = len(image_ids)

    # 构建 [N, K, 6]
    out = np.zeros((N, K, 6), dtype=np.float32)
    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}

    # 按行填充（rank 决定第2维位置，超界忽略；同rank多行则按score最高覆盖）
    # 先按 image_id, rank, score 降序排，确保同rank保留分数最高的
    df_sorted = df.sort_values(['image_id','rank','score'], ascending=[True, True, False])
    for img_id, g in df_sorted.groupby('image_id'):
        i = id_to_idx[img_id]
        for _, row in g.iterrows():
            r = int(row['rank'])
            if 0 <= r < K:
                x1n = row['x1n']; y1n = row['y1n']; x2n = row['x2n']; y2n = row['y2n']
                score = row['score']; cls_ = int(row['class'])
                out[i, r, :] = [x1n, y1n, x2n, y2n, score, float(cls_)]

    return out, image_ids

if __name__ == '__main__':
    arr, image_ids = csv_to_mat_array(csv_path, K=K)
    # 保存为 .mat
    sio.savemat(save_mat_path, {var_name: arr, 'image_ids': np.array(image_ids, dtype=object)})
    print(f'Saved: {save_mat_path}  shape={arr.shape}, var={var_name}')