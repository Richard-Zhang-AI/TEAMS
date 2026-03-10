import torch
import numpy as np

def pad_sequence_list(detection, seq_list, max_len):
    if max_len == 0:
        # 兼容全空的情况
        B = len(seq_list)
        return detection.new_zeros((B, 0, 6)), 0
    B = len(seq_list)
    out = detection.new_zeros((B, max_len, 6))
    lengths = []
    for b, det_b in enumerate(seq_list):
        n = det_b.size(0)
        lengths.append(int(n))
        if n > 0:
            out[b, :n] = det_b
    return out, lengths

def simple_nms_postprocess(detection, iou_threshold=0.5):
    """
    简单的NMS后处理，去除重叠框，保留置信度最高的框
    
    Args:
        detection: 检测结果张量 [N, 6] (x1, y1, x2, y2, score, class_id)
        iou_threshold: IoU阈值，超过此阈值的框被认为是重叠的
    
    Returns:
        过滤后的检测结果 [M, 6]，其中 M <= N
    """
    if detection.size(0) == 0:
        return detection
    
    # 按置信度降序排序
    scores = detection[:, 4]
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep_indices = []
    for i in sorted_indices:
        keep = True
        current_box = detection[i, :4]  # [x1, y1, x2, y2]
        
        # 检查与已保留框的重叠
        for j in keep_indices:
            kept_box = detection[j, :4]
            iou = calculate_iou(current_box, kept_box)
            if iou > iou_threshold:
                keep = False
                break
        
        if keep:
            keep_indices.append(i)
    # 返回过滤后的检测结果        
    keep_tensor = torch.tensor(keep_indices, device=detection.device, dtype=torch.long)
    return detection[keep_tensor]

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU值
    """
    # 计算交集区域
    x1_min = max(box1[0], box2[0])
    y1_min = max(box1[1], box2[1])
    x1_max = min(box1[2], box2[2])
    y1_max = min(box1[3], box2[3])
    
    # 如果没有交集
    if x1_min >= x1_max or y1_min >= y1_max:
        return 0.0
    
    # 计算交集面积
    inter_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def split_detection_sequences(detection, raw_detection, iou_threshold=0.2, gap_threshold_multiplier=2):
    """
    将检测结果拆分为三个序列：椎间盘、椎骨左侧、椎骨右侧
    
    Args:
        detection: 检测结果张量 [B, N, 6] (x1, y1, x2, y2, score, class_id)
        iou_threshold: NMS的IoU阈值
    
    Returns:
        dict: 包含三个序列的字典
    """
    B = detection.size(0)
    disc_list, left_list, right_list = [], [], []

    for b in range(B):
        det_b = detection[b]
        if raw_detection is not None:
            raw_det_b = raw_detection[b]
        else:
            raw_det_b = None
        # 过滤 padding 行
        keep = det_b[:, 4] > 0
        det_b = det_b[keep]
        # det_b = simple_nms_postprocess(det_b, iou_threshold=iou_threshold)
        if det_b.numel() == 0:
            disc_list.append(det_b.new_zeros((0, 6)))
            left_list.append(det_b.new_zeros((0, 6)))
            right_list.append(det_b.new_zeros((0, 6)))
            continue

        cls_b = det_b[:, 5].long()
        # 椎间盘 25~47
        discs = det_b[(cls_b >= 25) & (cls_b <= 47)]
        discs = simple_nms_postprocess(discs, iou_threshold=iou_threshold)
        if discs.size(0) > 0:
            ycenter = (discs[:, 1] + discs[:, 3]) * 0.5
            order = torch.argsort(ycenter)  # 由上到下
            discs = discs[order]
            # 漏检检测和补充
            if raw_det_b is not None:
                discs = detect_and_fill_missing_detections(
                    discs, raw_det_b, (25, 47), gap_threshold_multiplier
                )
        disc_list.append(discs)

        # 椎骨/棘突 0~24 -> 使用"两条平滑曲线"聚类：x≈f(y)，三次多项式，迭代优化划分
        bones_mask = (cls_b >= 0) & (cls_b <= 24)
        bones = det_b[bones_mask]
        bones = simple_nms_postprocess(bones, iou_threshold=iou_threshold)
        if bones.size(0) > 0:
            # 统计0-24类中每个类别的实例数量
            bones_cls = bones[:, 5].long()
            unique_classes, class_counts = torch.unique(bones_cls, return_counts=True)
            # 计算只有一个实例的类别比例
            single_instance_classes = (class_counts == 1).sum().item()
            total_classes = len(unique_classes)
            single_instance_ratio = single_instance_classes / total_classes if total_classes > 0 else 0
            # 如果绝大部分0-24类都只有一个实例（比例超过80%），则只分为一个序列
            if single_instance_ratio > 0.9:
                # 所有0-24类检测框都归类为left
                left_b = bones.clone()
                right_b = bones.new_zeros((0, 6))

            else:
                # 取每个框中心点 (cx, cy)
                cx = (bones[:, 0] + bones[:, 2]) * 0.5
                cy = (bones[:, 1] + bones[:, 3]) * 0.5

                pts = torch.stack([cx, cy], dim=1).detach().cpu().numpy().astype(np.float32)
                x = pts[:, 0]
                y = pts[:, 1]

                left_b = bones.new_zeros((0, bones.size(1)))
                right_b = bones.new_zeros((0, bones.size(1)))

                # 若点太少，直接按 x 中位数切分作为回退
                if pts.shape[0] < 6:
                    midx = np.median(x)
                    left_mask_np = x < midx
                    right_mask_np = ~left_mask_np
                else:
                    # 初始化：按 x 进行二分（鲁棒、无需外部库）
                    midx = np.median(x)
                    labels = (x >= midx).astype(np.int32)  # 0/1 两簇

                    def fit_poly(yv, xv, deg=3):
                        # 拟合 x = f(y)
                        try:
                            coef = np.polyfit(yv, xv, deg=deg)
                        except np.linalg.LinAlgError:
                            # 奇异情况回退到线性
                            coef = np.polyfit(yv, xv, deg=1)
                        return coef

                    def curve_residual(coef, yv, xv):
                        # 点到曲线的水平残差 |x - f(y)|
                        fx = np.polyval(coef, yv)
                        return np.abs(xv - fx)

                    max_iter = 12
                    for _ in range(max_iter):
                        # 防止某一簇为空 -> 立即回退到中位数切分
                        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
                            labels = (x >= midx).astype(np.int32)
                            break
                        # 拟合两条平滑曲线（三次多项式）
                        coef0 = fit_poly(y[labels == 0], x[labels == 0], deg=2)
                        coef1 = fit_poly(y[labels == 1], x[labels == 1], deg=2)
                        # 依据残差重新分配
                        r0 = curve_residual(coef0, y, x)
                        r1 = curve_residual(coef1, y, x)
                        new_labels = (r1 < r0).astype(np.int32)  # 更靠近曲线1则划为1
                        if np.array_equal(new_labels, labels):
                            break
                        labels = new_labels

                    # 迭代结束后，确定左右：比较两条曲线在共有 y 段上的平均 x
                    if (labels == 0).sum() > 0 and (labels == 1).sum() > 0:
                        coef0 = fit_poly(y[labels == 0], x[labels == 0], deg=3)
                        coef1 = fit_poly(y[labels == 1], x[labels == 1], deg=3)
                        y_min = max(float(y[labels == 0].min()), float(y[labels == 1].min()))
                        y_max = min(float(y[labels == 0].max()), float(y[labels == 1].max()))
                        if y_max > y_min:
                            y_s = np.linspace(y_min, y_max, 100)
                            mean_x0 = float(np.polyval(coef0, y_s).mean())
                            mean_x1 = float(np.polyval(coef1, y_s).mean())
                            # 平均 x 更小的为左
                            if mean_x0 <= mean_x1:
                                left_mask_np = (labels == 0)
                                right_mask_np = (labels == 1)
                            else:
                                left_mask_np = (labels == 1)
                                right_mask_np = (labels == 0)
                        else:
                            # y 段无重叠，回退到全局平均 x 判定
                            mean_x0 = float(x[labels == 0].mean())
                            mean_x1 = float(x[labels == 1].mean())
                            if mean_x0 <= mean_x1:
                                left_mask_np = (labels == 0)
                                right_mask_np = (labels == 1)
                            else:
                                left_mask_np = (labels == 1)
                                right_mask_np = (labels == 0)
                    else:
                        # 极端情况回退
                        midx = np.median(x)
                        left_mask_np = x < midx
                        right_mask_np = ~left_mask_np

                # 应用掩码得到两侧检测框序列（与原键对齐）
                left_b = bones[left_mask_np]
                right_b = bones[right_mask_np]
                # 强制判断：当类别为0的实例只有1个时，强制归为left列
                class_0_mask = (bones[:, 5] == 0)
                class_0_count = class_0_mask.sum().item()
                if class_0_count == 1:
                    # 找到类别为0的检测框
                    class_0_indices = torch.where(class_0_mask)[0]
                    class_0_box = bones[class_0_indices]
                    # 强制将其添加到left_b，从right_b中移除（如果存在）
                    left_b = torch.cat([left_b, class_0_box], dim=0)
                    # 从right_b中移除类别为0的框（如果存在）
                    if right_b.size(0) > 0:
                        right_class_0_mask = (right_b[:, 5] == 0)
                        if right_class_0_mask.any():
                            right_b = right_b[~right_class_0_mask]
            left_b = simple_nms_postprocess(left_b, iou_threshold=iou_threshold)
            right_b = simple_nms_postprocess(right_b, iou_threshold=iou_threshold)
            # 对左右序列分别进行漏检检测和补充
            if raw_det_b is not None:
                if left_b.size(0) > 0:
                    left_b = detect_and_fill_missing_detections(
                        left_b, raw_det_b, (0, 24), gap_threshold_multiplier
                    )
                
                if right_b.size(0) > 0:
                    right_b = detect_and_fill_missing_detections(
                        right_b, raw_det_b, (0, 24), gap_threshold_multiplier
                    )

        left_list.append(left_b)
        right_list.append(right_b)

    max_disc = max((d.size(0) for d in disc_list), default=0)
    max_left = max((d.size(0) for d in left_list), default=0)
    max_right = max((d.size(0) for d in right_list), default=0)

    seq_disc, len_disc = pad_sequence_list(detection, disc_list, max_disc)
    seq_left, len_left = pad_sequence_list(detection, left_list, max_left)
    seq_right, len_right = pad_sequence_list(detection, right_list, max_right)

    return {
        'seq_disc': seq_disc,                  # [B, Ld, 6]
        'seq_vertebra_left': seq_left,         # [B, Ll, 6]
        'seq_vertebra_right': seq_right,       # [B, Lr, 6]
        'seq_lengths': {
            'disc': len_disc,
            'left': len_left,
            'right': len_right
        }
    }

def detect_and_fill_missing_detections(sorted_sequence, raw_detections, class_range, gap_threshold_multiplier=1.5):
    """
    检测并补充漏检的检测框
    
    Args:
        sorted_sequence: 已排序的检测序列 [N, 6] (x1, y1, x2, y2, score, class_id)
        raw_detections: NMS前的原始检测结果 [M, 6]
        class_range: 类别范围 (min_class, max_class)
        gap_threshold_multiplier: 间隔阈值倍数，默认1.5
    
    Returns:
        补充后的检测序列
    """
    if sorted_sequence.size(0) <= 1:
        return sorted_sequence

    # 首先按y_center进行排序
    y_centers = (sorted_sequence[:, 1] + sorted_sequence[:, 3]) * 0.5
    sort_indices = torch.argsort(y_centers)
    sorted_sequence = sorted_sequence[sort_indices]
    
    # 计算相邻检测框之间的y差值
    y_centers = (sorted_sequence[:, 1] + sorted_sequence[:, 3]) * 0.5
    y_diffs = y_centers[1:] - y_centers[:-1]
    
    if y_diffs.size(0) == 0:
        return sorted_sequence
    
    # 计算正常间隔（使用中位数作为基准）
    normal_gap = torch.median(y_diffs)
    gap_threshold = normal_gap * gap_threshold_multiplier
    
    # 找出异常间隔的位置
    large_gaps = y_diffs > gap_threshold
    gap_indices = torch.where(large_gaps)[0]
    
    if gap_indices.size(0) == 0:
        return sorted_sequence
    
    # 从原始检测结果中筛选相同类别的框
    min_class, max_class = class_range
    class_mask = (raw_detections[:, 5] >= min_class) & (raw_detections[:, 5] <= max_class)
    candidate_detections = raw_detections[class_mask]
    
    if candidate_detections.size(0) == 0:
        return sorted_sequence
    
    # 为每个异常间隔寻找可能的漏检框
    filled_sequence = sorted_sequence.clone()
    
    for gap_idx in gap_indices:
        # 获取间隔前后的检测框
        prev_box = sorted_sequence[gap_idx]
        next_box = sorted_sequence[gap_idx + 1]
        
        # 计算间隔的y范围
        prev_y_center = (prev_box[1] + prev_box[3]) * 0.5
        next_y_center = (next_box[1] + next_box[3]) * 0.5
        prev_x_center = (prev_box[0] + prev_box[2]) * 0.5
        next_x_center = (next_box[0] + next_box[2]) * 0.5

        # 计算实际间隔大小
        actual_gap = next_y_center - prev_y_center
        
        # 估算应该补充的检测框数量
        expected_count = torch.round((actual_gap / normal_gap) / gap_threshold_multiplier).int().item()
        expected_count = max(1, expected_count)  # 至少补充1个

        # 将大间隔分成多个小范围
        y_step = actual_gap / (expected_count + 1)  # 分成expected_count+1段
        selected_candidates = []

        for i in range(expected_count):
            # 计算当前小范围的y中心
            current_y_center = prev_y_center + y_step * (i + 1)
            
            # 计算当前小范围的x中心（在前后两个框的x中心之间插值）
            current_x_center = prev_x_center + (next_x_center - prev_x_center) * (i + 1) / (expected_count + 1)
            
            # 定义当前小范围的搜索区域
            candidate_center = (current_x_center, current_y_center)
            x_min = candidate_center[0] - normal_gap * 0.1
            x_max = candidate_center[0] + normal_gap * 0.1
            y_min = candidate_center[1] - normal_gap * 0.1
            y_max = candidate_center[1] + normal_gap * 0.1
            
            # 在当前小范围内寻找候选框
            candidate_y_centers = (candidate_detections[:, 1] + candidate_detections[:, 3]) * 0.5
            candidate_x_centers = (candidate_detections[:, 0] + candidate_detections[:, 2]) * 0.5
            in_gap_mask = ((candidate_y_centers > y_min) & (candidate_y_centers < y_max) & 
                          (candidate_x_centers > x_min) & (candidate_x_centers < x_max))
            
            if in_gap_mask.any():
                gap_candidates = candidate_detections[in_gap_mask]
                
                # 选择置信度最高的候选框
                best_candidate_idx = torch.argmax(gap_candidates[:, 4])
                best_candidate = gap_candidates[best_candidate_idx:best_candidate_idx+1]
                selected_candidates.append(best_candidate)

            
        # 如果有选中的候选框，按y坐标排序并插入
        if selected_candidates:
            selected_candidates = torch.cat(selected_candidates, dim=0)
            selected_y_centers = (selected_candidates[:, 1] + selected_candidates[:, 3]) * 0.5
            selected_sort_indices = torch.argsort(selected_y_centers)
            selected_candidates = selected_candidates[selected_sort_indices]
            
            # 将选中的候选框插入到序列中
            filled_sequence = torch.cat([
                filled_sequence[:gap_idx+1],
                selected_candidates,
                filled_sequence[gap_idx+1:]
            ], dim=0)
    
    # 重新排序
    y_centers_filled = (filled_sequence[:, 1] + filled_sequence[:, 3]) * 0.5
    sort_indices = torch.argsort(y_centers_filled)
    filled_sequence = filled_sequence[sort_indices]
    
    return filled_sequence

def merge_processed_detections(sequence_results, original_detection):
    """
    将处理后的序列合并回原始的detection格式，包括49、50类
    
    Args:
        sequence_results: 拆分后的序列结果
        original_detection: 原始检测结果 [B, N, 6]
    
    Returns:
        更新后的detection [B, M, 6]
    """
    B = original_detection.size(0)
    updated_detections = []
    
    for b in range(B):
        # 收集当前batch的所有处理后的检测框
        batch_boxes = []
        
        # 添加椎间盘序列
        disc_seq = sequence_results['seq_disc'][b]
        if disc_seq.size(0) > 0:
            # 过滤掉padding的零行
            valid_disc = disc_seq[disc_seq[:, 4] > 0]
            if valid_disc.size(0) > 0:
                batch_boxes.append(valid_disc)
        
        # 添加椎骨左侧序列
        left_seq = sequence_results['seq_vertebra_left'][b]
        if left_seq.size(0) > 0:
            valid_left = left_seq[left_seq[:, 4] > 0]
            if valid_left.size(0) > 0:
                batch_boxes.append(valid_left)
        
        # 添加椎骨右侧序列
        right_seq = sequence_results['seq_vertebra_right'][b]
        if right_seq.size(0) > 0:
            valid_right = right_seq[right_seq[:, 4] > 0]
            if valid_right.size(0) > 0:
                batch_boxes.append(valid_right)
        
        # 从原始detection中提取49、50类（这些类没有参与拆分）
        original_batch = original_detection[b]
        if original_batch.size(0) > 0:
            # 过滤掉padding的零行
            valid_original = original_batch[original_batch[:, 4] > 0]
            if valid_original.size(0) > 0:
                # 提取49、50类
                class_49 = valid_original[(valid_original[:, 5] == 49)]
                class_50 = valid_original[(valid_original[:, 5] == 50)]
                class_49 = simple_nms_postprocess(class_49, iou_threshold=0.3)
                class_50 = simple_nms_postprocess(class_50, iou_threshold=0.3)
                special_classes = torch.cat([class_49, class_50], dim=0)
                if special_classes.size(0) > 0:
                    batch_boxes.append(special_classes)
        
        # 合并所有检测框
        if batch_boxes:
            merged_detection = torch.cat(batch_boxes, dim=0)
            # 按y坐标排序
            y_centers = (merged_detection[:, 1] + merged_detection[:, 3]) * 0.5
            sort_indices = torch.argsort(y_centers)
            merged_detection = merged_detection[sort_indices]
        else:
            merged_detection = original_detection.new_zeros((0, 6))
        
        updated_detections.append(merged_detection)
    
    # 计算最大长度并padding
    max_len = max((d.size(0) for d in updated_detections), default=0)
    if max_len == 0:
        return original_detection.new_zeros((B, 0, 6))
    
    # 创建新的detection张量
    new_detection = original_detection.new_zeros((B, max_len, 6))
    for b, det_b in enumerate(updated_detections):
        if det_b.size(0) > 0:
            new_detection[b, :det_b.size(0)] = det_b
    
    return new_detection