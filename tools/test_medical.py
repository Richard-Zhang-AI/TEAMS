import torch.utils.data as data
import glob
import os
import time
import cv2
import numpy as np
from lib.utils.snake import snake_config
from lib.utils import data_utils
from lib.config import cfg
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
import sys
import torch.nn.functional as F
import re
from tools.crf.extract_detection import split_detection_sequences, merge_processed_detections
from tools.crf.functions import _seq_to_numpy, _normalize_and_pad, _append_seq_rows
import csv

# 230
# TRAIN_IDS = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 97, 98, 99, 100, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 133, 135, 136, 137, 138, 139, 140, 141, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 163, 166, 168, 169, 170, 171, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 200, 202, 203, 204, 208, 210, 212, 215, 216, 217, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228]
# 1232
# TRAIN_IDS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 85, 86, 88, 89, 92, 93, 94, 95, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 167, 168, 169, 170, 172, 174, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 232, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 250, 251, 252, 253, 254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 276, 277, 278, 279, 280, 281, 282, 285, 286, 288, 290, 291, 292, 293, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 329, 330, 331, 332, 333, 334, 335, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 363, 364, 365, 366, 367, 369, 370, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418, 421, 422, 423, 424, 425, 426, 427, 429, 430, 431, 432, 433, 436, 437, 439, 440, 441, 445, 447, 448, 449, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 471, 472, 473, 474, 475, 477, 478, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 497, 498, 499, 500, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 516, 517, 518, 519, 520, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 547, 548, 550, 551, 553, 554, 555, 556, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 571, 572, 573, 574, 575, 576, 577, 578, 579, 581, 582, 583, 584, 585, 586, 587, 589, 590, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 614, 615, 616, 617, 618, 620, 621, 622, 623, 624, 625, 626, 627, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 652, 653, 654, 655, 656, 658, 659, 660, 661, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 689, 691, 692, 693, 694, 695, 696, 697, 698, 700, 701, 702, 703, 704, 705, 706, 708, 709, 710, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 728, 729, 730, 731, 732, 733, 734, 735, 736, 738, 739, 741, 742, 743, 745, 746, 748, 751, 752, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 767, 770, 771, 772, 773, 774, 775, 777, 779, 780, 781, 782, 784, 785, 786, 787, 788, 789, 792, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 805, 806, 807, 808, 809, 811, 812, 813, 815, 816, 817, 818, 819, 820, 821, 822, 823, 825, 827, 828, 829, 830, 831, 832, 833, 834, 836, 837, 839, 840, 841, 842, 843, 844, 846, 847, 848, 849, 851, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 864, 866, 867, 869, 871, 872, 874, 876, 877, 878, 879, 882, 883, 884, 885, 888, 889, 890, 892, 893, 894, 895, 896, 899, 900, 901, 902, 903, 904, 905, 906, 908, 909, 910, 911, 912, 913, 914, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 933, 934, 935, 936, 937, 938, 940, 941, 942, 944, 945, 946, 947, 948, 949, 950, 953, 954, 956, 957, 958, 959, 961, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 980, 981, 982, 983, 984, 985, 986, 987, 988, 990, 991, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1015, 1016, 1018, 1019, 1020, 1021, 1022, 1023, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1033, 1035, 1036, 1037, 1039, 1040, 1041, 1042, 1043, 1046, 1047, 1048, 1049, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1063, 1065, 1067, 1068, 1069, 1070, 1071, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1097, 1098, 1101, 1102, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1133, 1134, 1135, 1138, 1139, 1140, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1170, 1171, 1172, 1173, 1174, 1176, 1177, 1178, 1179, 1180, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232]


def nms_for_class(detection, poly, class_id, iou_threshold=0.5):
    """
    对特定类别的检测结果进行NMS过滤
    Args:
        detection: 检测结果 [N, 6] (x1, y1, x2, y2, score, class_id)
        poly: 对应的多边形坐标 [N, num_points, 2]
        class_id: 要过滤的类别ID
        iou_threshold: NMS的IoU阈值
    Returns:
        过滤后的detection和poly
    """
    # 找到该类别的所有检测结果
    class_mask = detection[:, 5] == class_id
    if not class_mask.any():
        return detection, poly
    
    class_detections = detection[class_mask]
    class_poly = poly[class_mask] if len(poly.shape) == 3 else [poly[i] for i in range(len(poly)) if class_mask[i]]
    
    # 如果该类别只有一个检测结果，直接返回
    if class_detections.shape[0] <= 1:
        return detection, poly
    
    # 按置信度排序
    scores = class_detections[:, 4]
    sorted_indices = torch.argsort(scores, descending=True)
    
    # NMS过滤
    keep_indices = []
    for i in sorted_indices:
        keep = True
        for j in keep_indices:
            # 计算IoU
            box1 = class_detections[i, :4]
            box2 = class_detections[j, :4]
            iou = calculate_box_iou(box1, box2)
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            keep_indices.append(i)
    
    # 构建保留的索引
    original_indices = torch.where(class_mask)[0]
    keep_original_indices = [original_indices[i] for i in keep_indices]
    
    # 创建新的detection和poly
    all_indices = []
    for i in range(detection.shape[0]):
        if not class_mask[i] or i in keep_original_indices:
            all_indices.append(i)
    
    filtered_detection = detection[all_indices]
    if len(poly.shape) == 3:
        filtered_poly = poly[all_indices]
    else:
        filtered_poly = [poly[i] for i in all_indices]
    
    return filtered_detection, filtered_poly


def calculate_box_iou(box1, box2):
    """计算两个检测框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 如果没有交集
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    
    # 计算交集面积
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def apply_nms_to_detections(detection, poly, target_classes=[50, 51], iou_threshold=0.5):
    """
    对指定类别应用NMS过滤
    Args:
        detection: 检测结果
        poly: 多边形坐标 (通常是list或tuple结构)
        target_classes: 需要进行NMS的类别列表
        iou_threshold: NMS的IoU阈值
    Returns:
        过滤后的detection和poly
    """
    if detection.shape[0] == 0:
        return detection, poly
    
    # 简化的NMS实现，专门针对指定类别
    keep_indices = []
    
    for class_id in target_classes:
        class_mask = detection[:, 5] == class_id
        if not class_mask.any():
            continue
            
        class_indices = torch.where(class_mask)[0]
        if len(class_indices) <= 1:
            keep_indices.extend(class_indices.tolist())
            continue
        
        # 按置信度排序
        class_scores = detection[class_indices, 4]
        sorted_indices = torch.argsort(class_scores, descending=True)
        
        # 只保留置信度最高的那个
        best_idx = class_indices[sorted_indices[0]]
        keep_indices.append(best_idx.item())
        
        print(f"类别 {class_id}: 从 {len(class_indices)} 个检测结果中保留置信度最高的1个")
    
    # 添加其他类别的所有检测结果
    for i in range(detection.shape[0]):
        if detection[i, 5] not in target_classes:
            keep_indices.append(i)
    
    # 过滤detection
    keep_indices = sorted(keep_indices)
    filtered_detection = detection[keep_indices]
    
    # 过滤poly - 根据poly的结构进行处理
    if isinstance(poly, (list, tuple)) and len(poly) > 2:
        # poly的结构通常是 [layer1, layer2, target_layer]
        filtered_poly = []
        for layer in poly:
            if hasattr(layer, 'shape') and len(layer.shape) >= 2:
                # 使用tensor索引
                keep_tensor = torch.tensor(keep_indices)
                filtered_poly.append(layer[keep_tensor])
            else:
                filtered_poly.append(layer)
        # 保持原始结构
        filtered_poly = type(poly)(filtered_poly)
    else:
        # 如果poly是tensor
        if hasattr(poly, 'shape'):
            keep_tensor = torch.tensor(keep_indices)
            filtered_poly = poly[keep_tensor]
        else:
            filtered_poly = poly
    
    return filtered_detection, filtered_poly

def mask_to_component_boxes(mask, min_area=10):
    """把单个二值掩码拆分成多个连通区域，并为每个区域生成一个xyxy框"""
    mask_u8 = (mask > 0.5).astype(np.uint8)
    result = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[1] if len(result) == 3 else result[0]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append([x, y, x + w - 1, y + h - 1])
    return boxes

def mask_to_instance_masks(mask, min_area=10):
    """将单个掩码拆分为多个实例掩码"""
    mask_u8 = (mask > 0.5).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8)
    instances = []
    for label_idx in range(1, num_labels):
        inst_mask = (labels == label_idx).astype(np.uint8)
        if inst_mask.sum() >= min_area:
            instances.append(inst_mask)
    return instances

def masks_to_instance_boxes(gt_masks, gt_classes, min_area=10):
    """
    将一组GT掩码展开为“实例级”的 bbox 与标签（每个连通组件作为一个实例）
    返回：
      - boxes: List[[x1,y1,x2,y2], ...]
      - labels: List[int]（与boxes一一对应）
    """
    boxes, labels = [], []
    for m, cls in zip(gt_masks, gt_classes):
        comps = mask_to_component_boxes(m, min_area=min_area)
        for b in comps:
            boxes.append(b)
            labels.append(cls)
    return boxes, labels

def align_classes_iou(detection_tensor, gt_masks, gt_classes, iou_thresh=0.5):
    """
    基于IoU的一对一匹配，对齐类别到GT。
    - detection_tensor: [1, K, 6]，列为 x1,y1,x2,y2,score,cls(0-based)
    - gt_masks: List[np.ndarray(H,W)]，二值掩码
    - gt_classes: List[int]，与掩码一一对应；注意通常是1..C（你的class_list[0]占位）
    结果：原地将 detection_tensor[0, :, 5] 更新为匹配到的 GT 类别 - 1
    """
    if detection_tensor.shape[1] == 0 or len(gt_masks) == 0:
        return 0

    det = detection_tensor[0].detach().cpu().numpy()
    # 仅保留有效框索引（score>0）
    valid_idx = np.where(det[:, 4] > -1)[0]
    if valid_idx.size == 0:
        return 0

    det_boxes = det[:, :4].astype(np.float32)
    # 从GT掩码得到GT框
    gt_boxes_list, gt_labels_list = masks_to_instance_boxes(gt_masks, gt_classes, min_area=10)
    if len(gt_boxes_list) == 0:
        return 0
    gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
    gt_labels = np.array(gt_labels_list, dtype=np.int32)

    # 计算IoU矩阵 [N,K] -> 这里是 [Nd, Ng]
    ious = np.zeros((det_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)
    for i in range(det_boxes.shape[0]):
        for j in range(gt_boxes.shape[0]):
            ious[i, j] = calculate_box_iou(det_boxes[i], gt_boxes[j])

    # 贪心一对一匹配（降序）
    pairs = []
    idxs = np.argwhere(ious >= iou_thresh)
    if idxs.size == 0:
        return 0
    vals = ious[idxs[:, 0], idxs[:, 1]]
    order = np.argsort(-vals)
    used_det = set()
    used_gt = set()
    for k in order:
        di, gi = idxs[k]
        if di in used_det or gi in used_gt:
            continue
        pairs.append((di, gi))
        used_det.add(di)
        used_gt.add(gi)

    # 回写类别（检测类0-based，GT通常是1-based）
    changed = 0
    for di, gi in pairs:
        new_cls_0based = int(gt_labels[gi]) - 1
        det_row = valid_idx[di]
        old_cls = int(detection_tensor[0, det_row, 5].item())
        if new_cls_0based != old_cls:
            detection_tensor[0, det_row, 5] = detection_tensor[0, det_row, 5].new_tensor(new_cls_0based)
            changed += 1
    return changed

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.imgs = []
        list_file = os.path.join(cfg.test.img_path, 'test_list.txt')
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        for name in lines:
            img_png = os.path.join(cfg.test.img_path, name)
            if not os.path.exists(img_png):
                raise FileNotFoundError(f"Image not found: {img_png}")
            
            self.imgs.append(img_png)

        print('======================')
        print('训练数据路径：', cfg.test.img_path)
        print('样本数（来自 test_list.txt）:', len(self.imgs))
        print('列表文件：', list_file)
        print('======================')

        # # 优先使用 cfg.test.img_path，否则回退到 cfg.train.data_path，保证与训练集一致
        # img_root = None
        # if hasattr(cfg, 'test') and getattr(cfg.test, 'img_path', '') and os.path.isdir(cfg.test.img_path):
        #     img_root = cfg.test.img_path
        # elif os.path.isdir(cfg.train.data_path):
        #     img_root = cfg.train.data_path

        # if img_root is not None:
        #     # 读取与训练相同命名规则的图片：*_image.png
        #     self.imgs = sorted(glob.glob(os.path.join(img_root, '*_image.png')))
        #     if len(self.imgs) == 0:
        #         # 兼容 jpg 命名
        #         self.imgs = sorted(glob.glob(os.path.join(img_root, '*_image.jpg')))
        # elif hasattr(cfg, 'test') and os.path.exists(cfg.test.img_path):
        #     # 单张测试
        #     self.imgs = [cfg.test.img_path]
        # else:
        #     raise Exception("测试图片文件夹不存在或为空！请检查 cfg.test.img_path 或 cfg.train.data_path。")

        # # # 仅保留测试集
        # if isinstance(TRAIN_IDS, (list, tuple)) and len(TRAIN_IDS) > 0 and len(self.imgs) > 1:
        #     id_to_path = {}
        #     all_ids = []
        #     for p in self.imgs:
        #         base = os.path.basename(p)
        #         m = re.match(r'^(\d+)_image\.(png|jpg)$', base, flags=re.IGNORECASE)
        #         if m:
        #             idx = int(m.group(1))
        #             id_to_path[idx] = p
        #             all_ids.append(idx)
        #     if len(all_ids) > 0:
        #         train_set = set(int(x) for x in TRAIN_IDS)
        #         test_ids = sorted(set(all_ids) - train_set)
        #         self.imgs = [id_to_path[i] for i in test_ids if i in id_to_path]

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img = self.imgs[index]
        img = cv2.imread(img)
        inp = img

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = ((width + x - 1) // x) * x
        input_h = ((height + x - 1) // x) * x

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = self.normalize_image(inp)  # inp = { ndarray:(3,544, 544) }

        # 将(3, 544, 544)的图片截取为（3,512,512）
        # =====================================
        # 计算中心裁剪的起始点
        # start_x = (544 - 512) // 2
        # start_y = (544 - 512) // 2
        #
        # # 计算裁剪区域的结束点
        # end_x = start_x + 512
        # end_y = start_y + 512
        #
        # # 使用切片操作裁剪图片
        # inp = inp[:, start_x:end_x, start_y:end_y]
        # # =====================================

        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'vis_GT': '', 'ann': ''}
        ret.update({'meta': meta})

        return ret, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


def poly2mask(ex):
    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy() * 4  # 修改为3.75，与可视化保持一致

    img = np.zeros((512, 512))
    ex = np.array(ex)
    ex = ex.astype(np.int32)
    for i in range(ex.shape[0]):
        # img = cv2.polylines(img,[ex[i]],True,1,1)
        # 移除cv2.polylines调用，只保留fillPoly
        img = cv2.fillPoly(img, [ex[i]], 1)
    return img


def cal_iou(mask, gtmask):
    """计算IoU，不考虑类别"""
    jiaoji = mask * gtmask
    bingji = ((mask + gtmask) != 0).astype(np.int16)
    return jiaoji.sum() / bingji.sum() if bingji.sum() > 0 else 0.0


# def cal_dice(mask, gtmask):
    # """直接计算Dice系数，而不是基于IoU"""
    # intersection = (mask * gtmask).sum()
    # union = mask.sum() + gtmask.sum()
    # return 2.0 * intersection / union if union > 0 else 0.0

def cal_dice(iou):
    # 直接基于iou计算dice
    return 2 * iou / (iou + 1)


def cal_class_wise_metrics(pred_masks, gt_masks, pred_classes, gt_classes):
    """
    计算类别相关的mIoU、mDice、HD和HD95
    公式：
    mIoU = (1/N) * Σ(|A_i ∩ B_i|/|A_i ∪ B_i|)
    mDice = (1/N) * Σ(2|A_i ∩ B_i|/(|A_i| + |B_i|))
    HD = Hausdorff距离（边缘贴合情况）
    HD95 = 95% Hausdorff距离（更稳定的边缘指标）
    其中A_i是第i类的真实分割，B_i是第i类的预测分割，N是器官总数
    
    Args:
        pred_masks: 预测掩码列表，每个元素是一个二值掩码
        gt_masks: 真实掩码列表，每个元素是一个二值掩码
        pred_classes: 预测类别列表
        gt_classes: 真实类别列表
    Returns:
        class_wise_ious: 各类别IoU字典
        class_wise_dices: 各类别Dice字典
        class_wise_hds: 各类别Hausdorff距离字典
        class_wise_hd95s: 各类别HD95字典
        mean_iou: 类别相关mIoU
        mean_dice: 类别相关mDice
        mean_hd: 类别相关平均HD
        mean_hd95: 类别相关平均HD95
    """
    class_wise_ious = {}
    class_wise_dices = {}
    class_wise_hds = {}
    class_wise_hd95s = {}
    
    
    
    # for i, class_id in enumerate(gt_classes):
    #     try:
    #         index = pred_classes.index(class_id)
    #         iou = cal_iou(pred_masks[index], gt_masks[i])
    #         dice = cal_dice(iou)    # 基于iou计算dice
    #     except ValueError:
    #         iou = 0
    #         dice = 0
    #     class_wise_ious[class_id] = iou
    #     class_wise_dices[class_id] = dice
        
    # 获取所有出现的类别
    all_classes = set(pred_classes + gt_classes)
    for class_id in all_classes:
        # 获取该类别的预测掩码
        pred_mask_class = np.zeros_like(pred_masks[0])
        gt_mask_class = np.zeros_like(gt_masks[0])
        
        for i, pred_cls in enumerate(pred_classes):
            if pred_cls == class_id:
                pred_mask_class = np.maximum(pred_mask_class, pred_masks[i])
        
        for i, gt_cls in enumerate(gt_classes):
            if gt_cls == class_id:
                gt_mask_class = np.maximum(gt_mask_class, gt_masks[i])
        
        # 计算该类别的IoU、Dice、HD和HD95
        if gt_mask_class.sum() > 0:  # 只计算GT中存在的类别
            iou = cal_iou(pred_mask_class, gt_mask_class)
            # dice = cal_dice(pred_mask_class, gt_mask_class)
            dice = cal_dice(iou)    # 基于iou计算dice
            
            # 计算Hausdorff距离指标
            hd = cal_hausdorff_distance(pred_mask_class, gt_mask_class)
            hd95 = cal_hausdorff_distance_95(pred_mask_class, gt_mask_class)
            
            # 处理无穷大值（当没有轮廓时返回无穷大）
            hd = 0.0 if hd == float('inf') else hd
            hd95 = 0.0 if hd95 == float('inf') else hd95
            
            class_wise_ious[class_id] = iou
            class_wise_dices[class_id] = dice
            class_wise_hds[class_id] = hd
            class_wise_hd95s[class_id] = hd95
    
    # 计算平均值
    mean_iou = np.mean(list(class_wise_ious.values())) if class_wise_ious else 0.0
    mean_dice = np.mean(list(class_wise_dices.values())) if class_wise_dices else 0.0
    mean_hd = np.mean(list(class_wise_hds.values())) if class_wise_hds else 0.0
    mean_hd95 = np.mean(list(class_wise_hd95s.values())) if class_wise_hd95s else 0.0
    
    return class_wise_ious, class_wise_dices, class_wise_hds, class_wise_hd95s, mean_iou, mean_dice, mean_hd, mean_hd95


def extract_contour(mask, tolerance=1):
    """提取掩码的边界轮廓，支持容忍度"""
    # 确保mask是二值的
    mask = (mask > 0.5).astype(np.uint8)
    
    # 提取轮廓 - 兼容不同版本的OpenCV
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(result) == 3:  # OpenCV 3.x 返回 (image, contours, hierarchy)
        _, contours, _ = result
    else:  # OpenCV 4.x 返回 (contours, hierarchy)
        contours, _ = result
    
    # 创建轮廓掩码
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, 1, thickness=tolerance)
    
    return contour_mask.astype(np.float32)


def cal_boundary_dice(pred_mask, gt_mask, tolerance=1):
    """计算边界Dice系数"""
    pred_contour = extract_contour(pred_mask, tolerance)
    gt_contour = extract_contour(gt_mask, tolerance)
    
    intersection = (pred_contour * gt_contour).sum()
    union = pred_contour.sum() + gt_contour.sum()
    
    return 2.0 * intersection / union if union > 0 else 0.0


def cal_mBoundF(pred_masks, gt_masks, pred_classes, gt_classes):
    """
    计算类别相关的mBoundF指标（边界质量评估）- 修复版本
    使用实例级别的匹配，而不是类别级别的合并
    
    Args:
        pred_masks: 预测掩码列表
        gt_masks: 真实掩码列表  
        pred_classes: 预测类别列表
        gt_classes: 真实类别列表
    Returns:
        mBoundF: 平均边界F-score
    """
    # tolerances = [1, 2, 3, 4, 5]
    tolerances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 增加到10像素
    boundary_dices_per_tolerance = []
    
    for tolerance in tolerances:
        # 使用实例级别的匹配，而不是类别级别的合并
        instance_boundary_dices = []
        
        # 只考虑GT中存在的类别的实例
        for i, gt_cls in enumerate(gt_classes):
            gt_mask = gt_masks[i]
            if gt_mask.sum() == 0:
                continue
                
            # 找到同类别的最佳匹配预测
            best_boundary_dice = 0.0
            for j, pred_cls in enumerate(pred_classes):
                if pred_cls == gt_cls:
                    pred_mask = pred_masks[j]
                    if pred_mask.sum() > 0:
                        boundary_dice = cal_boundary_dice(pred_mask, gt_mask, tolerance)
                        best_boundary_dice = max(best_boundary_dice, boundary_dice)
            
            # 将最佳匹配的边界Dice添加到列表
            instance_boundary_dices.append(best_boundary_dice)
        
        # 计算该容忍度下的平均边界Dice
        if instance_boundary_dices:
            avg_boundary_dice = np.mean(instance_boundary_dices)
            boundary_dices_per_tolerance.append(avg_boundary_dice)
    
    # 计算所有容忍度的平均值
    mBoundF = np.mean(boundary_dices_per_tolerance) if boundary_dices_per_tolerance else 0.0
    
    return mBoundF


def cal_mBoundF_agnostic(pred_mask, gt_mask):
    """
    计算类别无关的mBoundF指标（边界质量评估）
    公式：mBoundF = (1/5) * Σ(mDice_n(∂A_n, ∂B_n)) for n=1 to 5
    其中∂A_n和∂B_n分别表示真实和预测掩码的轮廓，n表示轮廓容忍度（1到5像素）
    
    Args:
        pred_mask: 预测掩码（整体合并后的）
        gt_mask: 真实掩码（整体合并后的）
    Returns:
        mBoundF_agnostic: 类别无关的平均边界F-score
    """
    # tolerances = [1, 2, 3, 4, 5]
    tolerances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 增加到10像素
    boundary_dices = []
    
    for tolerance in tolerances:
        boundary_dice = cal_boundary_dice(pred_mask, gt_mask, tolerance)
        boundary_dices.append(boundary_dice)
    
    # 计算所有容忍度的平均值
    mBoundF_agnostic = np.mean(boundary_dices) if boundary_dices else 0.0
    
    return mBoundF_agnostic

def compute_pq(pred_instance_masks, pred_instance_classes, gt_instance_masks, gt_instance_classes, iou_thresh=0.5):
    """
    计算Panoptic Quality (PQ) 指标
    Returns:
        pq_per_class: {class_id: {'pq':..., 'sq':..., 'rq':..., 'tp':..., 'fp':..., 'fn':...}}
        pq_mean: 所有类别PQ的平均值
    """
    if len(pred_instance_masks) == 0 and len(gt_instance_masks) == 0:
        return {}, 0.0

    class_ids = sorted(set(gt_instance_classes + pred_instance_classes))
    pq_per_class = {}

    for class_id in class_ids:
        pred_indices = [i for i, c in enumerate(pred_instance_classes) if c == class_id]
        gt_indices = [i for i, c in enumerate(gt_instance_classes) if c == class_id]

        tp = 0
        fp = len(pred_indices)
        fn = len(gt_indices)
        sum_iou = 0.0

        if pred_indices and gt_indices:
            iou_matrix = np.zeros((len(pred_indices), len(gt_indices)), dtype=np.float32)
            for pi, pred_idx in enumerate(pred_indices):
                for gi, gt_idx in enumerate(gt_indices):
                    iou_matrix[pi, gi] = cal_iou(pred_instance_masks[pred_idx], gt_instance_masks[gt_idx])

            coords = np.argwhere(iou_matrix >= iou_thresh)
            if coords.size > 0:
                vals = iou_matrix[coords[:, 0], coords[:, 1]]
                order = np.argsort(-vals)
                used_pred = set()
                used_gt = set()
                for idx in order:
                    pi, gi = coords[idx]
                    if pi in used_pred or gi in used_gt:
                        continue
                    used_pred.add(pi)
                    used_gt.add(gi)
                    tp += 1
                    sum_iou += iou_matrix[pi, gi]

                fp -= tp
                fn -= tp

        if tp == 0:
            pq = 0.0
            sq = 0.0
            rq = 0.0
        else:
            sq = sum_iou / tp
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            pq = sq * rq

        pq_per_class[class_id] = {'pq': pq, 'sq': sq, 'rq': rq, 'tp': tp, 'fp': max(fp, 0), 'fn': max(fn, 0)}

    pq_mean = np.mean([stats['pq'] for stats in pq_per_class.values()]) if pq_per_class else 0.0
    return pq_per_class, pq_mean


def extract_contour_points(mask):
    """
    从二值掩码中提取轮廓点集
    
    Args:
        mask: 二值掩码 (numpy array, 任意范围)
    Returns:
        contour_points: 轮廓点集的列表，每个元素是一个numpy数组 [[x1,y1], [x2,y2], ...]
    """
    # 确保mask是二值的 - 支持任意范围的掩码
    if mask.max() <= 1.0:
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_binary = (mask > 127).astype(np.uint8)
    
    # 检查是否有有效像素
    if mask_binary.sum() == 0:
        return []
    
    # 提取轮廓
    result = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = result[1] if len(result) == 3 else result[0]
    
    contour_points = []
    for contour in contours:
        # 将轮廓从 (n, 1, 2) 转换为 (n, 2)
        points = contour.reshape(-1, 2)
        if len(points) > 2:  # 至少需要3个点才能形成轮廓
            contour_points.append(points)
    
    return contour_points


def compute_hausdorff_distance(contour_pred, contour_gt):
    """
    计算两个轮廓点集之间的Hausdorff距离
    
    Args:
        contour_pred: 预测轮廓点集 [[x1,y1], [x2,y2], ...]
        contour_gt: 真实轮廓点集 [[x1,y1], [x2,y2], ...]
    Returns:
        hd: Hausdorff距离
    """
    if len(contour_pred) == 0 or len(contour_gt) == 0:
        return float('inf')
    
    # 转换为numpy数组
    pred_points = np.array(contour_pred, dtype=np.float32)
    gt_points = np.array(contour_gt, dtype=np.float32)
    
    # 计算预测轮廓到真实轮廓的距离
    distances_pred_to_gt = []
    for pred_point in pred_points:
        # 计算当前预测点到所有真实点的最小距离
        distances = np.sqrt(np.sum((gt_points - pred_point) ** 2, axis=1))
        distances_pred_to_gt.append(np.min(distances))
    
    # 计算真实轮廓到预测轮廓的距离
    distances_gt_to_pred = []
    for gt_point in gt_points:
        # 计算当前真实点到所有预测点的最小距离
        distances = np.sqrt(np.sum((pred_points - gt_point) ** 2, axis=1))
        distances_gt_to_pred.append(np.min(distances))
    
    # Hausdorff距离是两个方向最大距离的最大值
    max_pred_to_gt = np.max(distances_pred_to_gt) if distances_pred_to_gt else 0
    max_gt_to_pred = np.max(distances_gt_to_pred) if distances_gt_to_pred else 0
    
    hd = max(max_pred_to_gt, max_gt_to_pred)
    
    return hd


def compute_hausdorff_distance_95(contour_pred, contour_gt):
    """
    计算两个轮廓点集之间的95% Hausdorff距离 (HD95)
    剔除5%的离群点，使指标更稳定
    
    Args:
        contour_pred: 预测轮廓点集 [[x1,y1], [x2,y2], ...]
        contour_gt: 真实轮廓点集 [[x1,y1], [x2,y2], ...]
    Returns:
        hd95: 95% Hausdorff距离
    """
    if len(contour_pred) == 0 or len(contour_gt) == 0:
        return float('inf')
    
    # 转换为numpy数组
    pred_points = np.array(contour_pred, dtype=np.float32)
    gt_points = np.array(contour_gt, dtype=np.float32)
    
    # 计算预测轮廓到真实轮廓的距离
    distances_pred_to_gt = []
    for pred_point in pred_points:
        distances = np.sqrt(np.sum((gt_points - pred_point) ** 2, axis=1))
        distances_pred_to_gt.append(np.min(distances))
    
    # 计算真实轮廓到预测轮廓的距离
    distances_gt_to_pred = []
    for gt_point in gt_points:
        distances = np.sqrt(np.sum((pred_points - gt_point) ** 2, axis=1))
        distances_gt_to_pred.append(np.min(distances))
    
    # 合并所有距离
    all_distances = distances_pred_to_gt + distances_gt_to_pred
    
    if len(all_distances) == 0:
        return 0.0
    
    # 取95%分位数，排除5%的离群点
    hd95 = np.percentile(all_distances, 95)
    
    return hd95


def cal_hausdorff_distance(pred_mask, gt_mask):
    """
    计算预测掩码和真实掩码之间的Hausdorff距离
    
    Args:
        pred_mask: 预测掩码 (numpy array, 任意范围)
        gt_mask: 真实掩码 (numpy array, 任意范围)
    Returns:
        hd: Hausdorff距离 (像素单位)
    """
    # 提取轮廓点
    pred_contours = extract_contour_points(pred_mask)
    gt_contours = extract_contour_points(gt_mask)
    
    if not pred_contours or not gt_contours:
        return float('inf')
    
    # 如果有多个轮廓，计算所有组合中的最大Hausdorff距离
    max_hd = 0.0
    for pred_contour in pred_contours:
        for gt_contour in gt_contours:
            hd = compute_hausdorff_distance(pred_contour, gt_contour)
            max_hd = max(max_hd, hd)
    
    return max_hd


def cal_hausdorff_distance_95(pred_mask, gt_mask):
    """
    计算预测掩码和真实掩码之间的95% Hausdorff距离
    
    Args:
        pred_mask: 预测掩码 (numpy array, 任意范围)
        gt_mask: 真实掩码 (numpy array, 任意范围)
    Returns:
        hd95: 95% Hausdorff距离 (像素单位)
    """
    # 提取轮廓点
    pred_contours = extract_contour_points(pred_mask)
    gt_contours = extract_contour_points(gt_mask)
    
    if not pred_contours or not gt_contours:
        return float('inf')
    
    # 如果有多个轮廓，计算所有组合中的最大HD95
    max_hd95 = 0.0
    for pred_contour in pred_contours:
        for gt_contour in gt_contours:
            hd95 = compute_hausdorff_distance_95(pred_contour, gt_contour)
            max_hd95 = max(max_hd95, hd95)
    
    return max_hd95


def _find_latest_epoch(model_dir):
    """从 model_dir 中自动查找最大轮数 epoch 文件名，返回该 epoch 整数；找不到则返回 None"""
    if not os.path.isdir(model_dir):
        return None
    pths = glob.glob(os.path.join(model_dir, '*.pth'))
    latest_ep = None
    pattern = re.compile(r'(\d+)\.pth$')
    for p in pths:
        m = pattern.search(os.path.basename(p))
        if m:
            ep = int(m.group(1))
            latest_ep = ep if (latest_ep is None or ep > latest_ep) else latest_ep
    return latest_ep


# ==============================================================================
# ======================== 新增的可视化函数 ==============================
# ==============================================================================
def draw_dashed_polyline(img, points, color, thickness=2, dash_length=15, gap_length=8):
    """
    绘制虚线多边形 - 简化且可靠的版本
    
    Args:
        img: 图像
        points: 多边形点集，形状为 [N, 2]
        color: 颜色 (B, G, R)
        thickness: 线宽
        dash_length: 虚线段的长度（像素）
        gap_length: 间隔的长度（像素）
    """
    if len(points) < 2:
        return
    
    points = points.astype(np.float32)
    
    # 闭合多边形
    points_loop = np.concatenate([points, points[:1]], axis=0)
    
    # 沿着多边形路径绘制虚线
    is_drawing = True
    accumulated_length = 0.0
    
    for i in range(len(points_loop) - 1):
        pt1 = points_loop[i].copy()
        pt2 = points_loop[i + 1].copy()
        
        # 计算线段长度和方向
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        seg_length = np.sqrt(dx*dx + dy*dy)
        
        if seg_length < 0.1:
            continue
        
        # 单位方向向量
        dx /= seg_length
        dy /= seg_length
        
        # 沿着这条线段绘制虚线
        pos = 0.0
        while pos < seg_length:
            if is_drawing:
                # 绘制虚线
                draw_end = min(pos + (dash_length - accumulated_length), seg_length)
                if draw_end > pos:
                    start_pt = (int(pt1[0] + dx * pos), int(pt1[1] + dy * pos))
                    end_pt = (int(pt1[0] + dx * draw_end), int(pt1[1] + dy * draw_end))
                    cv2.line(img, start_pt, end_pt, color, thickness, lineType=cv2.LINE_AA)
                    
                    accumulated_length += (draw_end - pos)
                    pos = draw_end
                    
                    if accumulated_length >= dash_length:
                        is_drawing = False
                        accumulated_length = 0.0
                else:
                    break
            else:
                # 跳过间隔
                skip_end = min(pos + (gap_length - accumulated_length), seg_length)
                accumulated_length += (skip_end - pos)
                pos = skip_end
                
                if accumulated_length >= gap_length:
                    is_drawing = True
                    accumulated_length = 0.0


def visualize_polygons(pred_polys_tensor, orig_img, detection, save_name, class_colors, class_name, gt_masks=None, gt_classes=None, gt_color=(255, 255, 255), gt_thickness=2):
    """
    将预测多边形可视化并保存，根据检测类别使用不同颜色。
    
    Args:
        pred_polys_tensor (torch.Tensor): 预测的多边形张量，形状为 [N, 128, 2]。
        orig_img: 原始图像
        detection: 检测结果，包含类别信息
        save_name (str): 保存文件名
        class_colors (dict): 类别 id -> (R, G, B) 颜色元组，内部会转为 BGR 供 OpenCV 使用
        class_name (list): 类别名称列表
        gt_masks (list): 金标准掩码列表
        gt_classes (list): 金标准类别列表，与 gt_masks 一一对应
    """
    # 1. 准备工作：将Tensor转为Numpy数组
    pred_polys = pred_polys_tensor.detach().cpu().numpy().astype(np.int32)
    b = detection.shape[0]
    detection_num = detection.shape[1]
    detection_class = detection[:, :, 5]
    
    canvas = orig_img.copy()
    h, w = canvas.shape[:2]
    
    # 用于跟踪已显示的类别文本（避免重复）— 已关闭
    # shown_classes = set()
    
    # 1. 画金标准（实线，使用对应类别颜色）— 已关闭
    # if gt_masks is not None and len(gt_masks) > 0:
    #     for idx, m in enumerate(gt_masks):
    #         if m is None:
    #             continue
    #         
    #         # 获取对应的类别ID
    #         if gt_classes is not None and idx < len(gt_classes):
    #             gt_class_id = gt_classes[idx]
    #         else:
    #             # 如果没有提供类别信息，默认使用类别1
    #             gt_class_id = 1
    #         
    #         # 使用对应类别的颜色
    #         gt_color = class_colors[gt_class_id]
    #         
    #         mask_u8 = (m > 0.5).astype(np.uint8)
    #         result = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         contours = result[1] if len(result) == 3 else result[0]
    #         
    #         if contours:
    #             # 绘制实线轮廓
    #             cv2.drawContours(canvas, contours, -1, gt_color, thickness=gt_thickness)
    #             
    #             # 为每个轮廓计算质心并显示文本（每个类别只显示一次）
    #             for contour in contours:
    #                 if len(contour) < 3:
    #                     continue
    #                 # 计算质心
    #                 M = cv2.moments(contour)
    #                 if M["m00"] != 0:
    #                     cx = int(M["m10"] / M["m00"])
    #                     cy = int(M["m01"] / M["m00"])
    #                     cx = np.clip(cx, 0, w - 1)
    #                     cy = np.clip(cy, 0, h - 1)
    #                     
    #                     # 如果该类别还没有显示过文本，则显示
    #                     if gt_class_id not in shown_classes:
    #                         label = class_name[gt_class_id]
    #                         font = cv2.FONT_HERSHEY_SIMPLEX
    #                         font_scale = 0.5
    #                         thickness_text = 1
    #                         tx = cx + 3
    #                         ty = cy - 3
    #                         # 文本使用白色，以便在彩色背景上清晰可见
    #                         cv2.putText(canvas, label, (tx, ty), font, font_scale, gt_color, thickness_text, lineType=cv2.LINE_AA)
    #                         shown_classes.add(gt_class_id)
                
    # 2. 绘制预测多边形（实线，根据类别使用不同颜色，class_colors 为 RGB，OpenCV 需 BGR）
    for i in range(detection_num):
        if i >= pred_polys.shape[0]:
            print(f"IndexError: index {i} is out of bounds for axis 0 with size pred_polys.shape[0]({pred_polys.shape[0]})")
            break
        
        # 获取当前多边形的类别
        class_id = int(detection_class[0][i]) + 1
        if class_id not in class_colors:
            color_rgb = (255, 255, 255)
        else:
            color_rgb = class_colors[class_id]
        color = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB -> BGR for OpenCV
        
        # 绘制实线多边形
        pts = pred_polys[i].reshape(-1, 2)
        # 确保点在图像范围内
        pts = np.clip(pts, [0, 0], [w-1, h-1])
        pts = pts.astype(np.int32)
        
        # 过滤掉无效点（NaN或Inf）
        valid_mask = np.isfinite(pts).all(axis=1)
        pts = pts[valid_mask]
        
        # 绘制闭合多边形（实线）
        if len(pts) >= 3:
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # 文本位置与绘制 — 已关闭
        # pts = pred_polys[i].reshape(-1, 2)
        # cx = int(np.clip(pts[:,0].mean(), 0, w))
        # cy = int(np.clip(pts[:, 1].mean(), 0, h - 1))
        # label = class_name[class_id]
        # 
        # # 如果该类别还没有显示过文本，则显示
        # if class_id not in shown_classes:
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 0.5
        #     thickness_text = 1
        #     x1 = int(np.clip(cx, 0, w - 1))
        #     y1 = int(np.clip(cy, 0, h - 1))
        #     tx = x1 + 3
        #     ty = y1 - 3
        #     # 文本使用白色，以便在彩色背景上清晰可见
        #     cv2.putText(canvas, label, (tx, ty), font, font_scale, color, thickness_text, lineType=cv2.LINE_AA)
        #     shown_classes.add(class_id)
    
    # 3. 保存
    save_dir = cfg.test.visual_save_root
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, canvas)

def TEST():
    visual = 1
    network = make_network(cfg).cuda()
    save_root = cfg.test.visual_save_root
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    # 指标日志文件
    log_path = os.path.join(cfg.test.visual_save_root, 'metrics_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('test image metrics\n')

    # 自动选择 model_dir 下的最大 epoch 权重进行加载
    latest_epoch = _find_latest_epoch(cfg.model_dir)
    print(f"[WARN] 加载权重 cfg.test.epoch={cfg.test.epoch}")
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    # ===== 为三个序列各建一个CSV文件 =====
    csv_paths = {
        'seq_disc': os.path.join(save_root, 'seq_disc_detections.csv'),
        'seq_vertebra_left': os.path.join(save_root, 'seq_vertebra_left_detections.csv'),
        'seq_vertebra_right': os.path.join(save_root, 'seq_vertebra_right_detections.csv'),
    }
    csv_files = {k: open(p, 'w', newline='', encoding='utf-8') for k, p in csv_paths.items()}
    csv_writers = {k: csv.writer(f) for k, f in csv_files.items()}
    # 写表头
    for w in csv_writers.values():
        w.writerow(['image_id', 'rank', 'x1n', 'y1n', 'x2n', 'y2n', 'class', 'score'])

    dataset = Dataset()
    # 根据sbd_snake.yaml中的class_descriptions构建类别列
    # RAOS
    class_list = [
        "",  "Liver", "Spleen", "kidney(L)", "Kidney(R)", "Stomach",  "Gall", "Esophagus", "Panc", "DUodenum", "Colon", 
        "Intestine", "Adrenal(L)", "Adrenal(R)", "Rectum", "Bladder", "HoF(L)", "HoF(R)", "Prostate", "Seminal Vesicles"
    ]
    print(len(class_list))
    # 类别无关指标累计
    iousum = 0  # iou系数和
    dicesumm = 0  # Dice系数和
    mBoundF_agnostic_sum = 0  # 类别无关的mBoundF累计
    counter = 0  # 用于计数处理了多少个批次的数据
    class_wise_iou_sum = {}
    class_wise_dice_sum = {}
    class_wise_count = {}
    class_wise_hd_sum = {}  # 各类别Hausdorff距离累计
    class_wise_hd95_sum = {}  # 各类别95% Hausdorff距离累计
    mBoundF_sum = 0
    pq_sum = 0
    pq_counter = 0
    class_pq_sum = {}
    class_pq_count = {}

    # class_colors = [(0, 255, 245), (0, 255, 231), (0, 255, 216), (0, 255, 201), (0, 255, 186), (0, 255, 172), (0, 255, 157), (0, 255, 142), (0, 255, 127), (0, 255, 113), (0, 255, 98), (0, 255, 83),
    # (0, 255, 68), (0, 255, 53), (0, 255, 39), (0, 255, 24), (0, 255, 9), (6, 255, 0), (21, 255, 0), (35, 255, 0), (50, 255, 0), (65, 255, 0), (80, 255, 0), (94, 255, 0),
    # (109, 255, 0), (124, 255, 0), (139, 255, 0), (153, 255, 0), (168, 255, 0), (183, 255, 0), (198, 255, 0), (212, 255, 0), (227, 255, 0), (242, 255, 0), (255, 251, 0), (255, 236, 0),
    # (255, 222, 0), (255, 207, 0), (255, 192, 0), (255, 177, 0), (255, 163, 0), (255, 148, 0), (255, 133, 0), (255, 118, 0), (255, 104, 0), (255, 89, 0), (255, 74, 0), (255, 59, 0), (255, 44, 0), (255, 30, 0), (255, 15, 0), (255, 0, 0)]
    class_colors = {
        0: (255, 0, 0),      # 红色
        1: (0, 255, 0),      # 绿色
        2: (0, 0, 255),      # 蓝色
        3: (255, 255, 0),    # 黄色
        4: (255, 0, 255),    # 洋红色
        5: (0, 255, 255),    # 青色
        6: (128, 0, 0),      # 暗红色
        7: (0, 128, 0),      # 暗绿色
        8: (0, 0, 128),      # 深蓝色
        9: (128, 128, 0),    # 橄榄色
        10: (128, 0, 128),   # 紫色
        11: (0, 128, 128),   # 深青色
        12: (192, 192, 192), # 银色
        13: (128, 128, 128), # 灰色
        14: (255, 165, 0),   # 橙色
        15: (255, 20, 147),  # 深粉色
        16: (75, 0, 130),    # 靛蓝色
        17: (173, 216, 230), # 浅蓝色
        18: (34, 139, 34),   # 森林绿
        19: (255, 215, 0),   # 金色
        20: (219, 112, 147), # 浅粉色
        21: (255, 99, 71),   # 番茄红
        22: (154, 205, 50),  # 黄绿色
        23: (85, 107, 47),   # 深橄榄绿
        24: (139, 69, 19),   # 马鞍棕色
        25: (189, 183, 107), # 深卡其色
        26: (240, 230, 140), # 卡其色
        27: (250, 250, 210), # 浅黄色
        28: (230, 230, 250), # 薰衣草色
        29: (216, 191, 216), # 蓟色
        30: (255, 228, 225), # 薄雾玫瑰
        31: (240, 128, 128), # 浅珊瑚色
        32: (255, 160, 122), # 浅鲑鱼色
        33: (255, 127, 80),  # 珊瑚色
        34: (255, 69, 0),    # 橙红色
        35: (255, 140, 0),   # 深橙色
        36: (184, 134, 11),  # 深金黄色
        37: (218, 165, 32),  # 金菊色
        38: (238, 232, 170), # 浅金菊色
        39: (189, 183, 107), # 深卡其布色
        40: (143, 188, 143), # 深海洋绿
        41: (102, 205, 170), # 中海洋绿
        42: (32, 178, 170),  # 浅海洋绿
        43: (0, 139, 139),   # 深青色
        44: (100, 149, 237), # 矢车菊蓝
        45: (25, 25, 112),   # 午夜蓝
        46: (72, 61, 139),   # 深板岩蓝
        47: (106, 90, 205),  # 板岩蓝
        48: (123, 104, 238), # 中板岩蓝
        49: (147, 112, 219), # 中紫色
        50: (139, 0, 139),   # 深洋红色
        51: (148, 0, 211),   # 深紫色
        52: (153, 50, 204),  # 暗兰花紫
        53: (186, 85, 211),  # 中兰花紫
        54: (128, 0, 0),     # 栗色
        55: (165, 42, 42),   # 褐色
        56: (178, 34, 34),   # 火砖色
        57: (205, 92, 92),   # 印度红
        58: (220, 20, 60),   # 猩红色
        59: (255, 0, 0),      # 红色
        60: (18, 34, 34),   
        61: (205, 9, 9), 
        62: (20, 200, 60),  
        63: (0, 30, 200)     
        }
    # RAOS新颜色
    # class_colors = {
    #     0: (0, 0, 0),
    #     1: (255, 0, 0),      # 红色
    #     2: (0, 255, 0),      # 绿色
    #     3: (0, 0, 210),      # 蓝色
    #     4: (255, 255, 0),    # 黄色
    #     5: (0, 255, 255),    # 水绿色
    #     6: (224, 0, 224),    # 紫色
    #     7: (225, 211, 188),  # 茶色
    #     8: (0, 0, 168),      # 深蓝色
    #     9: (205, 133, 63),   # 茶色
    #     10: (223, 191, 149), # 茶色
    #     11: (102, 205, 170), # 绿色
    #     12: (63, 126, 126),  # 深青色
    #     13: (60, 60, 123),   # 深紫色
    #     14: (45, 135, 85),   # 深绿色
    #     15: (221, 197, 195), # 珊瑚色
    #     16: (215, 156, 215), # 淡紫色       
	#     17: (102, 87, 197),  # 靛蓝色
    #     18: (206, 133, 108), # 茶色
    #     19: (125, 32, 32),  # 深红
    #     }



    for batch, img_path in tqdm.tqdm(dataset):
        img_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        print(img_name)
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()

        # 处理GT掩码，按类别分组
        mask_paths = glob.glob(img_path.replace("_image.png", "_mask") + "*")
        gt_masks = []
        gt_classes = []
        mask_gt_combined = np.zeros((512, 512))  # 用于类别无关的整体计算
        gt_instance_masks = []
        gt_instance_classes = []
        
        for maskpath in mask_paths:
            # 从文件名提取类别信息
            # 假设文件名格式为: xxx_mask_classID.png
            class_match = re.search(r'_mask_(\d+)\.png', maskpath)
            if class_match:
                class_id = int(class_match.group(1))
                mask = cv2.imread(maskpath, 0)
                mask_binary = (mask > 0).astype(np.float32)
                gt_masks.append(mask_binary)
                gt_classes.append(class_id)
                mask_gt_combined = np.maximum(mask_gt_combined, mask_binary)
                instances = mask_to_instance_masks(mask_binary)
                for inst in instances:
                    gt_instance_masks.append(inst)
                    gt_instance_classes.append(class_id)

        with torch.no_grad():
            output = network(batch['inp'], batch)  # dict_keys(['ct_hm', 'wh', 'ct', 'detection', 'it_ex', 'ex', 'it_py', 'py'])
            poly = output['py']
            detection = output['detection']

                       
            
    
            # ===== 评估 =====
            # 预测实例掩码与类别
                
            # ===== 可视化 =====
            if visual:
                if len(poly) > 0:
                    visual_poly = poly[-1] * 4
                    visualize_polygons(visual_poly, img, detection=detection, save_name=img_name, 
                        class_colors=class_colors, class_name=class_list, 
                        gt_masks=gt_masks, gt_classes=gt_classes)
                else:
                    print(f"[VIS] No polygons to visualize for {img_name}, skipping visualization")
            

            
            # 如果没有找到按类别分组的掩码，使用原始方式
            if not gt_masks:
                for maskpath in mask_paths:
                    mask = cv2.imread(maskpath, 0)
                    mask_binary = (mask > 0).astype(np.float32)
                    mask_gt_combined = np.maximum(mask_gt_combined, mask_binary)
                    instances = mask_to_instance_masks(mask_binary)
                    if instances:
                        for inst in instances:
                            gt_instance_masks.append(inst)
                            gt_instance_classes.append(1)
                    else:
                        gt_instance_masks.append(mask_binary)
                        gt_instance_classes.append(1)
                # 对于无法识别类别的情况，假设为类别1
                gt_masks = [mask_gt_combined]
                gt_classes = [1]
            
            # 确保mask_gt的值在0-1之间
            mask_gt_combined = np.clip(mask_gt_combined, 0, 1)
            
            # 修复：正确生成类别无关的预测掩码，使用所有预测结果
            pred_mask_combined = np.zeros((512, 512))
            if len(poly) > 0 and poly[-1].shape[0] > 0:  # 确保有预测结果
                for j in range(poly[-1].shape[0]):
                    # 为每个预测创建单独的掩码，然后合并
                    single_poly = poly[-1][j:j+1]
                    single_mask = poly2mask(single_poly)
                    pred_mask_combined = np.maximum(pred_mask_combined, single_mask)
            
            # 计算类别无关的指标（兼容性）
            iou_overall = cal_iou(pred_mask_combined, mask_gt_combined)
            # dice_overall = cal_dice(pred_mask_combined, mask_gt_combined)
            dice_overall = cal_dice(iou_overall)
            mBoundF_agnostic = cal_mBoundF_agnostic(pred_mask_combined, mask_gt_combined)
            
            # 类别无关指标不再计算HD，因为所有解剖结构混合计算HD没有临床意义
            # HD和HD95只在类别相关指标中有意义
            
            print(f"Overall IoU: {iou_overall:.4f}, Dice: {dice_overall:.4f}, mBoundF: {mBoundF_agnostic:.4f}")
            iousum += iou_overall
            dicesumm += dice_overall
            mBoundF_agnostic_sum += mBoundF_agnostic
            
            # 计算类别相关的指标
            # 构建预测掩码列表和类别列表
            pred_masks = []
            pred_classes = []
            pred_instance_masks = []
            pred_instance_classes = []
            # 初始化默认指标值
            mean_iou = 0.0
            mean_dice = 0.0
            mBoundF = 0.0
            
            if len(poly) > 0 and poly[-1].shape[0] > 0:  # 确保有预测结果
                for j in range(poly[-1].shape[0]):
                    # 为每个预测创建单独的掩码
                    single_poly = poly[-1][j:j+1]  # 取单个poly，形状 [1, num_points, 2]
                    pred_mask = poly2mask(single_poly)
                    pred_masks.append(pred_mask)
                    pred_classes.append(int(detection[0, j, 5])+1)
                    pred_instance_masks.append(pred_mask)
                    pred_instance_classes.append(int(detection[0, j, 5]) + 1)

            pred_masks_merged = []
            pred_classes_merged = []
            if len(pred_masks) > 0:
                merged_by_class = {}
                for m, c in zip(pred_masks, pred_classes):
                    if c not in merged_by_class:
                        merged_by_class[c] = m.copy()
                    else:
                        merged_by_class[c] = np.maximum(merged_by_class[c], m)
                for c in sorted(merged_by_class.keys()):
                    pred_classes_merged.append(c)
                    pred_masks_merged.append(merged_by_class[c])

            if len(pred_masks) > 0 and len(gt_masks) > 0:
                try:
                    class_ious, class_dices, class_hds, class_hd95s, mean_iou, mean_dice, mean_hd, mean_hd95 = cal_class_wise_metrics(
                        pred_masks_merged, gt_masks, pred_classes_merged, gt_classes)
                    
                    # 计算mBoundF
                    mBoundF = cal_mBoundF(pred_masks_merged, gt_masks, pred_classes_merged, gt_classes)
                    mBoundF_sum += mBoundF
                    
                    print(f"Class-wise mIoU: {mean_iou:.4f}, mDice: {mean_dice:.4f}, mBoundF: {mBoundF:.4f}, HD: {mean_hd:.2f}, HD95: {mean_hd95:.2f}")
                    
                    # 累计各类别的指标
                    for class_id, iou in class_ious.items():
                        if class_id not in class_wise_iou_sum:
                            class_wise_iou_sum[class_id] = 0
                            class_wise_dice_sum[class_id] = 0
                            class_wise_count[class_id] = 0
                            # class_wise_hd_sum[class_id] = 0
                            # class_wise_hd95_sum[class_id] = 0
                        class_wise_iou_sum[class_id] += iou
                        class_wise_dice_sum[class_id] += class_dices[class_id]
                        # class_wise_hd_sum[class_id] += class_hds[class_id]
                        # class_wise_hd95_sum[class_id] += class_hd95s[class_id]
                        class_wise_count[class_id] += 1
                except Exception as e:
                    print(f"Error calculating class-wise metrics: {e}")
                    print(f"pred_masks count: {len(pred_masks)}, gt_masks count: {len(gt_masks)}")
                    print(f"pred_classes: {pred_classes}, gt_classes: {gt_classes}")
                    # 跳过这张图像的类别相关指标计算
            else:
                print(f"Skipping class-wise metrics: pred_masks={len(pred_masks)}, gt_masks={len(gt_masks)}")
            
            counter += 1
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f'\n{img_name}\n')
                f.write(f'Overall IoU: {iou_overall:.4f}, Dice: {dice_overall:.4f}\n')
                f.write(f'Class-wise mIoU: {mean_iou:.4f}, mDice: {mean_dice:.4f}\n')

            # ===== 计算PQ =====
            if gt_instance_masks or pred_instance_masks:
                pq_per_class, pq_mean = compute_pq(
                    pred_instance_masks, pred_instance_classes,
                    gt_instance_masks, gt_instance_classes, iou_thresh=0.5
                )
                print(f"Class-aware PQ: {pq_mean:.4f}")
                pq_sum += pq_mean
                pq_counter += 1
                for class_id, stats in pq_per_class.items():
                    class_pq_sum[class_id] = class_pq_sum.get(class_id, 0.0) + stats['pq']
                    class_pq_count[class_id] = class_pq_count.get(class_id, 0) + 1
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f'PQ: {pq_mean:.4f}\n')
            else:
                print("Skipping PQ calculation: no instances found in prediction or GT.")
            
    # 输出类别无关的指标
    print("\n=== 类别无关指标 (Class-agnostic Metrics) ===")
    print("Overall mIoU: {:.4f}".format(iousum / counter))
    print("Overall mDice: {:.4f}".format(dicesumm / counter))

    
      
    # 输出类别相关的指标
    print("\n=== 类别相关指标 (Class-aware Metrics) ===")
    if class_wise_count:
        # 计算各类别的平均指标
        class_avg_ious = {}
        class_avg_dices = {}
        class_avg_hds = {}
        class_avg_hd95s = {}
        for class_id in class_wise_count:
            class_avg_ious[class_id] = class_wise_iou_sum[class_id] / class_wise_count[class_id]
            class_avg_dices[class_id] = class_wise_dice_sum[class_id] / class_wise_count[class_id]
        
        # 计算整体平均
        overall_class_miou = np.mean(list(class_avg_ious.values()))
        overall_class_mdice = np.mean(list(class_avg_dices.values()))

        overall_pq = pq_sum / pq_counter if pq_counter > 0 else 0.0
        
        print("Class-aware mIoU: {:.4f}".format(overall_class_miou))
        print("Class-aware mDice: {:.4f}".format(overall_class_mdice))
        print("Class-aware PQ: {:.4f}".format(overall_pq))
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'\nProcessed {counter} images in total.\n')
            f.write("Overall mIoU: {:.4f}\n".format(iousum / counter))
            f.write("Overall mDice: {:.4f}\n".format(dicesumm / counter))
            f.write("Class-wise mIoU: {:.4f}\n".format(overall_class_miou))
            f.write("Class-wise mDice: {:.4f}\n".format(overall_class_mdice))
            f.write("Class-aware PQ: {:.4f}\n".format(overall_pq))
        
        # 输出各类别的详细指标
        print("\n=== 各类别详细指标 (Per-class Metrics) ===")
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'\n=== 各类别详细指标 ===\n')
        for class_id in sorted(class_avg_ious.keys()):
            class_name = class_list[class_id] if class_id < len(class_list) else f"Class_{class_id}"
            class_pq_avg = 0.0
            if class_id in class_pq_sum and class_pq_count.get(class_id, 0) > 0:
                class_pq_avg = class_pq_sum[class_id] / class_pq_count[class_id]
            print(f"Class {class_id} ({class_name}): IoU={class_avg_ious[class_id]:.4f}, "
                  f"Dice={class_avg_dices[class_id]:.4f}, "
                  f"PQ={class_pq_avg:.4f}")
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"Class {class_id} ({class_name}): IoU={class_avg_ious[class_id]:.4f}, "
                  f"Dice={class_avg_dices[class_id]:.4f}, "
                  f"PQ={class_pq_avg:.4f}\n")
    else:
        print("No class-aware metrics calculated (no detections or GT found)")
    
    print(f"\nProcessed {counter} images in total.")

    # 关闭CSV
    # for f in csv_files.values():
    #     try:
    #         f.close()
    #     except:
    #         pass
    
            
            
     