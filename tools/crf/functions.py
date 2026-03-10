import numpy as np
import torch
import csv

def _seq_to_numpy(seq_b):
    # seq_b: [L,6] (x1,y1,x2,y2,score,cls) 或 torch.Tensor
    if isinstance(seq_b, torch.Tensor):
        return seq_b.detach().cpu().numpy()
    return np.asarray(seq_b)

def _normalize_and_pad(seq_arr, H, W, K=20, cls_plus_one=True):
    """
    输入: seq_arr [L,6] -> x1,y1,x2,y2,score,cls(0-based)
    - 归一化坐标到[0,1]
    - 仅保留 score>0 的有效框
    - 按 score 降序取前K个；不足K用全0补
    - 类别可选 +1（与你数据集一致）
    返回: [K,6] -> x1n,y1n,x2n,y2n,score,cls
    """
    if seq_arr.ndim != 2 or seq_arr.shape[-1] < 6:
        return np.zeros((K, 6), dtype=np.float32)

    a = seq_arr.astype(np.float32)
    # 过滤无效框
    valid = a[:, 4] > -1
    a = a[valid] if valid.any() else np.zeros((0, 6), dtype=np.float32)

    if a.shape[0] > 0:
        # 归一化并裁剪
        a[:, 0] = np.clip(a[:, 0] / max(W, 1), 0, 1)
        a[:, 2] = np.clip(a[:, 2] / max(W, 1), 0, 1)
        a[:, 1] = np.clip(a[:, 1] / max(H, 1), 0, 1)
        a[:, 3] = np.clip(a[:, 3] / max(H, 1), 0, 1)
        # 类别索引偏移（若你的标注是1..C，设为True）
        if cls_plus_one:
            a[:, 5] = a[:, 5] + 1

        # 按 score 排序
        order = np.argsort(a[:, 5])
        a = a[order]

    # 截断/补零到K
    if a.shape[0] >= K:
        out = a[:K, :6]
    else:
        out = np.zeros((K, 6), dtype=np.float32)
        if a.shape[0] > 0:
            out[:a.shape[0], :6] = a[:, :6]
    return out

def _append_seq_rows(csv_writer, image_id, rows):
    # rows: [K,6]
    for i in range(rows.shape[0]):
        x1n, y1n, x2n, y2n, score, cls = rows[i].tolist()
        csv_writer.writerow([image_id, i, x1n, y1n, x2n, y2n, int(cls), score])

# ----------------------------------------------用到的一些函数------------------------------------------------- #
def trim_zero(input):
    """处理一张图的函数，裁剪补零。"""
    ix_unpadded = np.where(np.sum(input, axis=1) > 0)[0]  # 非补0的位置
    output = input[ix_unpadded]
    return output, ix_unpadded

def find_unique_detections(input_class_labels, class_num):
    """输入一张图的探测标签结果，观察某个探测正例（标签不是0的）是否是这个类别的唯一正例。
    如果是这个类别的唯一正例，就把相应的位置标记为1，否则标记为0（即，如果是负例，或者虽然是正例，但是不止1个，就都标记0）。
    输入的input_class_labels是n*1向量，n就是这张图中探测结果的总数。第0类是背景类，不考虑。
    class_num是算上背景类的总类别数，比如说一共3种椎骨，这个就是4。
    """
    unique = np.zeros_like(input_class_labels, dtype=bool)
    for i in range(1, class_num):  # 第0类是背景类，从第1类开始。
        ixi = np.where(input_class_labels==i)[0]
        if ixi.shape[0] == 1:  # 是否只有1个探测结果，是这个标签的？
            unique[ixi] = True
    return unique

def find_unique_or_most_confident_detections(input_labels_and_score, class_num):
    """
    跟上面的函数有点像，只不过这儿是，如果某一个探测结果是某个非背景类的唯一的一个，
        或者虽然不唯一但是是mrcnn得分最大的（合理吗？是不是mrcnn非零类得分+rpnd得分最大的好一些？），就都把相应位置标记出来。
    输入的input_class_labels是n*2向量，n就是这张图中探测结果的总数，然后一列是类别，一列是得分。第0类是背景类，不考虑。
    """
    class_labels = input_labels_and_score[:,0]
    class_lprobs = input_labels_and_score[:,1]
    mrcnn_probs = input_labels_and_score[:,2]  # 加的。。
    unique = np.zeros_like(class_labels, dtype=bool)
    for i in range(1, class_num):  # 第0类是背景类，从第1类开始。
        ixi = np.where(class_labels==i)[0]  # 属于这一类的探测结果索引号组成的向量，如，[0, 1, 25]
        if ixi.shape[0] == 1:  # 是否只有1个探测结果，是这个标签的？
            unique[ixi] = True
        elif ixi.shape[0] >= 2:  # 有2个以上的探测结果
            most_confident_pos_in_ix = np.argmax(class_lprobs[ixi])  # 是在ixi中的位置
            unique[ixi[most_confident_pos_in_ix]] = True  # 如果不止1个，那么，得分最大的予以保留。
            # 加一个，如果mrcnn得分大于0.999且别的这一类的mrcnn得分都小于0.95，那么也保留。就是为了split2的13的。。
            highest_mrcnn_pos = np.argmax(mrcnn_probs[ixi])
            highest_mrcnn = np.max(mrcnn_probs[ixi])
            mrcnn_diff = highest_mrcnn-mrcnn_probs[ixi]  # 最大得分和其他得分的差值
            mrcnn_diff_sort = np.sort(mrcnn_diff)
            next_highest_diff = mrcnn_diff_sort[1]  # 第二大的得分，和最大得分的差值。
            if highest_mrcnn>0.999 and next_highest_diff > 0.1:
                unique[ixi[highest_mrcnn_pos]] = True
        else:  # 没有这一类的探测结果
            pass
    return unique

def select_by_rpn_and_mrcnn(input_detections, keep, line, threshold):
    """根据rpn和mrcnn的结果删掉一些不太可能的探测结果。
    MRCNN的探测结果，会有下面的问题：
        MRCNN程序做测试的时候，是直接先把背景类的都删了（有一句keep = tf.where(class_ids > 0)[:, 0]），
        然后选出信心分值大于DETECTION_MIN_CONFIDENCE阈值的（有一句conf_keep =
            tf.where(class_ scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]），
        然后选出来这两类的交集，也就是说是前景类、且信心分值大于DETECTION_MIN_CONFIDENCE阈值的。
        这样的话，有可能一些分类错误（如第1类给判断成第0类即背景）的就检测不到了，
        也有可能有好几个信心分值的大于那个阈值的，就多弄出来了。
    这个函数主要解决“多弄出来”的问题。具体地，删除rpn信心分值小于threshold的mrcnn探测结果，
        但是，要保留每一类的、mrcnn得分最大的那一个，因为有的时候，mrcnn预测某一类的rpn分值都不大于threshold，
            这种情况虽然少，但是还是要留意一下。最后的输入keep就相当于是这个免死金牌。
    """
    ix = np.where(input_detections[:, line] > threshold)[0]
    ix1 = np.where(keep==True)[0]
    ix_all = np.array(list(set(ix).union(set(ix1))))  # 以上两个np.array的“并集”。
    # 要先变成变成set，再变回list，再变回np.array。
    selected = input_detections[ix_all, :]
    # sorted = np.sort(selected, axis=0)  # 注意不能用这个，会把所有的列都从小到大排序的。
    sorted = selected[selected[:, 0].argsort()]  # 按照第一列排序
    sorted_ix = ix_all[selected[:, 0].argsort()]  # 按照第一列排序。注意这儿要多一个[0]，因为np.where输出的是tuple。
    P = input_detections.shape[0] - sorted.shape[0]
    padded = np.pad(sorted, ((0, P), (0, 0)),'constant',constant_values = (0))  # 必须写((0, P), (0, 0))，
    # 而不能是((0, P))，因为((0, P))相当于是((0, P), (0, P))！
    padded_ix = np.pad(sorted_ix, ((0, P)), 'constant', constant_values=(-1))  # 必须写((0, P))，
    # 而不能是((0, P), (0, 0))，因为这个时候是1维的。然后要补-1而不是0，因为ix是本来就可能取0的。
    return padded, padded_ix

def select_mrcnn_by_rpn(input, ix):
    """这个是处理一张图的函数，先去掉ix里补的-1，然后选出来，然后再补0。"""
    ix_unpadded = np.where(ix > -1)[0]  # ix中非补-1的位置
    pad_len = input.shape[0] - ix_unpadded.shape[0]
    output = input[ix[ix_unpadded]]
    # 上句，原来是input[ix[ix_unpadded], :]，但是考虑到input的维度可能不一样，就删掉了那个“,:”，这样，只让他在第0维度挑选就可以了。
    #     比如，下面处理_mrcnn_boxes_and_scores_all_test的时候，输入是(100, 6)，而处理aver_score的时候，输入是(100,)。
    if len(output.shape) == 2:
        padded_output = np.pad(output, ((0, pad_len), (0, 0)), 'constant', constant_values=(0))
    elif len(output.shape) == 1:
        padded_output = np.pad(output, ((0, pad_len)), 'constant', constant_values=(0))
    else:
        padded_output = output
        print('输入维度不是1维或者2维，没有补零，留意是否有问题。')
    return padded_output
# --------------------------------------------用到的一些函数结束------------------------------------------------ #