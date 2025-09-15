# nms_video_card.py
from pathlib import Path
import numpy as np

def IoU(b1, b2):
    # b: [x_c, y_c, w, h] 归一化坐标
    x1 = max(b1[0]-b1[2]/2, b2[0]-b2[2]/2)
    y1 = max(b1[1]-b1[3]/2, b2[1]-b2[3]/2)
    x2 = min(b1[0]+b1[2]/2, b2[0]+b2[2]/2)
    y2 = min(b1[1]+b1[3]/2, b2[1]+b2[3]/2)
    if x2<=x1 or y2<=y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / (union + 1e-6)

src = Path('runs/detect/pred/labels')
dst = Path('labels/pred_nms'); dst.mkdir(exist_ok=True)

for txt in src.glob('*.txt'):
    d = np.loadtxt(txt, ndmin=2)          # [class, x, y, w, h, conf]
    keep = []
    # 只处理 class=0 (video_card)
    idx = np.where(d[:,0]==0)[0]
    if idx.size==0:                       # 没有 video_card 直接复制原文件
        np.savetxt(dst/txt.name, d, fmt='%g')
        continue
    boxes = d[idx, 1:5]                   # x y w h
    confs = d[idx, 5] if d.shape[1]==6 else np.ones(idx.size)
    order = confs.argsort()[::-1]         # 高置信度在前
    while order.size>0:
        i = order[0]
        keep.append(idx[i])
        iou = np.array([IoU(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][iou<=0.5]        # 删除 IoU>0.5 的框
    keep_arr = d[np.array(keep)]
    # 把其余非 video_card 类别拼回去
    other = d[d[:,0]!=0]
    out = np.vstack([keep_arr, other]) if other.size else keep_arr
    np.savetxt(dst/txt.name, out, fmt='%g')

print('✅ IoU>0.5 去重完成，结果在 labels/pred_nms/')