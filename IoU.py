
def calculate_iou(boxA, boxB):

    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    areaA = aw * ah
    areaB = bw * bh

    union = areaA + areaB - intersection

    if union == 0:
        return 0.0

    return intersection / union