import numpy as np
import sys
# from hausdorff import hausdorff_distance

  
def compute_tfpn(pred_mask,true_mask):

    c, r = true_mask.shape[1], pred_mask.shape[2]
    num_pixel = c*r

    # if true_mask.shape[2] != 400:
    #     print(true_mask)

    true_mask = true_mask>0
    pred_mask = pred_mask>0

    # print("pred_mask.shape", pred_mask.shape)
    # print("true_mask.shape", true_mask.shape)
    # if true_mask.shape[2] != 400:
    #     print(true_mask)
    
    tp = (true_mask & pred_mask).sum()
    fp = (~true_mask & pred_mask).sum()
    tn = (~(true_mask | pred_mask)).sum()
    fn = (true_mask & (~pred_mask)).sum()


    return {
        'TP': tp/num_pixel,
        'FP': fp/num_pixel,
        # 'tn': tn/num_pixel,        
        'FN': fn/num_pixel
    }

def compute_tfpns(n_batch, pred_masks, true_masks):
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp_sum += tfpn["TP"]
        fp_sum += tfpn["FP"]
        fn_sum += tfpn["FN"]
    return tp_sum/n_batch, fp_sum/n_batch, fn_sum/n_batch

def compute_e1(n_batch, pred_masks, true_masks):

    sum_e1 = 0
    for i in range(n_batch):
        tpfn = compute_tfpn(pred_masks[i],true_masks[i])
        fp, fn = tpfn['FP'], tpfn['FN']
        sum_e1 += fp+fn

    return sum_e1/n_batch


def compute_miou(n_batch, true_masks, pred_masks):

    sum_iou = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['TP'], tfpn['FP'], tfpn['FN']
        if tp+fn+fp == 0:
            iou=1
        else:
            iou=tp/(tp+fn+fp)
        sum_iou += iou

    return sum_iou/n_batch


def compute_dice(n_batch, pred_masks, true_masks):

    sum_dice = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['TP'], tfpn['FP'], tfpn['FN']
        if 2*tp+fn+fp == 0:
            dice=1
        else:
            dice=2*tp/(2*tp+fn+fp)
        sum_dice += dice

    return sum_dice/n_batch


def compute_f1(n_batch, pred_masks, true_masks):
    recall_sum = 0
    precision_sum = 0
    sum_f1 = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(pred_masks[i], true_masks[i])
        tp, fp, fn = tfpn['TP'], tfpn['FP'], tfpn['FN']
        if tp+fp == 0:
            precision = tp
        else:
            precision = tp / (tp+fp)
        recall = tp / (tp+fn)

        recall_sum += recall
        precision_sum += precision

        if precision+recall == 0:
            f1 = tp
        else:
            f1 = (2*precision*recall) / (precision+recall)
        if f1 > 999:
            f1 = 0
        sum_f1 += f1

    return sum_f1/n_batch, recall_sum/n_batch, precision_sum/n_batch


def get_coords(nparray):
    coords = []

    h, w = nparray.shape
    for i in range(h):
        for j in range(w):
            if nparray[i, j] > 0:
                coords.append([i, j])
    
    return np.asarray(coords)


# def Hausdorff(pred_edge, true_edge):
#     pred_edge = np.asarray(pred_edge>0)
#     true_edge = np.asarray(true_edge>0)
#     _, h, w = true_edge.shape

#     pred_coords = get_coords(pred_edge[0])
#     true_coords = get_coords(true_edge[0])

#     if len(pred_coords) == 0 or len(true_coords)==0:
#         hsdf =  float("inf")
#     else:
#         hsdf = hausdorff_distance(pred_coords, true_coords) / w

#     return hsdf


# def compute_hsdf(n_batch, pred_edges, true_edges):
#     hsdf = 0
#     for i in range(n_batch):
#         hsdf_i = Hausdorff(pred_edges[i], true_edges[i])
#         if hsdf_i == float("inf"):
#             continue
#         hsdf += hsdf_i
#     return hsdf/n_batch


def evaluate_loc(pred_masks, true_masks):#, pred_edges, true_edges, dataset_name):

    n_batch = true_masks.size()[0]
    
    e1 = compute_e1(n_batch, pred_masks, true_masks).item()
    # try:
    dice = compute_dice(n_batch, pred_masks, true_masks).item()
    
    iou = compute_miou(n_batch, pred_masks, true_masks).item()

    f1, recall, precision = compute_f1(n_batch, pred_masks, true_masks)

    tp, fp, fn = compute_tfpns(n_batch, pred_masks, true_masks)
    # print(type(e1), type(dice), type(iou))
    # except:
    #     dice = compute_dice(n_batch, pred_masks, true_masks)
    #     iou = compute_miou(n_batch, pred_masks, true_masks)
    # # caculate hausdorff takes too long
    # hsdf = compute_hsdf(n_batch, pred_edges.cpu(), true_edges.cpu())
    # print("iou >>>>>:", iou)
    # print("dice>>>>>>>", dice)
    return {
        # here, for some reason E1 and IoU were multiplied by 100
        'E1': e1,
        'IoU': iou,
        'Dice': dice,
        'TP':tp,
        'FP':fp,
        'FN':fn,
        'recall':recall,
        'precision':precision,
        'F1':f1,
        # 'Hsdf': hsdf*100
        
    }

