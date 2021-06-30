from IoU import intersection_over_union
import torch
from collections import Counter

def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, num_classes=22
):
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6

    # get all detections and ground truths for specific class c
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # number of GT training boxes per image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # amount_bboxes = {0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0])}
        # above example means image 0 has 3 GT box, image 1 has 5 GT box
        detections.sort(key=lambda x: x[2], reverse = True) # sort based on prob scores
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # for each bbox detected
        for detection_idx, detection in enumerate(detections):
            # get all GT bboxes for particular image.
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            num_gts = len(ground_truth_img)
            best_iou = 0
            # find the GT bbox that has highest IOU with curr detection bbox
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            # if IOU is greater than threshold
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # GT bbox has not been used
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1    # now covered
                else:   # if bbox already covered previously
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1


        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions)) # point (0,1)
        recalls = torch.cat((torch.tensor([0]), recalls))   # point (1, 0)
        average_precisions.append(torch.trapz(precisions, recalls)) # numerical integration

    return sum(average_precisions) / len(average_precisions)

