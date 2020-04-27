
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import directories

img_width = 848
img_height = 480

prediction_list = [os.path.splitext(file)[0] for file in os.listdir(directories.prediction_label_dir) if file.endswith('.txt')]
gt_list = [os.path.splitext(file)[0] for file in os.listdir(directories.ground_truth_label_dir) if file.endswith('.txt')]

def rel_to_px_bb(x, y, bb_width, bb_height, img_width, img_height):

    x_px = x * img_width
    y_px = y * img_height
    width_px = bb_width * img_width
    height_px = bb_height * img_height

    x_min = int(x_px - width_px/2)
    x_max = int(x_px + width_px/2)
    y_min = int(y_px - height_px/2)
    y_max = int(y_px + height_px/2)

    return [x_min, x_max, y_min, y_max]

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    class_t, x1_t, x2_t, y1_t, y2_t = gt_box[0:5]
    class_p, x1_p, x2_p, y1_p, y2_p = pred_box[0:5]

    #print("calc_iou_individual")
    #print(x1_t, x2_t, y1_t, y2_t)
    #print(x1_p, x2_p, y1_p, y2_p)

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    #print("interarea {}".format(inter_area))
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    #print("true_box_area {}".format(true_box_area))
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    #print("pred_box_area {}".format(pred_box_area))
    iou = float(inter_area) / (true_box_area + pred_box_area - inter_area)
    #print("iou {}".format(iou))
    #print()
    return iou

def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            #print("Comparing gt: \n{}\nand pred:\n{}".format(gt_box, pred_box))
            #print("iou: {}".format(iou))
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    #print("Total results: True Pos: {}, False Pos: {}, False Neg: {}".format(true_pos, false_pos, false_neg))

    try:
        precision = float(true_pos)/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = float(true_pos)/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def print_mAP(prec_recall_dataset):
    ''' calculates and prints the average precision for recall values [0:0.1:1]'''

    prec_sum = 0
    for recall in np.linspace(0, 1, 11, endpoint=True):
        index = 0
        precision = 0
        try:
            index = next(x[0] for x in enumerate(prec_recall_dataset[1]) if x[1] >= recall)
            precision = prec_recall_dataset[0][index]
        except:
            pass
        
        prec_sum += precision
        
        #print("Recall reaches {} at precision {} (index {})".format(recall, precision, index))

    avg_prec = prec_sum / 11.0

    print("Average precision: {}%".format(round(avg_prec*100, 2)))

    return round(avg_prec*100, 2)


def plot_pr_curve(
    prec_recall_datasets, interp_prec_recall_datasets, iou_thres_list, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    APs = []

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]

    for i, prec_recs_dataset in enumerate(prec_recall_datasets):
        #ax.scatter(prec_recs_dataset[1], prec_recs_dataset[0], label=iou_thres_list[i], s=1, color=COLORS[i])
        #ax.plot(prec_recs_dataset[1], prec_recs_dataset[0], label=iou_thres_list[i], linewidth=1.0, color=COLORS[i])
        ax.plot(prec_recs_dataset[1], prec_recs_dataset[0], linewidth=1.0, color=COLORS[i])
    
    for i, prec_recs_dataset in enumerate(interp_prec_recall_datasets):
        AP = print_mAP(prec_recs_dataset)
        APs.append(AP)

        
        #ax.plot(prec_recs_dataset[1], prec_recs_dataset[0], label=str(iou_thres_list[i]) + "_inter", linewidth=3.0, color=COLORS[i])
        ax.plot(prec_recs_dataset[1], prec_recs_dataset[0], label=str(iou_thres_list[i]) + ": AP=" + str(AP) + "%",linewidth=3.0, color=COLORS[i])

    meanAP = np.average(APs)


    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    #ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_title('')
    ax.set_xlim([0.0,1.05])
    ax.set_ylim([0.0,1.15])

    spaces = " " * 150

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=5, title='AP @ IoU Thres:' + spaces + 'Mean AP: ' + str(round(meanAP, 2)) + '%', frameon=True)
    plt.rc('grid', linestyle=":", color='red')
    plt.grid()

    plt.savefig(directories.fig_save_path)

    #plt.show()

def interpolate_prec_recall_datasets(datasets):
    interpolated_datasets = []
    for dataset in datasets:
        interpolated_dataset = [[], []]

        for idx, precision in enumerate(dataset[0]):
            max_prec_right = max(dataset[0][idx:])
            recall = dataset[1][idx]

            interpolated_dataset[0].append(max(precision, max_prec_right))
            interpolated_dataset[1].append(recall)
        
        # insert beginning point
        try:
            interpolated_dataset[0].insert(0, max(interpolated_dataset[0]))
            interpolated_dataset[1].insert(0, 0)
        except:
            pass
        

        # insert end point
        try:
            interpolated_dataset[1].append(max(interpolated_dataset[1]))
            interpolated_dataset[0].append(0)
        except:
            pass
        interpolated_datasets.append(interpolated_dataset)
    
    return interpolated_datasets


def AP_from_prec_rec(precisions, recalls):
    pass


if not len(prediction_list) == len(gt_list):
    print("Numbers of ground_truth and prediction label files do not match")

for gt in gt_list:
    if not ("prediction_" + gt) in prediction_list:
        print("Missing prediction for " + gt)

for pred in prediction_list:
    if not pred[11:] in gt_list:
        print("Missing ground truth for " + pred)


relevant_class = '0'
relevant_class_string = "FKLT-6410"
iou_thres_list = [0.5, 0.75, 0.85, 0.9, 0.95]
confidence_grouping_factor = 1

precision_recall_datasets = []

pred_labels_all_classes = []
gt_labels_all_classes = []

confidence_values = []


# read in label files
for frame, pred in enumerate(prediction_list): # loop over label files / frames
            
            # read in predictions
            pred_label_file = open(directories.prediction_label_dir + pred + ".txt", "r")
            pred_labels = [line.split(" ") for line in pred_label_file.read().splitlines()]
            # cast coord fields to int and confidence to float
            for label_index, label in enumerate(pred_labels):
                pred_labels[label_index][1:5] = [int(x) for x in label[1:5]]
                confidence = float(label[5])
                pred_labels[label_index][5] = confidence
                confidence_values.append(confidence)
            pred_labels_all_classes.append(pred_labels)
            

            # read in ground truth
            gt_label_file = open(directories.ground_truth_label_dir + pred[11:] + ".txt", "r")
            gt_labels = [line.split(" ") for line in gt_label_file.read().splitlines()]
            # convert gt_labels to px coords and overwrite
            for label_index, label in enumerate(gt_labels):
                rel_label = [float(x) for x in label]
                px_label = rel_to_px_bb(*rel_label[1:5], img_width=img_width, img_height=img_height)
                gt_labels[label_index][1:5] = px_label
            gt_labels_all_classes.append(gt_labels)

# sort and group confidence values
confidence_values.sort(reverse=True)
confidence_values = [val for idx, val in enumerate(confidence_values) if idx%confidence_grouping_factor == 0]
confidence_values.insert(0, 1.0)
confidence_values.append(0.0)

print("Prediction labels:\n{}\n... ({} total entries)".format(pred_labels_all_classes[0:2], len(pred_labels_all_classes)))
print("\nGround truth labels:\n{}\n... ({} total entries)".format(gt_labels_all_classes[0:2], len(gt_labels_all_classes)))
print("\nGrouped confidence values:\n{}\n...\n{}\n... ({} total entries)".format(confidence_values[:10], confidence_values[-10:], len(confidence_values)))

for iou_thres in iou_thres_list:
    print("\nGenerating precision-recall-curve for IoU-threshold: {}".format(iou_thres))

    recalls = []
    precisions = []


    for confidence_thres in confidence_values:
        #print("Confidence threshold: {}".format(confidence_thres))

        image_results = {}

        for frame, pred in enumerate(prediction_list): # loop over label files / frames
            
            # filter by class and confidence
            pred_labels = [x for x in pred_labels_all_classes[frame] if (x[0] == relevant_class and x[5] >= confidence_thres)]

            # filter by class
            gt_labels = [x for x in gt_labels_all_classes[frame] if x[0] == relevant_class]

            # match all BBs
            results = get_single_image_results(gt_labels, pred_labels, iou_thres)

            image_results[str(frame)] = results



        prec, recall = calc_precision_recall(image_results)

        if prec > 0.0 and recall > 0.0:
            precisions.append(prec)
            recalls.append(recall)
            
    precision_recall_datasets.append([precisions, recalls])

interp_precision_recall_datasets = interpolate_prec_recall_datasets(precision_recall_datasets)

#print(precision_recall_datasets[0])
#print(interp_precision_recall_datasets[0])

plot_pr_curve(precision_recall_datasets, interp_precision_recall_datasets, iou_thres_list, category=relevant_class_string)