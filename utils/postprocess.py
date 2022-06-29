from core.kitti_dataset import KittiDataset
from core.models.pixor import PIXOR
from utils.one_hot import one_hot
from utils.preprocess import trasform_label2metric
from core.losses import SmoothL1Loss

from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import json

import torch.nn.functional as F

def convert_format(boxes_array):
    """

    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons = convert_format(boxes)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)

def filter_pred(reg_pred, cls_pred, config, score_threshold, nms_threshold):
    geometry = config["geometry"]

    ratio = 4
    reg_pred = reg_pred[0]
    cls_pred = cls_pred[0]
    cos_t, sin_t, dx, dy, log_w, log_l = np.split(reg_pred, 6, axis=0)


    cls_probs = np.max(cls_pred, axis = 0)
    cls_ids = np.argmax(cls_pred, axis = 0)

    if isinstance(score_threshold, list):
        thres = np.array(score_threshold)
        thres_arr = np.repeat(np.expand_dims(thres, 1), cls_pred.shape[1], 1)
        thres_arr = np.repeat(np.expand_dims(thres_arr, 2), cls_pred.shape[2], 2)

        comparisons = cls_pred > thres_arr
        idxs = np.logical_and(np.any(comparisons, 0), cls_ids != 0)
    else:
        idxs = np.logical_and(cls_probs > score_threshold, cls_ids != 0)
    cls = cls_ids[idxs]
    scores = cls_probs[idxs]
    y = np.arange(geometry["label_shape"][0])
    x = np.arange(geometry["label_shape"][1])
    xx, yy = np.meshgrid(x, y)


    center_y = dy + yy * ratio * geometry["y_res"] + geometry["y_min"]
    center_x = dx + xx * ratio * geometry["x_res"] + geometry["x_min"]
    l = np.exp(log_l)
    w = np.exp(log_w)
    yaw2 = np.arctan2(sin_t, cos_t)
    yaw = yaw2 / 2
    
    cos_t = np.cos(yaw)
    sin_t = np.sin(yaw)

    rear_left_x = center_x - l/2 * cos_t - w/2 * sin_t
    rear_left_y = center_y - l/2 * sin_t + w/2 * cos_t
    rear_right_x = center_x - l/2 * cos_t + w/2 * sin_t
    rear_right_y = center_y - l/2 * sin_t - w/2 * cos_t
    front_right_x = center_x + l/2 * cos_t + w/2 * sin_t
    front_right_y = center_y + l/2 * sin_t - w/2 * cos_t
    front_left_x = center_x + l/2 * cos_t - w/2 * sin_t
    front_left_y = center_y + l/2 * sin_t + w/2 * cos_t

    decoded_reg = np.concatenate([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                               front_right_x, front_right_y, front_left_x, front_left_y], axis=0)
    decoded_reg = np.swapaxes(decoded_reg, 0, 1)
    decoded_reg = np.swapaxes(decoded_reg, 1, 2)
    decoded_reg = decoded_reg[idxs]
    corners = np.reshape(decoded_reg, (-1, 4, 2))

    center_y = center_y[0][idxs]
    center_x = center_x[0][idxs]
    l = l[0][idxs]
    w = w[0][idxs]
    yaw = yaw[0][idxs]

    if corners.shape[0] == 0:
        return np.array([])
    
    selected_idxs = non_max_suppression(corners, scores, nms_threshold) 
    boxes = np.stack([cls[selected_idxs], 
                      scores[selected_idxs], 
                      center_x[selected_idxs], 
                      center_y[selected_idxs], 
                      l[selected_idxs], 
                      w[selected_idxs], 
                      yaw[selected_idxs]])
    boxes = np.swapaxes(boxes, 0, 1)

    return boxes

def filter_pred_nms_free(reg_pred, cls_pred, config, score_threshold, nms_threshold):
    geometry = config["geometry"]

    ratio = 4
    reg_pred = reg_pred[0].detach()
    cls_pred = cls_pred[0].detach()
    cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(reg_pred, 6, dim=0)

    cls_probs, cls_ids = torch.max(cls_pred, dim = 0)

    idxs = torch.logical_or(cls_probs < score_threshold, cls_ids == 0)
    # make background and low probability detections 0 
    cls_probs[idxs] = 0
   
    pooled = F.max_pool2d(cls_probs.unsqueeze(0), 3, 1, 1).squeeze()
    selected_idxs = torch.logical_and(cls_probs == pooled, cls_probs > 0)

    y = torch.arange(geometry["label_shape"][0])
    x = torch.arange(geometry["label_shape"][1])

    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xx = xx.to(reg_pred.device)
    yy = yy.to(reg_pred.device)

    center_y = dy + yy * ratio * geometry["y_res"] + geometry["y_min"]
    center_x = dx + xx * ratio * geometry["x_res"] + geometry["x_min"]
    center_x = center_x.squeeze()
    center_y = center_y.squeeze()
    l = torch.exp(log_l).squeeze()
    w = torch.exp(log_w).squeeze()
    yaw2 = torch.atan2(sin_t, cos_t).squeeze()
    yaw = yaw2 / 2


    boxes = np.stack([cls_ids[selected_idxs].numpy(), 
                      cls_probs[selected_idxs].numpy(), 
                      center_x[selected_idxs].numpy(), 
                      center_y[selected_idxs].numpy(), 
                      l[selected_idxs].numpy(), 
                      w[selected_idxs].numpy(), 
                      yaw[selected_idxs].numpy()])
    #print(boxes.shape)
    boxes = np.swapaxes(boxes, 0, 1)

    return boxes

def filter_pred_afdet(cls_pred, offset_pred, size_pred, yaw_pred, config, thres):
    geometry = config["geometry"]

    offset_pred = offset_pred[0].detach()
    size_pred = size_pred[0].detach()
    yaw_pred = yaw_pred[0].detach()
    cls_pred = cls_pred[0].detach()

    cos_t, sin_t = torch.chunk(yaw_pred, 2, dim = 0)
    dx, dy = torch.chunk(offset_pred, 2, dim = 0)
    log_w, log_l = torch.chunk(size_pred, 2, dim = 0)

    cls_probs, cls_ids = torch.max(cls_pred, dim = 0)

   
    pooled = F.max_pool2d(cls_probs.unsqueeze(0), 3, 1, 1).squeeze()
    selected_idxs = torch.logical_and(cls_probs == pooled, cls_probs > thres)

    y = torch.arange(geometry["input_shape"][0])
    x = torch.arange(geometry["input_shape"][1])

    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xx = xx.to(offset_pred.device)
    yy = yy.to(offset_pred.device)

    center_y = dy + yy *  geometry["y_res"] + geometry["y_min"]
    center_x = dx + xx *  geometry["x_res"] + geometry["x_min"]
    center_x = center_x.squeeze()
    center_y = center_y.squeeze()
    l = torch.exp(log_l).squeeze()
    w = torch.exp(log_w).squeeze()
    yaw2 = torch.atan2(sin_t, cos_t).squeeze()
    yaw = yaw2 / 2


    boxes = np.stack([cls_ids[selected_idxs].numpy(), 
                      cls_probs[selected_idxs].numpy(), 
                      center_x[selected_idxs].numpy(), 
                      center_y[selected_idxs].numpy(), 
                      l[selected_idxs].numpy(), 
                      w[selected_idxs].numpy(), 
                      yaw[selected_idxs].numpy()])
    #print(boxes.shape)
    boxes = np.swapaxes(boxes, 0, 1)

    return boxes


def filter_pred_aux(reg_pred, cls_pred, occupancy_pred, config, score_threshold, occupancy_pow, nms_threshold):
    geometry = config["geometry"]

    ratio = 4
    reg_pred = reg_pred[0]
    cls_pred = cls_pred[0]
    cos_t, sin_t, dx, dy, log_w, log_l = np.split(reg_pred, 6, axis=0)


    cls_probs = np.max(cls_pred, axis = 0)
    cls_probs = cls_probs * np.float_power(occupancy_pred, occupancy_pow)
    cls_ids = np.argmax(cls_pred, axis = 0)

    if isinstance(score_threshold, list):
        thres = np.array(score_threshold)
        thres_arr = np.repeat(np.expand_dims(thres, 1), cls_pred.shape[1], 1)
        thres_arr = np.repeat(np.expand_dims(thres_arr, 2), cls_pred.shape[2], 2)

        weights = np.repeat(np.expand_dims(np.float_power(occupancy_pred, occupancy_pow), 0), cls_pred.shape[0], 0)
        comparisons = cls_pred * weights > thres_arr
        idxs = np.logical_and(np.take_along_axis(comparisons, cls_ids.reshape((1, cls_ids.shape[0], cls_ids.shape[1])), axis = 0).squeeze(), cls_ids != 0)
    else:
        idxs = np.logical_and(cls_probs > score_threshold, cls_ids != 0)
    cls = cls_ids[idxs]
    scores = cls_probs[idxs]
    y = np.arange(geometry["label_shape"][0])
    x = np.arange(geometry["label_shape"][1])
    xx, yy = np.meshgrid(x, y)


    center_y = dy + yy * ratio * geometry["y_res"] + geometry["y_min"]
    center_x = dx + xx * ratio * geometry["x_res"] + geometry["x_min"]
    l = np.exp(log_l)
    w = np.exp(log_w)
    yaw2 = np.arctan2(sin_t, cos_t)
    yaw = yaw2 / 2
    
    cos_t = np.cos(yaw)
    sin_t = np.sin(yaw)

    rear_left_x = center_x - l/2 * cos_t - w/2 * sin_t
    rear_left_y = center_y - l/2 * sin_t + w/2 * cos_t
    rear_right_x = center_x - l/2 * cos_t + w/2 * sin_t
    rear_right_y = center_y - l/2 * sin_t - w/2 * cos_t
    front_right_x = center_x + l/2 * cos_t + w/2 * sin_t
    front_right_y = center_y + l/2 * sin_t - w/2 * cos_t
    front_left_x = center_x + l/2 * cos_t - w/2 * sin_t
    front_left_y = center_y + l/2 * sin_t + w/2 * cos_t

    decoded_reg = np.concatenate([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                               front_right_x, front_right_y, front_left_x, front_left_y], axis=0)
    decoded_reg = np.swapaxes(decoded_reg, 0, 1)
    decoded_reg = np.swapaxes(decoded_reg, 1, 2)
    decoded_reg = decoded_reg[idxs]
    corners = np.reshape(decoded_reg, (-1, 4, 2))

    center_y = center_y[0][idxs]
    center_x = center_x[0][idxs]
    l = l[0][idxs]
    w = w[0][idxs]
    yaw = yaw[0][idxs]

    if corners.shape[0] == 0:
        return np.array([]), np.array([])
    
    selected_idxs = non_max_suppression(corners, scores, nms_threshold) 
    boxes = np.stack([cls[selected_idxs], 
                      scores[selected_idxs], 
                      center_x[selected_idxs], 
                      center_y[selected_idxs], 
                      l[selected_idxs], 
                      w[selected_idxs], 
                      yaw[selected_idxs]])
    boxes = np.swapaxes(boxes, 0, 1)
    return boxes, None
    weighted_cls_probs = cls_probs
    cls_probs = np.max(cls_pred, axis = 0)
    cls_ids = np.argmax(cls_pred, axis = 0)

    if isinstance(score_threshold, list):
        thres = np.array(score_threshold)
        thres_arr = np.repeat(np.expand_dims(thres, 1), cls_pred.shape[1], 1)
        thres_arr = np.repeat(np.expand_dims(thres_arr, 2), cls_pred.shape[2], 2)

        comparisons = cls_pred > thres_arr
        weights = np.repeat(np.expand_dims(np.float_power(occupancy_pred, occupancy_pow), 0), cls_pred.shape[0], 0)
        weighted_comparisons = cls_pred * weights < thres_arr
        idxs = np.logical_and(np.logical_and(np.take_along_axis(comparisons, cls_ids.reshape((1, cls_ids.shape[0], cls_ids.shape[1])), axis = 0).squeeze(), 
                                    cls_ids != 0), 
                             np.take_along_axis(weighted_comparisons, cls_ids.reshape((1, cls_ids.shape[0], cls_ids.shape[1])), axis = 0).squeeze())
    else:
        idxs = np.logical_and(np.logical_and(cls_probs > score_threshold, weighted_cls_probs < score_threshold), cls_ids != 0)
    cls = cls_ids[idxs]
    scores = weighted_cls_probs[idxs]
    y = np.arange(geometry["label_shape"][0])
    x = np.arange(geometry["label_shape"][1])
    xx, yy = np.meshgrid(x, y)


    center_y = dy + yy * ratio * geometry["y_res"] + geometry["y_min"]
    center_x = dx + xx * ratio * geometry["x_res"] + geometry["x_min"]
    l = np.exp(log_l)
    w = np.exp(log_w)
    yaw2 = np.arctan2(sin_t, cos_t)
    yaw = yaw2 / 2
    
    cos_t = np.cos(yaw)
    sin_t = np.sin(yaw)

    rear_left_x = center_x - l/2 * cos_t - w/2 * sin_t
    rear_left_y = center_y - l/2 * sin_t + w/2 * cos_t
    rear_right_x = center_x - l/2 * cos_t + w/2 * sin_t
    rear_right_y = center_y - l/2 * sin_t - w/2 * cos_t
    front_right_x = center_x + l/2 * cos_t + w/2 * sin_t
    front_right_y = center_y + l/2 * sin_t - w/2 * cos_t
    front_left_x = center_x + l/2 * cos_t - w/2 * sin_t
    front_left_y = center_y + l/2 * sin_t + w/2 * cos_t

    decoded_reg = np.concatenate([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                               front_right_x, front_right_y, front_left_x, front_left_y], axis=0)
    decoded_reg = np.swapaxes(decoded_reg, 0, 1)
    decoded_reg = np.swapaxes(decoded_reg, 1, 2)
    decoded_reg = decoded_reg[idxs]
    corners = np.reshape(decoded_reg, (-1, 4, 2))

    center_y = center_y[0][idxs]
    center_x = center_x[0][idxs]
    l = l[0][idxs]
    w = w[0][idxs]
    yaw = yaw[0][idxs]

    if corners.shape[0] == 0:
        return boxes, np.array([])
    
    selected_idxs = non_max_suppression(corners, scores, nms_threshold) 
    ignored_boxes = np.stack([cls[selected_idxs], 
                      scores[selected_idxs], 
                      center_x[selected_idxs], 
                      center_y[selected_idxs], 
                      l[selected_idxs], 
                      w[selected_idxs], 
                      yaw[selected_idxs]])
    ignored_boxes = np.swapaxes(ignored_boxes, 0, 1)

    return boxes, ignored_boxes



if __name__ == "__main__":
    geom = {
        "L1": -40.0,
        "L2": 40.0,
        "W1": 0.0,
        "W2": 70.0,
        "H1": -2.5,
        "H2": 1.0,
        "input_shape": [800, 700, 35],
        "label_shape": [200, 175, 7]
    }

    pointcloud_folder = "/home/stpc/data/kitti/velodyne/training_reduced/velodyne"
    label_folder = "/home/stpc/data/kitti/label_2/training/label_2_reduced"
    data_file = "/home/stpc/data/train/train.txt"

    with open("/home/stpc/proj/object_detection/configs/base.json", 'r') as f:
        config = json.load(f)


    dataset = KittiDataset(pointcloud_folder, label_folder, data_file, config["data"]["kitti"])
    data_loader = DataLoader(dataset, shuffle=True, batch_size=2)

    net = PIXOR(geom, use_bn=False)
    reg_loss = SmoothL1Loss()

    for data in data_loader:
        #cls_one_hot = one_hot(data["cls_map"], num_classes= 4 , device="cpu", dtype=data["cls_map"].dtype)
        #filter_pred(data["reg_map"].numpy(), cls_one_hot.numpy(), geom)
        #loss = reg_loss(data["reg_map"], data["reg_map"], data["cls_map"])
        #print(loss)
        pred = net(data["voxel"])
        print(pred["cls_map"].shape)
        #imgplot = plt.imshow(torch.squeeze(mask).permute(0, 1))
        #plt.show()
        #preds = net(data["voxel"])
        #print(preds["reg_map"].shape)
        #print(data["reg_map"].shape)
        break