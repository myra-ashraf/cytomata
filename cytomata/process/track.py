import os
import time

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def bbox_to_state(bbox):
    """
    Converts bounding box coordinates to Kalman state vector representation.

    Args:
        bbox (tuple): Bounding box coordinates [x1, y1, x2, y2]
            (x1, y1): coordinate of top-left corner
            (x2, y2): coordinate of bottom-right corner

    Returns:
        state (4x1 ndarray): Kalman state vector [x, y, s, r].
            (x, y): center coordinate
            s: scale/area
            r: aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w*h
    r = w/h
    return np.array([x, y, s, r]).reshape((4, 1))


def state_to_bbox(state):
    """
    Converts Kalman state vector to bounding box coordinate representation.

    Args:
        state (4x1 ndarray): Kalman state vector [x, y, s, r].
            (x, y): center coordinate
            s: scale/area
            r: aspect ratio
    Returns:
        bbox (tuple): Bounding box coordinates [x1, y1, x2, y2]
            (x1, y1): coordinate of top-left corner
            (x2, y2): coordinate of bottom-right corner
    """
    w = np.sqrt(state[2] * state[3])
    h = state[2] / w
    x1 = state[0] - w/2.
    y1 = state[1] - h/2.
    x2 = state[0] + w/2.
    y2 = state[1] + h/2.
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


def iou(bb_pr, bb_gt):
    """
    Computes intersection over union (IOU) between two bounding boxes.

    Args:
        bb_pr (tuple): Predicted bounding box [x1, y1, x2, y2]
        bb_gt (tuple): Ground truth bounding box [x1, y1, x2, y2]

    Returns:
        iou (float): Evaluation metric
    """
    xx1 = np.maximum(bb_pr[0], bb_gt[0])
    yy1 = np.maximum(bb_pr[1], bb_gt[1])
    xx2 = np.minimum(bb_pr[2], bb_gt[2])
    yy2 = np.minimum(bb_pr[3], bb_gt[3])
    inter_area = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
    bb_pr_area = (bb_pr[2] - bb_pr[0]) * (bb_pr[3] - bb_pr[1])
    bb_gt_area = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union_area = bb_pr_area + bb_gt_area - inter_area
    return inter_area / union_area


def assign(dets, trks, iou_threshold=0.3):
    """
    Assigns detections to tracked objects.

    Args:
        dets (tuple): Bounding boxes of detections [x1, y1, x2, y2]
        trks (tuple): Bounding boxes of tracks [x1, y1, x2, y2]

    Returns:
        matched (Nx2 ndarray): Indices of matched assignments
        unmatched_dets (1xN ndarray): Indices of detections
        unmatched_trks (1xN ndarray): Indices of tracks
    """
    if len(trks) == 0:
        matched = np.empty((0, 2), dtype=int)
        unmatched_dets = np.arange(len(dets))
        unmatched_trks = np.empty((0, 5),dtype=int)
        return matched, unmatched_dets, unmatched_trks
    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
          iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)
    unmatched_dets = [d for d, det in enumerate(dets) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t, trk in enumerate(trks) if t not in matched_indices[:, 1]]
    # filter out matched with low IOU
    matched = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matched.append(m.reshape(1, 2))
    if len(matched) == 0:
        matched = np.empty((0, 2),dtype=int)
    else:
        matched = np.concatenate(matched, axis=0)
    return matched, np.array(unmatched_dets), np.array(unmatched_trks)


class KalmanBoxTracker(object):
    """
    Internal state of individual tracked objects.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        """
        # define constant velocity model
        # dim_x = number of state variables
        # dim_z = number of measurement/input variables
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # F = state transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]
        ])
        # H = measurement function
        self.kf.H = np.array([
            [1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]
        ])
        # R = measurement uncertainty
        self.kf.R[2:,2:] *= 10.
        # P = covariance matrix
        self.kf.P[4:,4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # Q = process unvertainty
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        # x = state estimate
        self.kf.x[:4] = bbox_to_state(bbox)
        #
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_state(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(state_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return state_to_bbox(self.kf.x)


class Sort(object):
    """"""
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
          pos = self.trackers[t].predict()[0]
          trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
          if(np.any(np.isnan(pos))):
            to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
          self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = assign(dets,trks)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
          if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0],0]
            trk.update(dets[d,:][0])

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
              ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
              self.trackers.pop(i)
        if(len(ret)>0):
          return np.concatenate(ret)
        return np.empty((0,5))
