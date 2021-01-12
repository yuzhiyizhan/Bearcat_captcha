import io
import re
import os
import cv2
import sys
import math
import time
import json
import glob
import base64
import shutil
import argparse
import colorsys
import operator
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib as plt
from loguru import logger
from PIL import ImageDraw
from PIL import ImageFont
import xml.etree.ElementTree as ET
from works.simples.settings import PHI
from works.simples.settings import MODE
from works.simples.settings import MODEL
from works.simples.settings import PRUNING
from works.simples.settings import USE_GPU
from works.simples.settings import GT_PATH
from works.simples.settings import DR_PATH
from works.simples.settings import IMG_PATH
from works.simples.settings import MODEL_PATH
from works.simples.settings import MODEL_NAME
from works.simples.settings import PRUNING_MODEL_NAME
from works.simples.utils import Image_Processing
from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as KT
from works.simples.models import Models
from works.simples.models import YOLO_anchors
from works.simples.models import Efficientdet_anchors
from works.simples.settings import THRESHOLD
from works.simples.settings import TEST_PATH
from works.simples.settings import LABEL_PATH
from works.simples.settings import IMAGE_WIDTH
from works.simples.settings import OUTPUT_PATH
from works.simples.settings import IMAGE_SIZES
from works.simples.settings import IMAGE_HEIGHT
from works.simples.settings import NUMBER_CLASSES_FILE
from works.simples.settings import IMAGE_CHANNALS
from concurrent.futures import ThreadPoolExecutor

table = []
for i in range(256):
    if i < THRESHOLD:
        table.append(0)
    else:
        table.append(255)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

if MODE == 'EFFICIENTDET' or MODE == 'SSD':
    pass
else:
    raise ValueError('不是目标检测任务无法测试Map')

if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    if gpus:
        logger.info("use gpu device")
        logger.info(f'可用GPU数量: {len(gpus)}')
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            logger.error(e)
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(device=gpu, enable=True)
                tf.print(gpu)
        except RuntimeError as e:
            logger.error(e)
    else:
        tf.config.experimental.list_physical_devices(device_type="CPU")
        os.environ["CUDA_VISIBLE_DEVICE"] = "-1"
        logger.info("not found gpu device,convert to use cpu")
else:
    logger.info("use cpu device")
    # 禁用gpu
    tf.config.experimental.list_physical_devices(device_type="CPU")
    os.environ["CUDA_VISIBLE_DEVICE"] = "-1"


class EfficientDet_BBoxUtility(object):
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5, ignore_threshold=0.4,
                 nms_thresh=0.3, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k

    def _iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
        return iou

    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]
        # 逆向编码，将真实框转化为efficientdet预测结果的格式

        # 先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 逆向求取efficientdet应该有的预测结果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        return encoded_box.ravel()

    def ignore_box(self, box):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        ignored_box[:, 0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_priors, 4 + 1 + self.num_classes + 1))
        assignment[:, 4] = 0.0
        assignment[:, -1] = 0.0
        if len(boxes) == 0:
            return assignment
        # 对每一个真实框都进行iou计算
        ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        # 取重合程度最大的先验框，并且获取这个先验框的index
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        # (num_priors)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        # (num_priors)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1
        assignment[:, -1][ignore_iou_mask] = -1

        # (n, num_priors, 5)
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        # (n, num_priors)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合程度最大的先验框，并且获取这个先验框的index
        # (num_priors)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # (num_priors)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # (num_priors)
        best_iou_mask = best_iou > 0
        # 某个先验框它属于哪个真实框
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        # 哪些先验框存在真实框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4代表为背景的概率，为0
        assignment[:, 4][best_iou_mask] = 1
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的

        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox):
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height
        decode_bbox_center_y += prior_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3])
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)

        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, mbox_priorbox, confidence_threshold=0.4):
        # print(predictions)
        # 网络预测的结果
        mbox_loc = predictions[0]
        # 置信度
        mbox_conf = predictions[1]
        # 先验框
        mbox_priorbox = mbox_priorbox

        results = []
        # 对每一个图片进行处理
        for i in range(len(mbox_loc)):
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)

            bs_class_conf = mbox_conf[i]

            class_conf = np.expand_dims(np.max(bs_class_conf, 1), -1)
            class_pred = np.expand_dims(np.argmax(bs_class_conf, 1), -1)

            conf_mask = (class_conf >= confidence_threshold)[:, 0]

            detections = np.concatenate((decode_bbox[conf_mask], class_conf[conf_mask], class_pred[conf_mask]), 1)
            unique_class = np.unique(detections[:, -1])

            best_box = []
            if len(unique_class) == 0:
                results.append(best_box)
                continue
            # 4、对种类进行循环，
            # 非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
            # 对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
            for c in unique_class:
                cls_mask = detections[:, -1] == c

                detection = detections[cls_mask]
                scores = detection[:, 4]
                # 5、根据得分对该种类进行从大到小排序。
                arg_sort = np.argsort(scores)[::-1]
                detection = detection[arg_sort]
                while np.shape(detection)[0] > 0:
                    # 6、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                    best_box.append(detection[0])
                    if len(detection) == 1:
                        break
                    ious = self._iou(best_box[-1], detection[1:])
                    detection = detection[1:][ious < self._nms_thresh]
            results.append(best_box)
        # 获得，在所有预测结果里面，置信度比较高的框
        # 还有，利用先验框和efficientdet的预测结果，处理获得了真实框（预测框）的位置
        return results


class SSD_BBoxUtility(object):
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k

    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]
        # 逆向编码，将真实框转化为ssd预测结果的格式

        # 先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 逆向求取ssd应该有的预测结果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        # 除以0.1
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        # 除以0.2
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合程度最大的先验框，并且获取这个先验框的index
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4代表为背景的概率，为0
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
        decode_bbox_center_y += prior_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.5):
        # 网络预测的结果
        mbox_loc = predictions[:, :, :4]
        # 0.1，0.1，0.2，0.2
        variances = predictions[:, :, -4:]
        # 先验框
        mbox_priorbox = predictions[:, :, -8:-4]
        # 置信度
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        # 对每一个特征层进行处理
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox[i], variances[i])

            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence_threshold的框
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    # 进行iou的非极大抑制
                    idx = tf.image.non_max_suppression(tf.cast(boxes_to_process, tf.float32),
                                                       tf.cast(confs_to_process, tf.float32),
                                                       self._top_k,
                                                       iou_threshold=self._nms_thresh).numpy()
                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    # 将label、置信度、框的位置进行堆叠。
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    # 添加进result里
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                # 按照置信度进行排序
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                # 选出置信度最大的keep_top_k个
                results[-1] = results[-1][:keep_top_k]
        # 获得，在所有预测结果里面，置信度比较高的框
        # 还有，利用先验框和ssd的预测结果，处理获得了真实框（预测框）的位置
        return results


class Predict_Image(object):
    def __init__(self, model_path=None, app=False, classification=False, detection_results=False):
        self.iou = 0.3
        self.app = app
        self.confidence = 0.4
        self.ssdconfidence = 0.5
        self.model_path = model_path
        self.load_model()
        self.classification = classification
        self.detection_results = detection_results

        if MODE == 'EFFICIENTDET':
            self.prior = self._get_prior()
        elif MODE == 'YOLO' or MODE == 'YOLO_TINY':
            tf.compat.v1.disable_eager_execution()
            self.score = 0.5
            self.sess = KT.get_session()
            self.anchors = YOLO_anchors.get_anchors()

    def load_model(self):
        if PRUNING:
            self.model = tf.lite.Interpreter(model_path=self.model_path)
            self.model.allocate_tensors()
        else:
            self.model = operator.methodcaller(MODEL)(Models)
            self.model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        logger.debug('加载模型到内存')
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            result = f.read()
        self.num_classes_dict = json.loads(result)
        self.num_classes_list = list(json.loads(result).values())
        self.num_classes = len(self.num_classes_list)
        if MODE == 'EFFICIENTDET' or MODE == 'SSD':
            if MODE == 'EFFICIENTDET':
                self.bbox_util = EfficientDet_BBoxUtility(self.num_classes, nms_thresh=self.iou)
            elif MODE == 'SSD':
                self.bbox_util = SSD_BBoxUtility(self.num_classes)
            # 画框设置不同的颜色
            hsv_tuples = [(x / self.num_classes, 1., 1.)
                          for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))
        elif MODE == 'YOLO' or MODE == 'YOLO_TINY':
            num_classes = len(self.num_classes_list)
            # 画框设置不同的颜色
            hsv_tuples = [(x / self.num_classes, 1., 1.)
                          for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

            # 打乱颜色
            np.random.seed(10101)
            np.random.shuffle(self.colors)
            np.random.seed(None)
            self.input_image_shape = K.placeholder(shape=(2,))

            self.boxes, self.scores, self.classes = YOLO_anchors.yolo_eval(self.model.output, self.anchors,
                                                                           num_classes, self.input_image_shape,
                                                                           score_threshold=self.score,
                                                                           iou_threshold=self.iou)

    def preprocess_input(self, image):
        image /= 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std
        return image

    def letterbox_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        if MODE == 'EFFICIENTDET':
            new_image = Image.new('RGB', size, (0, 0, 0))
        elif MODE == 'YOLO' or MODE == 'YOLO_TINY':
            new_image = Image.new('RGB', size, (128, 128, 128))
        elif MODE == 'SSD':
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            x_offset, y_offset = (w - nw) // 2 / 300, (h - nh) // 2 / 300
            return new_image, x_offset, y_offset
        else:
            raise ValueError('new_image error')
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def ssd_correct_boxes(self, top, left, bottom, right, input_shape, image_shape):
        new_shape = image_shape * np.min(input_shape / image_shape)

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1)
        box_hw = np.concatenate((bottom - top, right - left), axis=-1)

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[:, 0:1],
            box_mins[:, 1:2],
            box_maxes[:, 0:1],
            box_maxes[:, 1:2]
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def efficientdet_correct_boxes(self, top, left, bottom, right, input_shape, image_shape):
        new_shape = image_shape * np.min(input_shape / image_shape)

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1)
        box_hw = np.concatenate((bottom - top, right - left), axis=-1)

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[:, 0:1],
            box_mins[:, 1:2],
            box_maxes[:, 0:1],
            box_maxes[:, 1:2]
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def _get_prior(self):
        return Efficientdet_anchors.get_anchors(IMAGE_SIZES[PHI])

    def decode_image(self, image):
        if self.app:
            image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            iw, ih = image.size
            w, h = IMAGE_WIDTH, IMAGE_HEIGHT
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            if IMAGE_CHANNALS == 3:
                new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
                new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            else:
                new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
                new_image = new_image.convert('L')
                new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
                new_image = new_image.point(table, 'L')
            image = np.array(new_image, dtype=np.float32)
            image = np.expand_dims(image, axis=0)
            image = image / 255.
            return image
        else:
            with open(image, 'rb') as image_file:
                image = Image.open(image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                iw, ih = image.size
                w, h = IMAGE_WIDTH, IMAGE_HEIGHT
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)
                image = image.resize((nw, nh), Image.BICUBIC)
                if IMAGE_CHANNALS == 3:
                    new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
                    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
                else:
                    new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
                    new_image = new_image.convert('L')
                    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
                    new_image = new_image.point(table, 'L')
                image = np.array(new_image, dtype=np.float32)
                image = np.expand_dims(image, axis=0)
                image = image / 255.
                image_file.close()
                return image

    def decode_label(self, image):
        path, label = os.path.split(image)
        label, suffix = os.path.splitext(label)
        label = re.split('_', label)[0]
        return label

    def recognition_probability(self, recognition_rate_liat):
        mean_section = np.mean(recognition_rate_liat)
        std_section = np.std(recognition_rate_liat)
        sqrt_section = np.sqrt(len(recognition_rate_liat))
        min_confidence = mean_section - (2.58 * (std_section / sqrt_section))
        return min_confidence

    def decode_vector(self, vector, num_classes):

        text_list = []
        recognition_rate_liat = []
        if MODE == 'ORDINARY':
            vector = vector[0]
            for i in vector:
                text = num_classes.get(str(np.argmax(i)))
                if text:
                    text_list.append(text)
            text = ''.join(text_list)
            for i in vector:
                recognition_rate = np.max(i) / np.sum(np.abs(i))
                recognition_rate_liat.append(recognition_rate)
            recognition_rate = self.recognition_probability(recognition_rate_liat)
            return text, recognition_rate
        elif MODE == 'NUM_CLASSES':
            vector = vector[0]
            text = np.argmax(vector)
            text = num_classes.get(str(text))
            recognition_rate = np.max(vector) / np.sum(np.abs(vector))
            return text, recognition_rate
        elif MODE == 'CTC':
            vector = vector[0]
            for i in vector:
                text = num_classes.get(str(np.argmax(i)))
                if text:
                    text_list.append(text)
            text = ''.join(text_list)
            for i in vector:
                recognition_rate_liat = [np.max(r) / np.sum(np.abs(r)) for r in i]
            recognition_rate = np.abs(self.recognition_probability(recognition_rate_liat))
            return text, recognition_rate
        elif MODE == 'CTC_TINY':
            # for i in vector[0]:
            #     texts = num_classes.get(str(np.argmax(i)))
            #     if texts:
            #         text_list.append(texts)
            # text = ''.join(text_list)
            out = K.get_value(
                K.ctc_decode(vector, input_length=np.ones(vector.shape[0]) * vector.shape[1], greedy=True)[0][0])
            text = ''.join([num_classes.get(str(x), '') for x in out[0]])
            for i in vector:
                recognition_rate_liat = [np.max(r) / np.sum(np.abs(r)) for r in i]
            recognition_rate = self.recognition_probability(recognition_rate_liat)
            return text, recognition_rate
        else:
            raise ValueError(f'还没写{MODE}这种预测方法')

    def predict_image(self, image_path):
        global right_value
        global predicted_value
        start_time = time.time()
        recognition_rate_list = []
        if MODE == 'EFFICIENTDET':
            if self.app:
                image = Image.fromarray(image_path)
            else:
                image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_shape = np.array(np.shape(image)[0:2])

            crop_img = self.letterbox_image(image, [IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]])
            photo = np.array(crop_img, dtype=np.float32)

            # 图片预处理，归一化
            photo = np.reshape(self.preprocess_input(photo),
                               [1, IMAGE_SIZES[PHI], IMAGE_SIZES[PHI], 3])
            if PRUNING:
                model = self.model
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                model.set_tensor(input_details[0]['index'], photo)
                model.invoke()
                pred1 = model.get_tensor(output_details[0]['index'])
                pred2 = model.get_tensor(output_details[1]['index'])
                preds = (pred2, pred1)
            else:
                preds = self.model.predict(photo)
            # 将预测结果进行解码
            results = self.bbox_util.detection_out(preds, self.prior, confidence_threshold=self.confidence)

            if len(results[0]) <= 0:
                if self.detection_results:
                    _, file_name = os.path.split(image_path)
                    file_name, _ = os.path.splitext(file_name)
                    with open(os.path.join(DR_PATH, file_name + '.txt'), mode='a', encoding='utf-8') as f:
                        f.write(f'Fail {0} {0} {0} {0} {0}\n')
                return image
            results = np.array(results)

            # 筛选出其中得分高于confidence的框
            det_label = results[0][:, 5]
            det_conf = results[0][:, 4]
            det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:,
                                                                                                           3]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
                det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(
                det_ymax[top_indices],
                -1)

            # 去掉灰条
            boxes = self.efficientdet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                    np.array([IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]]),
                                                    image_shape)

            font = ImageFont.truetype(font='simhei.ttf',
                                      size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

            thickness = (np.shape(image)[0] + np.shape(image)[1]) // IMAGE_SIZES[PHI]

            for i, c in enumerate(top_label_indices):
                predicted_class = self.num_classes_list[int(c)]
                score = top_conf[i]
                recognition_rate_list.append(score)
                top, left, bottom, right = boxes[i]
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
                if self.classification:
                    image_crop = image.crop((left, top, right, bottom))
                    image_bytearr = io.BytesIO()
                    image_crop.save(image_bytearr, format='JPEG')
                    image_bytes = image_bytearr.getvalue()
                    data = {'data': [f'data:image;base64,{base64.b64encode(image_bytes).decode()}']}
                    response = requests.post('http://127.0.0.1:7860/api/predict/', json=data).json()
                    result = json.loads(response.get('data')[0].get('label'))
                    predicted_class = result.get('result')
                    recognition_rate = result.get('recognition_rate')
                    recognition_rate = float(recognition_rate.replace('%', '')) / 100
                    recognition_rate_list.append(recognition_rate)
                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                if self.detection_results:
                    _, file_name = os.path.split(image_path)
                    file_name, _ = os.path.splitext(file_name)
                    with open(os.path.join(DR_PATH, file_name + '.txt'), mode='a', encoding='utf-8') as f:
                        f.write(f'{predicted_class} {score} {left} {top} {right} {bottom}\n')
                logger.info(label)
                label = label.encode('utf-8')
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[int(c)])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[int(c)])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            end_time = time.time()
            logger.info(f'识别时间为{end_time - start_time}s')
            logger.info(f'总体置信度为{round(self.recognition_probability(recognition_rate_list), 2) * 100}%')
            return image

        elif MODE == 'YOLO' or MODE == 'YOLO_TINY':
            if self.app:
                image = Image.fromarray(image_path)
            else:
                image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            new_image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
            boxed_image = self.letterbox_image(image, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    KT.learning_phase(): 0
                })

            # logger.debug('Found {} boxes for {}'.format(len(out_boxes), 'img'))
            # 设置字体
            font = ImageFont.truetype(font='simhei.ttf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i, c in list(enumerate(out_classes)):
                predicted_class = self.num_classes_list[c]
                box = out_boxes[i]
                score = out_scores[i]
                recognition_rate_list.append(score)
                top, left, bottom, right = box
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                logger.debug(label)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            end_time = time.time()
            logger.info(f'识别时间为{end_time - start_time}s')
            logger.info(f'总体置信度为{round(self.recognition_probability(recognition_rate_list), 2) * 100}%')
            return image

        elif MODE == 'SSD':
            if self.app:
                image = Image.fromarray(image_path)
            else:
                image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_shape = np.array(np.shape(image)[0:2])
            crop_img, x_offset, y_offset = self.letterbox_image(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
            photo = np.array(crop_img, dtype=np.float64)
            photo = tf.keras.applications.imagenet_utils.preprocess_input(
                np.reshape(photo, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]))
            preds = self.model(photo).numpy()
            results = self.bbox_util.detection_out(preds, confidence_threshold=self.ssdconfidence)
            if len(results[0]) <= 0:
                if self.detection_results:
                    _, file_name = os.path.split(image_path)
                    file_name, _ = os.path.splitext(file_name)
                    with open(os.path.join(DR_PATH, file_name + '.txt'), mode='a', encoding='utf-8') as f:
                        f.write(f'Fail {0} {0} {0} {0} {0}\n')
                return image
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:,
                                                                                                           5]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.ssdconfidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
                det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(
                det_ymax[top_indices],
                -1)
            boxes = self.ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array([IMAGE_HEIGHT, IMAGE_WIDTH]), image_shape)
            font = ImageFont.truetype(font='simhei.ttf',
                                      size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

            thickness = (np.shape(image)[0] + np.shape(image)[1]) // IMAGE_HEIGHT
            for i, c in enumerate(top_label_indices):
                predicted_class = self.num_classes_list[int(c)]
                score = top_conf[i]
                recognition_rate_list.append(score)
                top, left, bottom, right = boxes[i]
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
                if self.classification:
                    image_crop = image.crop((left, top, right, bottom))
                    image_bytearr = io.BytesIO()
                    image_crop.save(image_bytearr, format='JPEG')
                    image_bytes = image_bytearr.getvalue()
                    data = {'data': [f'data:image;base64,{base64.b64encode(image_bytes).decode()}']}
                    response = requests.post('http://127.0.0.1:7860/api/predict/', json=data).json()
                    result = json.loads(response.get('data')[0].get('label'))
                    predicted_class = result.get('result')
                    recognition_rate = result.get('recognition_rate')
                    recognition_rate = float(recognition_rate.replace('%', '')) / 100
                    recognition_rate_list.append(recognition_rate)
                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                if self.detection_results:
                    _, file_name = os.path.split(image_path)
                    file_name, _ = os.path.splitext(file_name)
                    with open(os.path.join(DR_PATH, file_name + '.txt'), mode='a', encoding='utf-8') as f:
                        f.write(f'{predicted_class} {score} {left} {top} {right} {bottom}\n')
                logger.info(label)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[int(c)])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[int(c)])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            end_time = time.time()
            logger.info(f'识别时间为{end_time - start_time}s')
            logger.info(f'总体置信度为{round(self.recognition_probability(recognition_rate_list), 2) * 100}%')
            return image

        else:
            if PRUNING:
                model = self.model
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                image_object = self.decode_image(image_path)
                model.set_tensor(input_details[0]['index'], image_object)
                model.invoke()
                vertor = model.get_tensor(output_details[0]['index'])
            else:
                model = self.model
                image_object = self.decode_image(image_path)
                vertor = model.predict(image_object)
            text, recognition_rate = self.decode_vector(vector=vertor, num_classes=self.num_classes_dict)
            right_text = self.decode_label(image_path)
            logger.info(f'预测为{text},真实为{right_text}') if text == right_text else logger.error(
                f'预测为{text},真实为{right_text}')
            logger.info(f'识别率为:{recognition_rate * 100}%') if recognition_rate > 0.7 else logger.error(
                f'识别率为:{recognition_rate * 100}%')
            if str(text) != str(right_text):
                logger.error(f'预测失败的图片路径为:{image_path}')
                right_value = right_value + 1
                logger.info(f'正确率:{(predicted_value / right_value) * 100}%')
                if predicted_value > 0:
                    logger.info(f'预测正确{predicted_value}张图片')
            else:
                predicted_value = predicted_value + 1
                right_value = right_value + 1
                logger.info(f'正确率:{(predicted_value / right_value) * 100}%')
                if predicted_value > 0:
                    logger.info(f'预测正确{predicted_value}张图片')
            end_time = time.time()
            logger.info(f'已识别{right_value}张图片')
            logger.info(f'识别时间为{end_time - start_time}s')
            # return Image.fromarray(image_object[0] * 255)

    def close_session(self):
        if MODE == 'YOLO' or MODE == 'YOLO_TINY':
            self.sess.close()

    def api(self, image):

        if MODE == 'EFFICIENTDET':
            result_list = []
            recognition_rate_list = []
            start_time = time.time()
            image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_shape = np.array(np.shape(image)[0:2])

            crop_img = self.letterbox_image(image, [IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]])
            photo = np.array(crop_img, dtype=np.float32)
            # 图片预处理，归一化
            photo = np.reshape(self.preprocess_input(photo),
                               [1, IMAGE_SIZES[PHI], IMAGE_SIZES[PHI], 3])

            if PRUNING:
                model = self.model
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                model.set_tensor(input_details[0]['index'], photo)
                model.invoke()
                pred1 = model.get_tensor(output_details[0]['index'])
                pred2 = model.get_tensor(output_details[1]['index'])
                preds = (pred2, pred1)
            else:
                preds = self.model.predict(photo)

            # 将预测结果进行解码
            results = self.bbox_util.detection_out(preds, self.prior, confidence_threshold=self.confidence)

            if len(results[0]) <= 0:
                return {'times': str(time.time() - start_time)}
            results = np.array(results)

            # 筛选出其中得分高于confidence的框
            det_label = results[0][:, 5]
            det_conf = results[0][:, 4]
            det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:,
                                                                                                           3]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
                det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(
                det_ymax[top_indices],
                -1)

            # 去掉灰条
            boxes = self.efficientdet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                    np.array([IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]]),
                                                    image_shape)

            def classifications(image, left, top, right, bottom):
                image_crop = image.crop((left, top, right, bottom))
                image_bytearr = io.BytesIO()
                image_crop.save(image_bytearr, format='JPEG')
                image_bytes = image_bytearr.getvalue()
                data = {'data': [f'data:image;base64,{base64.b64encode(image_bytes).decode()}']}
                response = requests.post('http://127.0.0.1:7860/api/predict/', json=data).json()
                result = json.loads(response.get('data')[0].get('label'))
                predicted_class = result.get('result')
                recognition_rate = result.get('recognition_rate')
                recognition_rate = float(recognition_rate.replace('%', '')) / 100
                recognition_rate_list.append(recognition_rate)
                label = {"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}
                result_list.append(label)

            with ThreadPoolExecutor(max_workers=10) as t:
                for i, c in enumerate(top_label_indices):
                    predicted_class = self.num_classes_list[int(c)]
                    score = top_conf[i]
                    recognition_rate_list.append(score)
                    top, left, bottom, right = boxes[i]
                    top = top - 5
                    left = left - 5
                    bottom = bottom + 5
                    right = right + 5

                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                    right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

                    if self.classification:
                        t.submit(classifications, image, left, top, right, bottom)
                        # image_crop = image.crop((left, top, right, bottom))
                        # image_bytearr = io.BytesIO()
                        # image_crop.save(image_bytearr, format='JPEG')
                        # image_bytes = image_bytearr.getvalue()
                        # data = {'data': [f'data:image;base64,{base64.b64encode(image_bytes).decode()}']}
                        # response = requests.post('http://127.0.0.1:7860/api/predict/', json=data).json()
                        # result = json.loads(response.get('data')[0].get('label'))
                        # predicted_class = result.get('result')
                        # recognition_rate = result.get('recognition_rate')
                        # recognition_rate = float(recognition_rate.replace('%', '')) / 100
                        # recognition_rate_list.append(recognition_rate)
                        # label = {"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}
                        # result_list.append(label)
                    else:
                        label = {"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}
                        result_list.append(label)

            recognition_rate = self.recognition_probability(recognition_rate_list)
            end_time = time.time()
            times = end_time - start_time
            return {'result': str(result_list), 'recognition_rate': str(round(recognition_rate * 100, 2)) + '%',
                    'times': str(times)}

        elif MODE == 'YOLO' or MODE == 'YOLO_TINY':
            result_list = []
            recognition_rate_list = []
            start_time = time.time()
            image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            new_image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
            boxed_image = self.letterbox_image(image, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    KT.learning_phase(): 0
                })

            for i, c in list(enumerate(out_classes)):
                predicted_class = self.num_classes_list[c]
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                label = {"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}
                result_list.append(label)
                recognition_rate_list.append(score)

            recognition_rate = self.recognition_probability(recognition_rate_list)
            end_time = time.time()
            times = end_time - start_time
            return {'result': str(result_list), 'recognition_rate': str(round(recognition_rate * 100, 2)) + '%',
                    'times': str(times)}

        else:
            start_time = time.time()
            if PRUNING:
                model = self.model
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                input_data = self.decode_image(image)
                model.set_tensor(input_details[0]['index'], input_data)
                model.invoke()
                vertor = model.get_tensor(output_details[0]['index'])
            else:
                model = self.model
                vertor = model.predict(self.decode_image(image=image))
            result, recognition_rate = self.decode_vector(vector=vertor, num_classes=self.num_classes_dict)
            end_time = time.time()
            times = end_time - start_time
            return {'result': str(result), 'recognition_rate': str(round(recognition_rate * 100, 2)) + '%',
                    'times': str(times)}


def get_ground_truth(image):
    if len(os.listdir(GT_PATH)) == 0:
        paths = os.path.splitext(os.path.split(image)[-1])[0]
        try:
            label = (paths, glob.glob(f'{LABEL_PATH}/*/{paths}.xml')[0])
        except:
            label = (paths, glob.glob(f'{LABEL_PATH}/{paths}.xml')[0])
        index, label_xml = label
        file = open(label_xml, encoding='utf-8')
        for i in ET.parse(file).getroot().iter('object'):
            with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
                result = f.read()
            num_classes_dict = json.loads(result)
            if len(num_classes_dict) == 1:
                classes = 'block'
            else:
                classes = i.find('name').text
            xmlbox = i.find('bndbox')
            box = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                   int(float(xmlbox.find('xmax').text)),
                   int(float(xmlbox.find('ymax').text)))
            with open(os.path.join(GT_PATH, index + '.txt'), mode='a', encoding='utf-8') as f:
                f.write(f'{classes} {box[0]} {box[1]} {box[2]} {box[3]}\n')
        file.close()
    _, file = os.path.split(image)
    new_file = os.path.join(IMG_PATH, file)
    logger.debug(f'正在复制{image}到{new_file}')
    shutil.copy(image, new_file)
    logger.success('复制完成')


if not os.path.exists(GT_PATH):
    os.mkdir(GT_PATH)

if not os.path.exists(DR_PATH):
    os.mkdir(DR_PATH)

if not os.path.exists(IMG_PATH):
    os.mkdir(IMG_PATH)

if len(os.listdir(IMG_PATH)) == 0:
    test_image_list = Image_Processing.extraction_image(TEST_PATH)
    for image in test_image_list:
        get_ground_truth(image)
if len(os.listdir(DR_PATH)) == 0:
    if PRUNING:
        model_path = os.path.join(MODEL_PATH, PRUNING_MODEL_NAME)
    else:
        model_path = os.path.join(MODEL_PATH, MODEL_NAME)

    logger.debug(f'加载模型{model_path}')
    if not os.path.exists(model_path):
        raise OSError(f'{model_path}没有模型')
    Predict = Predict_Image(model_path=model_path, detection_results=True)
    for image in test_image_list:
        Predict.predict_image(image)

MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# make sure that the cwd() is the location of the python script (so that every path makes sense)

if os.path.exists(IMG_PATH):
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            args.no_animation = True
else:
    args.no_animation = True

# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
    try:
        import cv2

        show_animation = True
    except ImportError:
        print("\"opencv-python\" not found, please install to visualize the results.")
        args.no_animation = True

# try to import Matplotlib if the user didn't choose the option --no-plot
draw_plot = False
if not args.no_plot:
    try:
        import matplotlib.pyplot as plt

        draw_plot = True
    except ImportError:
        print("\"matplotlib\" not found, please install it to get the resulting plots.")
        args.no_plot = True


def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


"""
 throw error and exit
"""


def error(msg):
    print(msg)
    sys.exit(0)


"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path, encoding='utf-8') as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


"""
 Draws text in image
"""


def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


"""
 Plot - adjust axes
"""


def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 Draw plot using Matplotlib
"""


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


"""
 Create a ".temp_files/" and "output/" directory
"""
TEMP_FILES_PATH = ".temp_files"
if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
    os.makedirs(TEMP_FILES_PATH)
output_files_path = OUTPUT_PATH
if os.path.exists(output_files_path):  # if it exist already
    # reset the output directory
    shutil.rmtree(output_files_path)

os.makedirs(output_files_path)
if draw_plot:
    os.makedirs(os.path.join(output_files_path, "classes"))
if show_animation:
    os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))

"""
 ground-truth
     Load each of the ground-truth files into a temporary ".json" file.
     Create a list of all the class names present in the ground-truth (gt_classes).
"""
# get a list with the ground-truth files
ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class = {}
counter_images_per_class = {}

gt_files = []
for txt_file in ground_truth_files_list:
    # print(txt_file)
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    # check if there is a correspondent detection-results file
    temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
    if not os.path.exists(temp_path):
        error_msg = "Error. File not found: {}\n".format(temp_path)
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
        error(error_msg)
    lines_list = file_lines_to_list(txt_file)
    # create ground-truth dictionary
    bounding_boxes = []
    is_difficult = False
    already_seen_classes = []
    for line in lines_list:
        try:
            if "difficult" in line:
                class_name, left, top, right, bottom, _difficult = line.split()
                is_difficult = True
            else:
                class_name, left, top, right, bottom = line.split()
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format.\n"
            error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
            error_msg += " Received: " + line
            error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
            error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
            error(error_msg)
        # check if class is in the ignore list, if yes skip
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " + bottom
        if is_difficult:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
            is_difficult = False
        else:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)

    # dump bounding_boxes into a ".json" file
    new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
    gt_files.append(new_temp_file)
    with open(new_temp_file, 'w', encoding='utf-8') as outfile:
        json.dump(bounding_boxes, outfile)

gt_classes = list(gt_counter_per_class.keys())
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
# print(gt_classes)
# print(gt_counter_per_class)

"""
 Check format of the flag --set-class-iou (if used)
    e.g. check if class exists
"""
if specific_iou_flagged:
    n_args = len(args.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
    # [class_1] [IoU_1] [class_2] [IoU_2]
    # specific_iou_classes = ['class_1', 'class_2']
    specific_iou_classes = args.set_class_iou[::2]  # even
    # iou_list = ['IoU_1', 'IoU_2']
    iou_list = args.set_class_iou[1::2]  # odd
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 detection-results
     Load each of the detection-results files into a temporary ".json" file.
"""
# get a list with the detection-results files
dr_files_list = glob.glob(DR_PATH + '/*.txt')
dr_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in dr_files_list:
        # print(txt_file)
        # the first time it checks if all the corresponding ground-truth files exist
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
        if class_index == 0:
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                error(error_msg)
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                error(error_msg)
            if tmp_class_name == class_name:
                # print("match")
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
                # print(bounding_boxes)
    # sort detection-results by decreasing confidence
    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
    with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w', encoding='utf-8') as outfile:
        json.dump(bounding_boxes, outfile)

"""
 Calculate the AP for each class
"""
sum_AP = 0.0
ap_dictionary = {}
lamr_dictionary = {}
# open file to store the output
with open(output_files_path + "/output.txt", 'w', encoding='utf-8') as output_file:
    output_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            if show_animation:
                # find ground truth image
                ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                # tifCounter = len(glob.glob1(myPath,"*.tif"))
                if len(ground_truth_img) == 0:
                    error("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                    error("Error. Multiple image with id: " + file_id)
                else:  # found image
                    # print(IMG_PATH + "/" + ground_truth_img[0])
                    # Load image
                    img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])

                    if img is None:
                        img = cv2.imdecode(np.fromfile(IMG_PATH + "/" + ground_truth_img[0], dtype=np.uint8),
                                           cv2.IMREAD_COLOR)
                    # load image with draws of multiple detections
                    img_cumulative_path = output_files_path + "/images/" + ground_truth_img[0]
                    if os.path.isfile(img_cumulative_path):
                        img_cumulative = cv2.imread(img_cumulative_path)

                        if img_cumulative is None:
                            img_cumulative = cv2.imdecode(np.fromfile(img_cumulative_path, dtype=np.uint8),
                                                          cv2.IMREAD_COLOR)

                    else:
                        img_cumulative = img.copy()

                    # Add bottom border to image
                    bottom_border = 60
                    BLACK = [0, 0, 0]
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # assign detection as true positive/don't care/false positive
            if show_animation:
                status = "NO MATCH FOUND!"  # status is only used in the animation
            # set minimum overlap
            min_overlap = MINOVERLAP

            if specific_iou_flagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name.encode('utf-8'))
                    min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w', encoding='utf-8') as f:
                            f.write(json.dumps(ground_truth_data))
                        if show_animation:
                            status = "MATCH!"
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                        if show_animation:
                            status = "REPEATED MATCH!"
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

            """
             Draw image to show animation
            """
            if show_animation:
                height, widht = img.shape[:2]

                white = (255, 255, 255)
                light_blue = (255, 200, 100)
                green = (0, 255, 0)
                light_red = (30, 30, 255)
                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2.0))
                text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                    else:
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                        color = green
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                    float(detection["confidence"]) * 100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if ovmax > 0:  # if there is intersections between the bounding-boxes
                    bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                    cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                    cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                    cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                # show image
                cv2.imshow("Animation", img)
                cv2.waitKey(20)  # show for 20 ms
                # save image to output
                output_img_path = output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(
                    idx) + ".jpg"
                if class_name.isalpha():
                    cv2.imwrite(output_img_path, img)
                    # save the image with all the objects drawn to it
                    cv2.imwrite(img_cumulative_path, img_cumulative)
                else:
                    cv2.imencode('.jpg', img)[1].tofile(output_img_path)
                    # save the image with all the objects drawn to it

                    cv2.imencode('.jpg', img_cumulative)[1].tofile(img_cumulative_path)

        # print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        # print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        # print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        # print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
        """
         Write to output.txt
        """
        rounded_prec = ['%.2f' % elem for elem in prec]
        rounded_rec = ['%.2f' % elem for elem in rec]
        output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        if not args.quiet:
            print(text)
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
        lamr_dictionary[class_name] = lamr

        """
         Draw plot
        """
        if draw_plot:
            plt.plot(rec, prec, '-o')
            # add a new penultimate point to the list (mrec[-2], 0.0)
            # since the last line segment (and respective area) do not affect the AP value
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
            # set window title
            fig = plt.gcf()  # gcf - get current figure
            fig.canvas.set_window_title('AP ' + class_name)
            # set plot title
            plt.title('class: ' + text)
            # plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca()  # gca - get current axes
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
            # Alternative option -> wait for button to be pressed
            # while not plt.waitforbuttonpress(): pass # wait for key display
            # Alternative option -> normal display
            # plt.show()
            # save the plot
            fig.savefig(output_files_path + "/classes/" + class_name + ".png")
            plt.cla()  # clear axes for next plot

    if show_animation:
        cv2.destroyAllWindows()

    output_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP * 100)
    output_file.write(text + "\n")
    print(text)

"""
 Draw false negatives
"""
if show_animation:
    pink = (203, 192, 255)
    for tmp_file in gt_files:
        ground_truth_data = json.load(open(tmp_file))
        # print(ground_truth_data)
        # get name of corresponding image
        start = TEMP_FILES_PATH + '/'
        img_id = tmp_file[tmp_file.find(start) + len(start):tmp_file.rfind('_ground_truth.json')]
        img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
        img = cv2.imread(img_cumulative_path)

        # if not img:
        #     img = cv2.imdecode(np.fromfile(img_cumulative_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img_path = IMG_PATH + '/' + img_id + ".jpg"
            img = cv2.imread(img_path)
            if img is None:
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # draw false negatives
        for obj in ground_truth_data:
            if not obj['used']:
                bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), pink, 2)
        if class_name.isalpha():
            # cv2.imencode('.jpg', img)[1].tofile(img_cumulative_path)
            cv2.imwrite(img_cumulative_path, img)
        else:
            cv2.imencode('.jpg', img)[1].tofile(img_cumulative_path)

# remove the temp_files directory
shutil.rmtree(TEMP_FILES_PATH)

"""
 Count total of detection-results
"""
# iterate through all the files
det_counter_per_class = {}
for txt_file in dr_files_list:
    # get lines to list
    lines_list = file_lines_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        # check if class is in the ignore list, if yes skip
        if class_name in args.ignore:
            continue
        # count that object
        if class_name in det_counter_per_class:
            det_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            det_counter_per_class[class_name] = 1
# print(det_counter_per_class)
dr_classes = list(det_counter_per_class.keys())
# logger.debug(dr_classes)
"""
 Plot the total number of occurences of each class in the ground-truth
"""
if draw_plot:
    window_title = "ground-truth-info"
    plot_title = "ground-truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = output_files_path + "/ground-truth-info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
    )

"""
 Write number of ground-truth objects per class to results.txt
"""
with open(output_files_path + "/output.txt", 'a', encoding='utf-8') as output_file:
    output_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in dr_classes:
    # if class exists in detection-result but not in ground-truth then there are no true positives in that class
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0
# print(count_true_positives)

"""
 Plot the total number of occurences of each class in the "detection-results" folder
"""
if draw_plot:
    window_title = "detection-results-info"
    # Plot title
    plot_title = "detection-results\n"
    plot_title += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = output_files_path + "/detection-results-info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = count_true_positives
    draw_plot_func(
        det_counter_per_class,
        len(det_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
    )

"""
 Write number of detected objects per class to output.txt
"""
with open(output_files_path + "/output.txt", 'a', encoding='utf-8') as output_file:
    output_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        output_file.write(text)

"""
 Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
"""
if draw_plot:
    window_title = "lamr"
    plot_title = "log-average miss rate"
    x_label = "log-average miss rate"
    output_path = output_files_path + "/lamr.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        lamr_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if draw_plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP * 100)
    x_label = "Average Precision"
    output_path = output_files_path + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )
