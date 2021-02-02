import io
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
from works.simples.settings import PRUNING
from works.simples.settings import USE_GPU
from works.simples.settings import GT_PATH
from works.simples.settings import DR_PATH
from works.simples.settings import IMG_PATH
from works.simples.settings import MODEL_PATH
from works.simples.settings import MODEL_NAME
from works.simples.settings import PRUNING_MODEL_NAME
from works.simples.utils import Image_Processing
from tensorflow.compat.v1.keras import backend as KT
from works.simples.settings import THRESHOLD
from works.simples.settings import TEST_PATH
from works.simples.settings import LABEL_PATH
from works.simples.settings import IMAGE_WIDTH
from works.simples.settings import OUTPUT_PATH
from works.simples.settings import IMAGE_SIZES
from works.simples.settings import IMAGE_HEIGHT
from works.simples.settings import NUMBER_CLASSES_FILE
from works.simples.utils import Predict_Image

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


class Predict_Images(Predict_Image):
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
                # if self.detection_results:
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
                # if self.detection_results:
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
                # if self.detection_results:
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
                # if self.detection_results:
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


def get_ground_truth(image):
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


def get_images_optional(image):
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

test_image_list = Image_Processing.extraction_image(TEST_PATH)
if len(os.listdir(IMG_PATH)) == 0:
    for image in test_image_list:
        get_images_optional(image)

if len(os.listdir(GT_PATH)) == 0:
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
    Predict = Predict_Images(model_path=model_path)
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
