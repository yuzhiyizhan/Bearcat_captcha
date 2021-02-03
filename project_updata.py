import re


def project(string: str, work_path: str, project_name: str):
    string = re.sub('{', '{{', string)
    string = re.sub('}', '}}', string)
    string = re.sub(work_path, '{work_path}', string)
    string = re.sub(project_name, '{project_name}', string)

    print(string)


if __name__ == '__main__':
    string = """import io
import re
import os
import cv2
import time
import json
import glob
import base64
import shutil
import random
import hashlib
import colorsys
import operator
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from loguru import logger
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.image import imread
from matplotlib.patches import Rectangle
from timeit import default_timer as timer
from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as KT
from works.simple.models import Models
from works.simple.models import Yolo_model
from works.simple.models import YOLO_anchors
from works.simple.models import Yolo_tiny_model
from works.simple.models import Efficientdet_anchors
from works.simple.settings import PHI
from works.simple.settings import MODE
from works.simple.settings import MODEL
from works.simple.settings import DIVIDE
from works.simple.settings import PRUNING
from works.simple.settings import THRESHOLD
from works.simple.settings import MAX_BOXES
from works.simple.settings import TEST_PATH
from works.simple.settings import LABEL_PATH
from works.simple.settings import BASIC_PATH
from works.simple.settings import CONFIDENCE
from works.simple.settings import IMAGE_WIDTH
from works.simple.settings import IMAGE_SIZES
from works.simple.settings import DIVIDE_RATO
from works.simple.settings import IMAGE_HEIGHT
from works.simple.settings import NUMBER_CLASSES_FILE
from works.simple.settings import CAPTCHA_LENGTH
from works.simple.settings import IMAGE_CHANNALS
from works.simple.settings import VALIDATION_PATH
from works.simple.settings import DATA_ENHANCEMENT
from works.simple.settings import TRAIN_PATH
from concurrent.futures import ThreadPoolExecutor

mean_time = []
right_value = 0
predicted_value = 0
start = time.time()
time_list = []
table = []
for i in range(256):
    if i < THRESHOLD:
        table.append(0)
    else:
        table.append(255)
try:
    if MODE == 'CTC_TINY':
        input_len = np.int64(Models.captcha_model_ctc_tiny().get_layer('reshape_len').output_shape[1])
except:
    pass


class Image_Processing(object):
    @classmethod
    # 提取全部图片plus
    def extraction_image(self, path: str, mode=MODE) -> list:
        try:
            data_path = []
            datas = [os.path.join(path, i) for i in os.listdir(path)]
            for data in datas:
                data_path = data_path + [os.path.join(data, i) for i in os.listdir(data)]
            return data_path
        except:
            return [os.path.join(path, i) for i in os.listdir(path)]

    @classmethod
    def extraction_label(self, path_list: list, suffix=True, divide='_', mode=MODE):
        if mode == 'ORDINARY':
            if suffix:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                paths = [re.split(divide, i)[0] for i in paths]
            else:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
            ocr_path = []
            for i in paths:
                for s in i:
                    ocr_path.append(s)
            n_class = sorted(set(ocr_path))
            save_dict = dict((index, name) for index, name in enumerate(n_class))
            if not os.path.exists(os.path.join(BASIC_PATH, NUMBER_CLASSES_FILE)):
                with open(NUMBER_CLASSES_FILE, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(save_dict, ensure_ascii=False))
            with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
                make_dict = json.loads(f.read())
            make_dict = dict((name, index) for index, name in make_dict.items())
            label_list = [self.text2vector(label, make_dict=make_dict) for label in paths]
            return label_list
        elif mode == 'NUM_CLASSES':
            if suffix:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                paths = [re.split(divide, i)[0] for i in paths]
            else:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
            n_class = sorted(set(paths))
            save_dict = dict((index, name) for index, name in enumerate(n_class))
            if not os.path.exists(os.path.join(BASIC_PATH, NUMBER_CLASSES_FILE)):
                with open(NUMBER_CLASSES_FILE, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(save_dict, ensure_ascii=False))
            with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
                make_dict = json.loads(f.read())
            make_dict = dict((name, index) for index, name in make_dict.items())
            label_list = [self.text2vector(label, make_dict=make_dict, mode=MODE) for label in paths]
            return label_list
        elif mode == 'CTC':
            if suffix:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                paths = [re.split(divide, i)[0] for i in paths]
            else:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
            ocr_path = []
            for i in paths:
                for s in i:
                    ocr_path.append(s)
            n_class = sorted(set(ocr_path))
            save_dict = dict((index, name) for index, name in enumerate(n_class))
            if not os.path.exists(os.path.join(BASIC_PATH, NUMBER_CLASSES_FILE)):
                with open(NUMBER_CLASSES_FILE, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(save_dict, ensure_ascii=False))
            with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
                make_dict = json.loads(f.read())
            make_dict = dict((name, index) for index, name in make_dict.items())
            label_list = [self.text2vector(label, make_dict=make_dict) for label in paths]
            return label_list
        elif mode == 'YOLO' or mode == 'YOLO_TINY' or mode == 'EFFICIENTDET' or mode == 'SSD':
            n_class = []
            paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
            try:
                label = [(i, glob.glob(f'{LABEL_PATH}/*/{i}.xml')[0]) for i in paths]
            except:
                label = [(i, glob.glob(f'{LABEL_PATH}/{i}.xml')[0]) for i in paths]
            path = [(i, os.path.splitext(os.path.split(i)[-1])[0]) for i in path_list]
            for index, label_xml in label:
                file = open(label_xml, encoding='utf-8')
                for i in ET.parse(file).getroot().iter('object'):
                    classes = i.find('name').text
                    n_class.append(classes)
                file.close()
            n_class = sorted(set(n_class))
            save_dict = dict((index, name) for index, name in enumerate(n_class))
            if not os.path.exists(os.path.join(BASIC_PATH, NUMBER_CLASSES_FILE)):
                with open(NUMBER_CLASSES_FILE, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(save_dict, ensure_ascii=False))
            with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
                make_dict = json.loads(f.read())
            make_dict = dict((name, index) for index, name in make_dict.items())
            label_dict = {}
            for index, label_xml in label:
                file = open(label_xml, encoding='utf-8')
                box_classes = []
                for i in ET.parse(file).getroot().iter('object'):
                    classes = i.find('name').text
                    xmlbox = i.find('bndbox')
                    classes_id = make_dict.get(classes, '0')
                    box = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                           int(float(xmlbox.find('xmax').text)),
                           int(float(xmlbox.find('ymax').text)))
                    box = ','.join([str(a) for a in box]) + ',' + str(classes_id)
                    box_classes.append(box)
                box = np.array([np.array(list(map(int, box.split(',')))) for box in box_classes])
                label_dict[index] = box
                file.close()
            label_list = ([label_dict.get(value) for index, value in path])
            return label_list
        elif mode == 'CTC_TINY':
            if suffix:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                paths = [re.split(divide, i)[0] for i in paths]
            else:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
            ocr_path = []
            for i in paths:
                for s in i:
                    ocr_path.append(s)
            n_class = sorted(set(ocr_path))
            save_dict = dict((index, name) for index, name in enumerate(n_class))
            if not os.path.exists(os.path.join(BASIC_PATH, NUMBER_CLASSES_FILE)):
                with open(NUMBER_CLASSES_FILE, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(save_dict, ensure_ascii=False))
            with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
                make_dict = json.loads(f.read())
            make_dict = dict((name, index) for index, name in make_dict.items())
            label_list = [self.text2vector(label, make_dict=make_dict) for label in paths]
            return label_list
        else:
            raise ValueError(f'没有mode={mode}提取标签的方法')

    @classmethod
    def text2vector(self, label, make_dict: dict, mode=MODE):
        if mode == 'ORDINARY':
            if len(label) > CAPTCHA_LENGTH:
                raise ValueError(f'标签{label}长度大于预设值{CAPTCHA_LENGTH},建议设置CAPTCHA_LENGTH为{len(label) + 2}')
            num_classes = len(make_dict)
            label_ver = np.ones((CAPTCHA_LENGTH), dtype=np.int64) * num_classes
            for index, c in enumerate(label):
                if not make_dict.get(c):
                    raise ValueError(f'错误的值{c}')
                label_ver[index] = make_dict.get(c)
            label_ver = list(tf.keras.utils.to_categorical(label_ver, num_classes=num_classes + 1).ravel())
            return label_ver
        elif mode == 'NUM_CLASSES':
            num_classes = len(make_dict)
            label_ver = np.zeros((num_classes), dtype=np.int64) * num_classes
            label_ver[int(make_dict.get(label))] = 1.
            return label_ver
        elif mode == 'CTC':
            label_ver = []
            for c in label:
                if not make_dict.get(c):
                    raise ValueError(f'错误的值{c}')
                label_ver.append(int(make_dict.get(c)))
            label_ver = np.array(label_ver)
            return label_ver
        elif mode == 'CTC_TINY':
            if len(label) > CAPTCHA_LENGTH:
                raise ValueError(f'标签{label}长度大于预设值{CAPTCHA_LENGTH},建议设置CAPTCHA_LENGTH为{len(label) + 2}')
            num_classes = len(make_dict)
            label_ver = np.ones((CAPTCHA_LENGTH), dtype=np.int64) * num_classes
            for index, c in enumerate(label):
                if not make_dict.get(c):
                    raise ValueError(f'错误的值{c}')
                label_ver[index] = make_dict.get(c)
            # label_ver = list(tf.keras.utils.to_categorical(label_ver, num_classes=num_classes + 1).ravel())
            return label_ver
        else:
            raise ValueError(f'没有mode={mode}提取标签的方法')

    @classmethod
    def _shutil_move(self, full_path, des_path, number):
        shutil.move(full_path, des_path)
        logger.debug(f'剩余数量{number}')

    # 分割数据集
    @classmethod
    def split_dataset(self, path: list, proportion=DIVIDE_RATO) -> bool:
        if DIVIDE:
            number = 0
            logger.debug(f'数据集有{len(path)},{proportion * 100}%作为验证集,{proportion * 100}%作为测试集')
            division_number = int(len(path) * proportion)
            logger.debug(f'验证集数量为{division_number},测试集数量为{division_number}')
            validation_dataset = random.sample(path, division_number)
            with ThreadPoolExecutor(max_workers=50) as t:
                for i in validation_dataset:
                    number = number + 1
                    logger.debug(f'准备移动{(number / len(validation_dataset)) * 100}%')
                    t.submit(path.remove, i)
            validation = [os.path.join(VALIDATION_PATH, os.path.split(i)[-1]) for i in validation_dataset]
            validation_lenght = len(validation)
            with ThreadPoolExecutor(max_workers=50) as t:
                for full_path, des_path in zip(validation_dataset, validation):
                    validation_lenght -= 1
                    t.submit(Image_Processing._shutil_move, full_path, des_path, validation_lenght)

            test_dataset = random.sample(path, division_number)
            test = [os.path.join(TEST_PATH, os.path.split(i)[-1]) for i in test_dataset]
            test_lenght = len(test)
            with ThreadPoolExecutor(max_workers=50) as t:
                for full_path, des_path in zip(test_dataset, test):
                    test_lenght -= 1
                    t.submit(Image_Processing._shutil_move, full_path, des_path, test_lenght)
            logger.success(f'任务结束')
            return True
        else:
            logger.debug(f'数据集有{len(path)},{proportion * 100}%作为测试集')
            division_number = int(len(path) * proportion)
            logger.debug(f'测试集数量为{division_number}')
            test_dataset = random.sample(path, division_number)
            test = [os.path.join(TEST_PATH, os.path.split(i)[-1]) for i in test_dataset]
            test_lenght = len(test)
            with ThreadPoolExecutor(max_workers=50) as t:
                for full_path, des_path in zip(test_dataset, test):
                    test_lenght -= 1
                    t.submit(Image_Processing._shutil_move, full_path, des_path, test_lenght)
            logger.success(f'任务结束')
            return True

    # # 增强图片
    # @classmethod
    # def preprosess_save_images(self, image, number):
    #     logger.info(f'开始处理{image}')
    #     with open(image, 'rb') as images:
    #         im = Image.open(images)
    #         blur_im = im.filter(ImageFilter.BLUR)
    #         contour_im = im.filter(ImageFilter.CONTOUR)
    #         detail_im = im.filter(ImageFilter.DETAIL)
    #         edge_enhance_im = im.filter(ImageFilter.EDGE_ENHANCE)
    #         edge_enhance_more_im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #         emboss_im = im.filter(ImageFilter.EMBOSS)
    #         flnd_edges_im = im.filter(ImageFilter.FIND_EDGES)
    #         smooth_im = im.filter(ImageFilter.SMOOTH)
    #         smooth_more_im = im.filter(ImageFilter.SMOOTH_MORE)
    #         sharpen_im = im.filter(ImageFilter.SHARPEN)
    #         maxfilter_im = im.filter(ImageFilter.MaxFilter)
    #         minfilter_im = im.filter(ImageFilter.MinFilter)
    #         modefilter_im = im.filter(ImageFilter.ModeFilter)
    #         medianfilter_im = im.filter(ImageFilter.MedianFilter)
    #         unsharpmask_im = im.filter(ImageFilter.UnsharpMask)
    #         left_right_im = im.transpose(Image.FLIP_LEFT_RIGHT)
    #         top_bottom_im = im.transpose(Image.FLIP_TOP_BOTTOM)
    #         rotate_list = [im.rotate(i) for i in list(range(1, 360, 60))]
    #         brightness_im = ImageEnhance.Brightness(im).enhance(0.5)
    #         brightness_up_im = ImageEnhance.Brightness(im).enhance(1.5)
    #         color_im = ImageEnhance.Color(im).enhance(0.5)
    #         color_up_im = ImageEnhance.Color(im).enhance(1.5)
    #         contrast_im = ImageEnhance.Contrast(im).enhance(0.5)
    #         contrast_up_im = ImageEnhance.Contrast(im).enhance(1.5)
    #         sharpness_im = ImageEnhance.Sharpness(im).enhance(0.5)
    #         sharpness_up_im = ImageEnhance.Sharpness(im).enhance(1.5)
    #         image_list = [im, blur_im, contour_im, detail_im, edge_enhance_im, edge_enhance_more_im, emboss_im,
    #                       flnd_edges_im,
    #                       smooth_im, smooth_more_im, sharpen_im, maxfilter_im, minfilter_im, modefilter_im,
    #                       medianfilter_im,
    #                       unsharpmask_im, left_right_im,
    #                       top_bottom_im, brightness_im, brightness_up_im, color_im, color_up_im, contrast_im,
    #                       contrast_up_im, sharpness_im, sharpness_up_im] + rotate_list
    #         for index, file in enumerate(image_list):
    #             paths, files = os.path.split(image)
    #             files, suffix = os.path.splitext(files)
    #             new_file = os.path.join(paths, train_enhance_path, files + str(index) + suffix)
    #             file.save(new_file)
    #     logger.success(f'处理完成{image},还剩{number}张图片待增强')

    @classmethod
    def preprosess_save_images(self, image, number):
        logger.info(f'开始处理{image}')
        name = os.path.splitext(image)[0]
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                                  samplewise_center=False,
                                                                  featurewise_std_normalization=False,
                                                                  samplewise_std_normalization=False,
                                                                  zca_whitening=False,
                                                                  zca_epsilon=1e-6,
                                                                  rotation_range=40,
                                                                  width_shift_range=0.2,
                                                                  height_shift_range=0.2,
                                                                  brightness_range=(0.7, 1.3),
                                                                  shear_range=30,
                                                                  zoom_range=0.2,
                                                                  channel_shift_range=0.,
                                                                  fill_mode='nearest',
                                                                  cval=0.,
                                                                  horizontal_flip=False,
                                                                  vertical_flip=False,
                                                                  rescale=1 / 255,
                                                                  preprocessing_function=None,
                                                                  data_format=None,
                                                                  validation_split=0.0,
                                                                  dtype=None)

        img = tf.keras.preprocessing.image.load_img(image)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, 0)
        i = 0
        for _ in datagen.flow(x, batch_size=1, save_to_dir=TRAIN_PATH, save_prefix=name, save_format='jpg'):
            i += 1
            if i == DATA_ENHANCEMENT:
                break
        logger.success(f'处理完成{image},还剩{number}张图片待增强')

    # @classmethod
    # # 展示图片处理后的效果
    # def show_image(self, image_path):
    #     '''
    #     展示图片处理后的效果
    #     :param image_path:
    #     :return:
    #     '''
    #     image = Image.open(image_path)
    #     while True:
    #         width, height = image.size
    #         if IMAGE_HEIGHT < height:
    #             resize_width = int(IMAGE_HEIGHT / height * width)
    #             image = image.resize((resize_width, IMAGE_HEIGHT))
    #         if IMAGE_WIDTH < width:
    #             resize_height = int(IMAGE_WIDTH / width * height)
    #             image = image.resize((IMAGE_WIDTH, resize_height))
    #         if IMAGE_WIDTH >= width and IMAGE_HEIGHT >= height:
    #             break
    #     width, height = image.size
    #     image = np.array(image)
    #     image = np.pad(image, ((0, IMAGE_HEIGHT - height), (0, IMAGE_WIDTH - width), (0, 0)), 'constant',
    #                    constant_values=0)
    #     image = Image.fromarray(image)
    #     image_bytearr = io.BytesIO()
    #     image.save(image_bytearr, format='JPEG')
    #     plt.imshow(image)
    #     plt.show()

    @staticmethod
    def show_image(image):
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
            new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
            new_image = new_image.convert('L')
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            table = []
            for i in range(256):
                if i < THRESHOLD:
                    table.append(0)
                else:
                    table.append(255)
            new_image = new_image.point(table, 'L')
        new_image.show()

    @staticmethod
    # 图片画框
    def tagging_image(image_path, box):
        im = imread(image_path)
        plt.figure()
        plt.imshow(im)
        ax = plt.gca()
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

    @staticmethod
    def tagging_image2(image_path, box):
        img = cv2.imread(image_path)
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        cv2.imshow('example.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


class SSD_Generator(object):
    def __init__(self, bbox_util, batch_size,
                 image_list, label_list, image_size, num_classes,
                 ):
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.image_list = image_list
        self.image_label = label_list
        self.image_size = image_size
        self.num_classes = num_classes - 1

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, line, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):

        image = Image.open(line)
        iw, ih = image.size
        h, w = input_shape
        box = label

        # resize image
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def generate(self):
        for i in range(len(self.image_list)):
            lines = self.image_list
            label = self.image_label
            inputs = []
            targets = []
            # n = len(lines)
            for i in range(len(lines)):
                img, y = self.get_random_data(lines[i], label[i], self.image_size[0:2])
                # i = (i + 1) % n
                if len(y) != 0:
                    boxes = np.array(y[:, :4], dtype=np.float32)
                    boxes[:, 0] = boxes[:, 0] / self.image_size[1]
                    boxes[:, 1] = boxes[:, 1] / self.image_size[0]
                    boxes[:, 2] = boxes[:, 2] / self.image_size[1]
                    boxes[:, 3] = boxes[:, 3] / self.image_size[0]
                    one_hot_label = np.eye(self.num_classes)[np.array(y[:, 4], np.int32)]
                    if ((boxes[:, 3] - boxes[:, 1]) <= 0).any() and ((boxes[:, 2] - boxes[:, 0]) <= 0).any():
                        continue

                    y = np.concatenate([boxes, one_hot_label], axis=-1)

                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tf.keras.applications.imagenet_utils.preprocess_input(tmp_inp), tmp_targets


class YOLO_Generator(object):

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        if y2 - y1 < 5:
                            continue
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, image_list, label_list, input_shape, max_boxes=MAX_BOXES, hue=.1, sat=1.5,
                                    val=1.5):
        '''random preprocessing for real-time data augmentation'''
        h, w = input_shape
        min_offset_x = 0.4
        min_offset_y = 0.4
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []
        index = 0

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        for image, label in zip(image_list, label_list):
            # 打开图片
            image = Image.open(image)
            if image.mode != 'RGB':
                image = image.convert("RGB")
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = label
            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = w / h
            scale = self.rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        # 将box进行调整
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes) > 0:
            if len(new_boxes) > max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    def get_random_data(self, image, label, input_shape, max_boxes=MAX_BOXES, jitter=.3, hue=.1, sat=1.5, val=1.5):

        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        iw, ih = image.size
        h, w = input_shape
        box = label

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        if MODE == 'YOLO':
            scale = self.rand(.25, 2)
        elif MODE == 'YOLO_TINY':
            scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1
        # 将box进行调整
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):

        # 一共有三个特征层数
        num_layers = len(anchors) // 3
        # 先验框
        # 678为 142,110,  192,243,  459,401
        # 345为 36,75,  76,55,  72,146
        # 012为 12,16,  19,36,  40,28
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')  # 416,416

        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # 计算比例
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # m张图
        m = true_boxes.shape[0]

        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                           dtype='float32') for l in range(num_layers)]

        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        # 长宽要大于0才有效
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):

            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue

            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            # 计算真实框和哪个先验框最契合
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')

                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1
        return y_true

    def data_generator(self, image_list, label_list, batch_size, input_shape, anchors, num_classes, mosaic=False):

        n = len(image_list)
        i = 0
        flag = True
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if mosaic:
                    if flag and (i + 4) < n:
                        image, box = self.get_random_data_with_Mosaic(image_list[i:i + 4], label_list[i:i + 4],
                                                                      input_shape)
                        # image /= 255.
                        i = (i + 1) % n
                    else:
                        image, box = self.get_random_data(image_list[i], label_list[i], input_shape)
                        # image /= 255.
                        i = (i + 1) % n
                    flag = bool(1 - flag)
                else:
                    image, box = self.get_random_data(image_list[i], label_list[i], input_shape)
                    # image /= 255.
                    i = (i + 1) % n
                image_data.append(image)
                box_data.append(box)
            image_data = np.array(image_data)
            box_data = np.array(box_data)

            y_true = self.preprocess_true_boxes(box_data, input_shape, anchors, num_classes)

            yield [image_data, *y_true], np.zeros(batch_size)


class Efficientdet_Generator(object):
    def __init__(self, bbox_util, batch_size,
                 image_list, label_list, image_size, num_classes,
                 ):
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.image_list = image_list
        self.image_label = label_list
        self.image_size = image_size
        self.num_classes = num_classes

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def preprocess_input(self, image):
        image /= 255
        mean = (0.406, 0.456, 0.485)
        std = (0.225, 0.224, 0.229)
        image -= mean
        image /= std
        return image

    def get_random_data(self, line, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):

        image = Image.open(line)
        iw, ih = image.size
        h, w = input_shape
        box = label

        # resize image
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def generate(self, eager=False):
        while True:
            lines = self.image_list
            label = self.image_label
            inputs = []
            target0 = []
            target1 = []
            # n = len(lines)
            for i in range(len(lines)):
                img, y = self.get_random_data(lines[i], label[i], self.image_size[0:2])
                # i = (i + 1) % n
                if len(y) != 0:
                    boxes = np.array(y[:, :4], dtype=np.float32)
                    boxes[:, 0] = boxes[:, 0] / self.image_size[1]
                    boxes[:, 1] = boxes[:, 1] / self.image_size[0]
                    boxes[:, 2] = boxes[:, 2] / self.image_size[1]
                    boxes[:, 3] = boxes[:, 3] / self.image_size[0]
                    one_hot_label = np.eye(self.num_classes)[np.array(y[:, 4], np.int32)]

                    y = np.concatenate([boxes, one_hot_label], axis=-1)

                # 计算真实框对应的先验框，与这个先验框应当有的预测结果
                assignment = self.bbox_util.assign_boxes(y)
                regression = assignment[:, :5]
                classification = assignment[:, 5:]

                inputs.append(self.preprocess_input(img))
                target0.append(np.reshape(regression, [-1, 5]))
                target1.append(np.reshape(classification, [-1, self.num_classes + 1]))
                if len(target0) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = [np.array(target0, dtype=np.float32), np.array(target1, dtype=np.float32)]
                    inputs = []
                    target0 = []
                    target1 = []
                    if eager:
                        yield tmp_inp, tmp_targets[0], tmp_targets[1]
                    else:
                        yield tmp_inp, tmp_targets


# 打包数据
class WriteTFRecord(object):
    @staticmethod
    def pad_image(image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        iw, ih = image.size
        w, h = IMAGE_WIDTH, IMAGE_HEIGHT
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        if IMAGE_CHANNALS == 3:
            new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
            new_image = new_image.convert('L')
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            new_image = new_image.point(table, 'L')
        image = np.array(new_image)
        image = Image.fromarray(image)
        image_bytearr = io.BytesIO()
        image.save(image_bytearr, format='JPEG')
        # plt.imshow(image)
        # plt.show()
        image_bytes = image_bytearr.getvalue()
        return image_bytes

    @staticmethod
    def WriteTFRecord(TFRecord_path, datasets: list, labels: list, file_name='dataset', spilt=100, mode=MODE):
        number = 0
        if mode == 'CTC' or mode == 'CTC_TINY':
            num_count = len(datasets)
            labels_count = len(labels)
            if not os.path.exists(TFRecord_path):
                os.mkdir(TFRecord_path)
            logger.info(f'文件个数为:{num_count}')
            logger.info(f'标签个数为:{labels_count}')
            while True:
                if datasets:
                    number = number + 1
                    image_list = datasets[:spilt]
                    label_list = labels[:spilt]
                    for i in image_list:
                        datasets.remove(i)
                    for i in label_list:
                        labels.remove(i)
                    filename = file_name + str(number) + '.tfrecords'
                    filename = os.path.join(TFRecord_path, filename)
                    writer = tf.io.TFRecordWriter(filename)
                    logger.info(f'开始保存{filename}')
                    for image, label in zip(image_list, label_list):
                        start_time = time.time()
                        num_count -= 1
                        image_bytes = WriteTFRecord.pad_image(image)
                        logger.info(image)
                        logger.info(f'剩余{num_count}图片待打包')
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                                         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))}))
                        # 序列化
                        serialized = example.SerializeToString()
                        writer.write(serialized)
                        end_time = time.time()
                        now_time = end_time - start_time
                        time_list.append(now_time)
                        logger.debug(f'已耗时: {running_time(end_time - start)}')
                        logger.debug(f'预计耗时: {running_time(np.mean(time_list) * num_count)}')
                    logger.info(f'保存{filename}成功')
                    writer.close()
                else:
                    return None
        else:
            num_count = len(datasets)
            labels_count = len(labels)
            if not os.path.exists(TFRecord_path):
                os.mkdir(TFRecord_path)
            logger.info(f'文件个数为:{num_count}')
            logger.info(f'标签个数为:{labels_count}')
            while True:
                if datasets:
                    number = number + 1
                    image_list = datasets[:spilt]
                    label_list = labels[:spilt]
                    for i in image_list:
                        datasets.remove(i)
                    for i in label_list:
                        labels.remove(i)
                    filename = file_name + str(number) + '.tfrecords'
                    filename = os.path.join(TFRecord_path, filename)
                    writer = tf.io.TFRecordWriter(filename)
                    logger.info(f'开始保存{filename}')
                    for image, label in zip(image_list, label_list):
                        start_time = time.time()
                        num_count -= 1
                        image_bytes = WriteTFRecord.pad_image(image)
                        logger.info(image)
                        logger.info(f'剩余{num_count}图片待打包')
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                                         'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))}))
                        # 序列化
                        serialized = example.SerializeToString()
                        writer.write(serialized)
                        end_time = time.time()
                        now_time = end_time - start_time
                        time_list.append(now_time)
                        logger.debug(f'已耗时: {running_time(end_time - start)}')
                        logger.debug(f'预计耗时: {running_time(np.mean(time_list) * num_count)}')
                    logger.info(f'保存{filename}成功')
                    writer.close()
                else:
                    return None


# 映射函数
def parse_function(exam_proto, mode=MODE):
    if mode == 'ORDINARY':
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            make_dict = json.loads(f.read())
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([CAPTCHA_LENGTH, len(make_dict) + 1], tf.float32)
        }
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return (img_tensor, label_tensor)
    elif mode == 'NUM_CLASSES':
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            make_dict = json.loads(f.read())
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([len(make_dict)], tf.float32)
        }
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return (img_tensor, label_tensor)

    elif mode == 'CTC':
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.VarLenFeature(tf.int64)
        }
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return (img_tensor, label_tensor)
    elif mode == 'CTC_TINY':
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([CAPTCHA_LENGTH], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return {'inputs': img_tensor, 'label': label_tensor,
                'input_len': np.array([input_len], dtype=np.int64),
                'label_len': np.array([CAPTCHA_LENGTH], dtype=np.int64)}, np.ones(1, dtype=np.float32)
    else:
        raise ValueError(f'没有mode={mode}映射的方法')


class YOLO_Predict_Image(object):

    def __init__(self, model_path, score=0.5, iou=0.3, eager=False, **kwargs):
        self.model_path = model_path
        self.score = score
        self.iou = iou
        self.eager = eager
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.sess = KT.get_session()
        self.anchors = YOLO_anchors.get_anchors()
        self.load_model()

    def letterbox_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def load_model(self):
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            self.class_names = list(json.loads(f.read()).values())
        # self.class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
        #                     'traffic light',
        #                     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        #                     'sheep', 'cow',
        #                     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        #                     'suitcase', 'frisbee',
        #                     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        #                     'surfboard',
        #                     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        #                     'apple',
        #                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        #                     'sofa',
        #                     'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
        #                     'keyboard',
        #                     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        #                     'vase', 'scissors',
        #                     'teddy bear', 'hair drier', 'toothbrush']

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        if MODE == 'YOLO':
            self.yolo_model = Yolo_model.yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), num_anchors // 3,
                                                   num_classes)
        elif MODE == 'YOLO_TINY':
            self.yolo_model = Yolo_tiny_model.yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), num_anchors // 2,
                                                        num_classes)
        self.yolo_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)

        # print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        if self.eager:
            self.input_image_shape = tf.keras.layers.Input([2, ], batch_size=1)
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = tf.keras.layers.Lambda(YOLO_anchors.yolo_eval, output_shape=(1,), name='yolo_eval',
                                             arguments={'anchors': self.anchors, 'num_classes': len(self.class_names),
                                                        'image_shape': (IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        'score_threshold': self.score, 'eager': True})(inputs)
            self.yolo_model = tf.keras.Model([self.yolo_model.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2,))

            self.boxes, self.scores, self.classes = YOLO_anchors.yolo_eval(self.yolo_model.output, self.anchors,
                                                                           num_classes, self.input_image_shape,
                                                                           score_threshold=self.score,
                                                                           iou_threshold=self.iou)

    def predict_image(self, image):
        start = timer()
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        new_image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
        boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        self.yolo_model.summary()
        if self.eager:
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.yolo_model.predict([image_data, input_image_shape])
        else:
            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    KT.learning_phase(): 0
                })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # small_pic = []
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
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

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

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

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


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


class Efficientdet_Predict_Image(object):
    def __init__(self, model_path, **kwargs):
        self.image_sizes = IMAGE_SIZES
        self.phi = PHI
        self.iou = 0.3
        self.model_path = model_path
        self.model_image_size = [self.image_sizes[self.phi], self.image_sizes[self.phi], 3]
        self.prior = self._get_prior()
        self.confidence = 0.4
        self.load_model()

    def _get_prior(self):
        return Efficientdet_anchors.get_anchors(self.image_sizes[self.phi])

    def load_model(self):
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            self.class_names = list(json.loads(f.read()).values())
        # 计算总的种类
        self.num_classes = len(self.class_names)
        self.bbox_util = EfficientDet_BBoxUtility(self.num_classes, nms_thresh=self.iou)

        # 载入模型
        self.Efficientdet = Models.captcha_model_efficientdet()
        self.Efficientdet.load_weights(self.model_path, by_name=True, skip_mismatch=True)

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

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
        new_image = Image.new('RGB', size, (0, 0, 0))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

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

    def predict_image(self, image):
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = self.letterbox_image(image, [self.model_image_size[0], self.model_image_size[1]])
        photo = np.array(crop_img, dtype=np.float32)

        # 图片预处理，归一化
        photo = np.reshape(self.preprocess_input(photo),
                           [1, self.model_image_size[0], self.model_image_size[1], self.model_image_size[2]])

        preds = self.Efficientdet.predict(photo)
        # 将预测结果进行解码

        results = self.bbox_util.detection_out(preds, self.prior, confidence_threshold=self.confidence)

        if len(results[0]) <= 0:
            return image
        results = np.array(results)

        # 筛选出其中得分高于confidence的框
        det_label = results[0][:, 5]
        det_conf = results[0][:, 4]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:, 3]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
            det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(det_ymax[top_indices],
                                                                                                  -1)

        # 去掉灰条
        boxes = self.efficientdet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                np.array([self.model_image_size[0], self.model_image_size[1]]),
                                                image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

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
        return image


class Predict_Image(object):
    def __init__(self, model_path=None, app=False, classification=False):
        self.iou = 0.3
        self.app = app
        self.confidence = CONFIDENCE
        self.model_path = model_path
        self.get_parameter()
        self.load_model()
        self.classification = classification

    def get_parameter(self):
        if MODE == 'EFFICIENTDET':
            self.prior = self._get_prior()
        elif MODE == 'YOLO' or MODE == 'YOLO_TINY':
            tf.compat.v1.disable_eager_execution()
            self.score = CONFIDENCE
            self.sess = KT.get_session()
            self.anchors = YOLO_anchors.get_anchors()

    def load_model(self):
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            result = f.read()
        self.num_classes_dict = json.loads(result)
        self.num_classes_list = list(json.loads(result).values())
        self.num_classes = len(self.num_classes_list)
        if MODE == 'EFFICIENTDET' or MODE == 'SSD':
            if PRUNING:
                self.model = tf.lite.Interpreter(model_path=self.model_path)
                self.model.allocate_tensors()
            else:
                self.model = operator.methodcaller(MODEL)(Models)
                self.model.load_weights(self.model_path)
            logger.debug('加载模型到内存')
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
            num_anchors = len(self.anchors)
            num_classes = len(self.num_classes_list)
            # 画框设置不同的颜色
            if MODE == 'YOLO':
                self.model = Yolo_model.yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), num_anchors // 3,
                                                  num_classes)
            elif MODE == 'YOLO_TINY':
                self.model = Yolo_tiny_model.yolo_body(tf.keras.layers.Input(shape=(None, None, 3)),
                                                       num_anchors // 2,
                                                       num_classes)
            self.model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
            logger.debug('加载模型到内存')
            # print('{} model, anchors, and classes loaded.'.format(model_path))

            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.num_classes_list), 1., 1.)
                          for x in range(len(self.num_classes_list))]
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
                new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
                new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            else:
                new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
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
                    new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
                    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
                else:
                    new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
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
        global mean_time
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
            mean_time.append(end_time - start_time)
            logger.info(f'识别时间为{end_time - start_time}s')
            logger.info(f'平均识别时间为{np.mean(mean_time)}s')
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
                    # KT.learning_phase(): 0
                })
            if len(out_boxes) <= 0:
                return image
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
            mean_time.append(end_time - start_time)
            logger.info(f'识别时间为{end_time - start_time}s')
            logger.info(f'平均识别时间为{np.mean(mean_time)}s')
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
            results = self.bbox_util.detection_out(preds, confidence_threshold=self.confidence)
            if len(results[0]) <= 0:
                return image
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:,
                                                                                                           5]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
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
            mean_time.append(end_time - start_time)
            logger.info(f'识别时间为{end_time - start_time}s')
            logger.info(f'平均识别时间为{np.mean(mean_time)}s')
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
            mean_time.append(end_time - start_time)
            logger.info(f'已识别{right_value}张图片')
            logger.info(f'识别时间为{end_time - start_time}s')
            logger.info(f'平均识别时间为{np.mean(mean_time)}s')
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
                    # KT.learning_phase(): 0
                })
            if len(out_boxes) <= 0:
                return {'times': str(time.time() - start_time)}

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


def cheak_path(path):
    number = 0
    while True:
        if os.path.exists(path):
            paths, name = os.path.split(path)
            name, mix = os.path.splitext(name)
            number = number + 1
            name = re.split('_', name)[0]
            name = name + f'_V{number}.0'
            path = os.path.join(paths, name + mix)
        else:
            return path


def running_time(time):
    m = time / 60
    h = m / 60
    if m > 1:
        if h > 1:
            return str('%.2f' % h) + 'h'
        else:
            return str('%.2f' % m) + 'm'
    else:
        return str('%.2f' % time) + 's'


def MD5(str_input):
    m5 = hashlib.md5()
    m5.update(str(str_input).encode('utf-8'))
    str_input = m5.hexdigest()
    return str_input

"""

    project(string, work_path='works', project_name='simple')
