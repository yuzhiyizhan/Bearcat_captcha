import os


def callback(work_path, project_name):
    return f"""import re
import os
import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm.keras import TqdmCallback
from tensorflow.keras import backend as K
from {work_path}.{project_name}.settings import LR
from {work_path}.{project_name}.settings import MODE
from {work_path}.{project_name}.settings import EPOCHS
from {work_path}.{project_name}.settings import LOG_DIR
from {work_path}.{project_name}.settings import CSV_PATH
from {work_path}.{project_name}.settings import BATCH_SIZE
from {work_path}.{project_name}.settings import TRAIN_PATH
from {work_path}.{project_name}.settings import UPDATE_FREQ
from {work_path}.{project_name}.settings import LR_PATIENCE
from {work_path}.{project_name}.settings import EARLY_PATIENCE
from {work_path}.{project_name}.settings import CHECKPOINT_PATH
from {work_path}.{project_name}.settings import COSINE_SCHEDULER
from {work_path}.{project_name}.settings import CHECKPOINT_FILE_PATH
from {work_path}.{project_name}.utils import Image_Processing

# 开启可视化的命令
'''
tensorboard --logdir "logs"
'''


# 回调函数官方文档
# https://keras.io/zh/callbacks/
class CallBack(object):
    @classmethod
    def calculate_the_best_weight(self):
        if os.listdir(CHECKPOINT_PATH):
            value = Image_Processing.extraction_image(CHECKPOINT_PATH)
            extract_num = [os.path.splitext(os.path.split(i)[-1])[0] for i in value]
            num = [re.split('-', i) for i in extract_num]
            losses = [float('-' + str(abs(float(i[2])))) for i in num]
            model_dict = dict((ind, val) for ind, val in zip(losses, value))
            return model_dict.get(max(losses))
        else:
            logger.debug('没有可用的检查点')

    @classmethod
    def cosine_scheduler(self):
        train_number = len(Image_Processing.extraction_image(TRAIN_PATH))
        warmup_epoch = int(EPOCHS * 0.2)
        total_steps = int(EPOCHS * train_number / BATCH_SIZE)
        warmup_steps = int(warmup_epoch * train_number / BATCH_SIZE)
        cosine_scheduler_callback = WarmUpCosineDecayScheduler(learning_rate_base=LR, total_steps=total_steps,
                                                               warmup_learning_rate=LR * 0.1,
                                                               warmup_steps=warmup_steps,
                                                               hold_base_rate_steps=train_number,
                                                               min_learn_rate=LR * 0.2)
        return cosine_scheduler_callback

    @classmethod
    def callback(self, model):
        call = []
        if os.path.exists(CHECKPOINT_PATH):
            if os.listdir(CHECKPOINT_PATH):
                logger.debug('load the model')
                model.load_weights(os.path.join(CHECKPOINT_PATH, self.calculate_the_best_weight()))
                logger.debug(f'读取的权重为{{os.path.join(CHECKPOINT_PATH, self.calculate_the_best_weight())}}')
        if MODE == 'YOLO' or MODE == 'YOLO_TINY' or MODE == 'EFFICIENTDET':
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH,
                                                             verbose=1,
                                                             monitor='loss',
                                                             save_weights_only=True,
                                                             save_best_only=False, period=1)
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH,
                                                             verbose=1,
                                                             save_weights_only=True,
                                                             save_best_only=False, period=1)
        call.append(cp_callback)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_images=False,
                                                              update_freq=UPDATE_FREQ, write_graph=False)
        call.append(tensorboard_callback)
        if COSINE_SCHEDULER:
            lr_callback = self.cosine_scheduler()
        else:
            if MODE == 'YOLO' or MODE == 'YOLO_TINY' or MODE == 'EFFICIENTDET':
                lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=LR_PATIENCE)
            else:
                lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=LR_PATIENCE)
        call.append(lr_callback)

        csv_callback = tf.keras.callbacks.CSVLogger(filename=CSV_PATH, append=True)
        call.append(csv_callback)
        if MODE == 'YOLO' or MODE == 'YOLO_TINY' or MODE == 'EFFICIENTDET':
            early_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, verbose=1,
                                                              patience=EARLY_PATIENCE)
        else:
            early_callback = tf.keras.callbacks.EarlyStopping(min_delta=0, verbose=1,
                                                              patience=EARLY_PATIENCE)
        call.append(early_callback)
        call.append(TqdmCallback())
        return (model, call)


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate_base, total_steps, global_step_init=0, warmup_learning_rate=0.0, warmup_steps=0,
                 hold_base_rate_steps=0, min_learn_rate=0., verbose=1):
        super(WarmUpCosineDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

        self.interval_epoch = [0.05, 0.15, 0.30, 0.50]
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch) - 1):
            self.interval_reset.append(self.interval_epoch[i + 1] - self.interval_epoch[i])
        self.interval_reset.append(1 - self.interval_epoch[-1])

    def cosine_decay_with_warmup(self, global_step, learning_rate_base, total_steps, warmup_learning_rate=0.0,
                                 warmup_steps=0,
                                 hold_base_rate_steps=0, min_learn_rate=0, ):
        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                             'warmup_steps.')
        # 这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
        learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
                                                               (
                                                                       global_step - warmup_steps - hold_base_rate_steps) / float(
            total_steps - warmup_steps - hold_base_rate_steps)))
        # 如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                     learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            # 线性增长的实现
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            # 只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate)

        learning_rate = max(learning_rate, min_learn_rate)
        return learning_rate

    # 更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        self.global_step_for_interval = self.global_step_for_interval + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_batch_begin(self, batch, logs=None):
        # 每到一次最低点就重新更新参数
        if self.global_step_for_interval in [0] + [int(i * self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = self.cosine_decay_with_warmup(global_step=self.global_step,
                                           learning_rate_base=self.learning_rate_base,
                                           total_steps=self.total_steps,
                                           warmup_learning_rate=self.warmup_learning_rate,
                                           warmup_steps=self.warmup_steps,
                                           hold_base_rate_steps=self.hold_base_rate_steps,
                                           min_learn_rate=self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            logger.info(f'Batch {{self.global_step}}: setting learning rate to {{lr}}.')


if __name__ == '__main__':
    logger.debug(CallBack.calculate_the_best_weight())

"""


def app(work_path, project_name):
    return f"""import os
import json
import numpy as np
import gradio as gr
import tensorflow as tf
from loguru import logger
from {work_path}.{project_name}.settings import USE_GPU
from {work_path}.{project_name}.settings import APP_MODEL_PATH
from {work_path}.{project_name}.utils import Predict_Image

if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    if gpus:
        logger.info("use gpu device")
        logger.info(f'可用GPU数量: {{len(gpus)}}')
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

if APP_MODEL_PATH:
    model_path = os.path.join(APP_MODEL_PATH, os.listdir(APP_MODEL_PATH)[0])
    logger.debug(f'{{model_path}}模型读取成功')
else:
    raise OSError(f'{{APP_MODEL_PATH}}没有存放模型文件')

Predict = Predict_Image(model_path=model_path, app=True)


def show_label(image):
    return_dict = Predict.api(image=image)
    return json.dumps(return_dict, ensure_ascii=False)


def show_image(image):
    return Predict.predict_image(image)


if __name__ == '__main__':
    gr.Interface(show_label, gr.inputs.Image(), "label").launch(share=False)
    # gr.Interface(show_image, gr.inputs.Image(), "image").launch()

"""


def captcha_config():
    return '''{
  "train_dir": "train_dataset",
  "validation_dir": "validation_dataset",
  "test_dir": "test_dataset",
  "image_suffix": "jpg",
  "characters": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
  "count": 20000,
  "char_count": [4,5,6],
  "width": 100,
  "height": 60
}
'''


def check_file(work_path, project_name):
    return f"""import os
import numpy as np
from PIL import Image
from loguru import logger
from {work_path}.{project_name}.settings import MODE
from {work_path}.{project_name}.settings import TRAIN_PATH
from {work_path}.{project_name}.settings import VALIDATION_PATH
from {work_path}.{project_name}.settings import TEST_PATH
from {work_path}.{project_name}.settings import LABEL_PATH
from {work_path}.{project_name}.utils import Image_Processing
from concurrent.futures import ThreadPoolExecutor

train_image = Image_Processing.extraction_image(TRAIN_PATH)


def cheak_image(image_path):
    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file)
        # if image.mode != 'RGB':
        #     logger.error(f'{{image}}模式为{{image.mode}}')
        try:
            image.verify()
            width, height = image.size
            width_list.append(width)
            height_list.append(height)
        except SyntaxError as e:
            logger.error(e)
            try:
                logger.error(f'图片: {{image_path}} 已损坏')
                image_file.close()
                os.remove(image_path)
                logger.debug(f'已删除{{image_path}}')
            except Exception as e:
                logger.error(e)


validation_image = Image_Processing.extraction_image(VALIDATION_PATH)

test_image = Image_Processing.extraction_image(TEST_PATH)

width_list = []
height_list = []

image_list = train_image + validation_image + test_image

if MODE == 'YOLO' or MODE == 'YOLO_TINY' or MODE == 'EFFICIENTDET':
    logger.debug(f'标签总数{{len(Image_Processing.extraction_image(LABEL_PATH))}}')

with ThreadPoolExecutor(max_workers=20) as t:
    for i in image_list:
        task = t.submit(cheak_image, i)



logger.debug(f'图片总数{{len(image_list)}}')
logger.info(f'所有图片最大的高为{{np.max(height_list)}}')
logger.info(f'所有图片最大的宽为{{np.max(width_list)}}')

"""


def delete_file(work_path, project_name):
    return f"""# 增强后文件太多，手动删非常困难，直接用代码删
import shutil
from loguru import logger
from {work_path}.{project_name}.settings import WEIGHT
from {work_path}.{project_name}.settings import TRAIN_PATH
from {work_path}.{project_name}.settings import TEST_PATH
from {work_path}.{project_name}.settings import LABEL_PATH
from {work_path}.{project_name}.settings import VALIDATION_PATH
from {work_path}.{project_name}.settings import TRAIN_PACK_PATH
from {work_path}.{project_name}.settings import VALIDATION_PACK_PATH
from concurrent.futures import ThreadPoolExecutor


def del_file(path):
    try:
        shutil.rmtree(path)
        logger.debug(f'成功删除{{path}}')
    except WindowsError as e:
        logger.error(e)


if __name__ == '__main__':
    path = [TRAIN_PATH, TEST_PATH, VALIDATION_PATH, TRAIN_PACK_PATH, VALIDATION_PACK_PATH, LABEL_PATH,
            WEIGHT]
    with ThreadPoolExecutor(max_workers=50) as t:
        for i in path:
            t.submit(del_file, i)

"""


def utils(work_path, project_name):
    return f"""import io
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
from {work_path}.{project_name}.models import Models
from {work_path}.{project_name}.models import Yolo_model
from {work_path}.{project_name}.models import YOLO_anchors
from {work_path}.{project_name}.models import Yolo_tiny_model
from {work_path}.{project_name}.models import Efficientdet_anchors
from {work_path}.{project_name}.settings import PHI
from {work_path}.{project_name}.settings import MODE
from {work_path}.{project_name}.settings import MODEL
from {work_path}.{project_name}.settings import DIVIDE
from {work_path}.{project_name}.settings import PRUNING
from {work_path}.{project_name}.settings import THRESHOLD
from {work_path}.{project_name}.settings import MAX_BOXES
from {work_path}.{project_name}.settings import TEST_PATH
from {work_path}.{project_name}.settings import LABEL_PATH
from {work_path}.{project_name}.settings import BASIC_PATH
from {work_path}.{project_name}.settings import IMAGE_WIDTH
from {work_path}.{project_name}.settings import IMAGE_SIZES
from {work_path}.{project_name}.settings import DIVIDE_RATO
from {work_path}.{project_name}.settings import IMAGE_HEIGHT
from {work_path}.{project_name}.settings import NUMBER_CLASSES_FILE
from {work_path}.{project_name}.settings import CAPTCHA_LENGTH
from {work_path}.{project_name}.settings import IMAGE_CHANNALS
from {work_path}.{project_name}.settings import VALIDATION_PATH
from {work_path}.{project_name}.settings import DATA_ENHANCEMENT
from {work_path}.{project_name}.settings import TRAIN_PATH
from concurrent.futures import ThreadPoolExecutor

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
        input_len = np.int64(Models.captcha_ctc_tiny().get_layer('reshape_len').output_shape[1])
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
        elif mode == 'YOLO' or mode == 'YOLO_TINY' or mode == 'EFFICIENTDET':
            n_class = []
            paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
            try:
                label = [(i, glob.glob(f'{{LABEL_PATH}}/*/{{i}}.xml')[0]) for i in paths]
            except:
                label = [(i, glob.glob(f'{{LABEL_PATH}}/{{i}}.xml')[0]) for i in paths]
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
            label_dict = {{}}
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
            raise ValueError(f'没有mode={{mode}}提取标签的方法')

    @classmethod
    def text2vector(self, label, make_dict: dict, mode=MODE):
        if mode == 'ORDINARY':
            if len(label) > CAPTCHA_LENGTH:
                raise ValueError(f'标签{{label}}长度大于预设值{{CAPTCHA_LENGTH}},建议设置CAPTCHA_LENGTH为{{len(label) + 2}}')
            num_classes = len(make_dict)
            label_ver = np.ones((CAPTCHA_LENGTH), dtype=np.int64) * num_classes
            for index, c in enumerate(label):
                if not make_dict.get(c):
                    raise ValueError(f'错误的值{{c}}')
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
                    raise ValueError(f'错误的值{{c}}')
                label_ver.append(int(make_dict.get(c)))
            label_ver = np.array(label_ver)
            return label_ver
        elif mode == 'CTC_TINY':
            if len(label) > CAPTCHA_LENGTH:
                raise ValueError(f'标签{{label}}长度大于预设值{{CAPTCHA_LENGTH}},建议设置CAPTCHA_LENGTH为{{len(label) + 2}}')
            num_classes = len(make_dict)
            label_ver = np.ones((CAPTCHA_LENGTH), dtype=np.int64) * num_classes
            for index, c in enumerate(label):
                if not make_dict.get(c):
                    raise ValueError(f'错误的值{{c}}')
                label_ver[index] = make_dict.get(c)
            # label_ver = list(tf.keras.utils.to_categorical(label_ver, num_classes=num_classes + 1).ravel())
            return label_ver
        else:
            raise ValueError(f'没有mode={{mode}}提取标签的方法')

    @classmethod
    def _shutil_move(self, full_path, des_path, number):
        shutil.move(full_path, des_path)
        logger.debug(f'剩余数量{{number}}')

    # 分割数据集
    @classmethod
    def split_dataset(self, path: list, proportion=DIVIDE_RATO) -> bool:
        if DIVIDE:
            number = 0
            logger.debug(f'数据集有{{len(path)}},{{proportion * 100}}%作为验证集,{{proportion * 100}}%作为测试集')
            division_number = int(len(path) * proportion)
            logger.debug(f'验证集数量为{{division_number}},测试集数量为{{division_number}}')
            validation_dataset = random.sample(path, division_number)
            with ThreadPoolExecutor(max_workers=50) as t:
                for i in validation_dataset:
                    number = number + 1
                    logger.debug(f'准备移动{{(number / len(validation_dataset)) * 100}}%')
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
            logger.debug(f'数据集有{{len(path)}},{{proportion * 100}}%作为测试集')
            division_number = int(len(path) * proportion)
            logger.debug(f'测试集数量为{{division_number}}')
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
    #     logger.info(f'开始处理{{image}}')
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
    #     logger.success(f'处理完成{{image}},还剩{{number}}张图片待增强')

    @classmethod
    def preprosess_save_images(self, image, number):
        logger.info(f'开始处理{{image}}')
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
        logger.success(f'处理完成{{image}},还剩{{number}}张图片待增强')

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
            new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
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

        grid_shapes = [input_shape // {{0: 32, 1: 16, 2: 8}}[l] for l in range(num_layers)]

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
            n = len(lines)
            for i in range(len(lines)):
                img, y = self.get_random_data(lines[i], label[i], self.image_size[0:2])
                i = (i + 1) % n
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
            new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = Image.new('P', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
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
            logger.info(f'文件个数为:{{num_count}}')
            logger.info(f'标签个数为:{{labels_count}}')
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
                    logger.info(f'开始保存{{filename}}')
                    for image, label in zip(image_list, label_list):
                        start_time = time.time()
                        num_count -= 1
                        image_bytes = WriteTFRecord.pad_image(image)
                        logger.info(f'剩余{{num_count}}图片待打包')
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={{'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                                         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))}}))
                        # 序列化
                        serialized = example.SerializeToString()
                        writer.write(serialized)
                        end_time = time.time()
                        now_time = end_time - start_time
                        time_list.append(now_time)
                        logger.debug(f'已耗时: {{running_time(end_time - start)}}')
                        logger.debug(f'预计耗时: {{running_time(np.mean(time_list) * num_count)}}')
                    logger.info(f'保存{{filename}}成功')
                    writer.close()
                else:
                    return None
        else:
            num_count = len(datasets)
            labels_count = len(labels)
            if not os.path.exists(TFRecord_path):
                os.mkdir(TFRecord_path)
            logger.info(f'文件个数为:{{num_count}}')
            logger.info(f'标签个数为:{{labels_count}}')
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
                    logger.info(f'开始保存{{filename}}')
                    for image, label in zip(image_list, label_list):
                        start_time = time.time()
                        num_count -= 1
                        image_bytes = WriteTFRecord.pad_image(image)
                        logger.info(f'剩余{{num_count}}图片待打包')
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={{'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                                         'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))}}))
                        # 序列化
                        serialized = example.SerializeToString()
                        writer.write(serialized)
                        end_time = time.time()
                        now_time = end_time - start_time
                        time_list.append(now_time)
                        logger.debug(f'已耗时: {{running_time(end_time - start)}}')
                        logger.debug(f'预计耗时: {{running_time(np.mean(time_list) * num_count)}}')
                    logger.info(f'保存{{filename}}成功')
                    writer.close()
                else:
                    return None


# 映射函数
def parse_function(exam_proto, mode=MODE):
    if mode == 'ORDINARY':
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            make_dict = json.loads(f.read())
        features = {{
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([CAPTCHA_LENGTH, len(make_dict) + 1], tf.float32)
        }}
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return (img_tensor, label_tensor)
    elif mode == 'NUM_CLASSES':
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            make_dict = json.loads(f.read())
        features = {{
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([len(make_dict)], tf.float32)
        }}
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return (img_tensor, label_tensor)

    elif mode == 'CTC':
        features = {{
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.VarLenFeature(tf.int64)
        }}
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return (img_tensor, label_tensor)
    elif mode == 'CTC_TINY':
        features = {{
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([CAPTCHA_LENGTH], tf.int64)
        }}
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        label_tensor = parsed_example['label']
        return {{'inputs': img_tensor, 'label': label_tensor,
                'input_len': np.array([input_len], dtype=np.int64),
                'label_len': np.array([CAPTCHA_LENGTH], dtype=np.int64)}}, np.ones(1, dtype=np.float32)
    else:
        raise ValueError(f'没有mode={{mode}}映射的方法')


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

        # print('{{}} model, anchors, and classes loaded.'.format(model_path))

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
                                             arguments={{'anchors': self.anchors, 'num_classes': len(self.class_names),
                                                        'image_shape': (IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        'score_threshold': self.score, 'eager': True}})(inputs)
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
        if self.eager:
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.yolo_model.predict([image_data, input_image_shape])
        else:
            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={{
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    KT.learning_phase(): 0
                }})

        print('Found {{}} boxes for {{}}'.format(len(out_boxes), 'img'))
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
            label = '{{}} {{:.2f}}'.format(predicted_class, score)
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


class BBoxUtility(object):
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
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.iou)

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
            label = '{{}} {{:.2f}}'.format(predicted_class, score)
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
        self.confidence = 0.4
        self.model_path = model_path
        self.load_model()
        self.classification = classification

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
        if MODE == 'EFFICIENTDET':

            self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.iou)
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
        else:
            raise ValueError('new_image error')
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
            raise ValueError(f'还没写{{MODE}}这种预测方法')

    def predict_image(self, image):
        global right_value
        global predicted_value
        start_time = time.time()
        recognition_rate_list = []
        if MODE == 'EFFICIENTDET':
            if self.app:
                image = Image.fromarray(image)
            else:
                image = Image.open(image)
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
                    data = {{'data': [f'data:image;base64,{{base64.b64encode(image_bytes).decode()}}']}}
                    response = requests.post('http://127.0.0.1:7860/api/predict/', json=data).json()
                    result = json.loads(response.get('data')[0].get('label'))
                    predicted_class = result.get('result')
                    recognition_rate = result.get('recognition_rate')
                    recognition_rate = float(recognition_rate.replace('%', '')) / 100
                    recognition_rate_list.append(recognition_rate)
                # 画框框
                label = '{{}} {{:.2f}}'.format(predicted_class, score)
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
            logger.info(f'识别时间为{{end_time - start_time}}s')
            logger.info(f'总体置信度为{{round(self.recognition_probability(recognition_rate_list), 2) * 100}}%')
            return image

        elif MODE == 'YOLO' or MODE == 'YOLO_TINY':
            if self.app:
                image = Image.fromarray(image)
            else:
                image = Image.open(image)
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
                feed_dict={{
                    self.model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    KT.learning_phase(): 0
                }})

            # logger.debug('Found {{}} boxes for {{}}'.format(len(out_boxes), 'img'))
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
                label = '{{}} {{:.2f}}'.format(predicted_class, score)
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
            logger.info(f'识别时间为{{end_time - start_time}}s')
            logger.info(f'总体置信度为{{round(self.recognition_probability(recognition_rate_list), 2) * 100}}%')
            return image

        else:
            if PRUNING:
                model = self.model
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                image_object = self.decode_image(image)
                model.set_tensor(input_details[0]['index'], image_object)
                model.invoke()
                vertor = model.get_tensor(output_details[0]['index'])
            else:
                model = self.model
                image_object = self.decode_image(image)
                vertor = model.predict(image_object)
            text, recognition_rate = self.decode_vector(vector=vertor, num_classes=self.num_classes_dict)
            right_text = self.decode_label(image)
            logger.info(f'预测为{{text}},真实为{{right_text}}') if text == right_text else logger.error(
                f'预测为{{text}},真实为{{right_text}}')
            logger.info(f'识别率为:{{recognition_rate * 100}}%') if recognition_rate > 0.7 else logger.error(
                f'识别率为:{{recognition_rate * 100}}%')
            if str(text) != str(right_text):
                logger.error(f'预测失败的图片路径为:{{image}}')
                right_value = right_value + 1
                logger.info(f'正确率:{{(predicted_value / right_value) * 100}}%')
                if predicted_value > 0:
                    logger.info(f'预测正确{{predicted_value}}张图片')
            else:
                predicted_value = predicted_value + 1
                right_value = right_value + 1
                logger.info(f'正确率:{{(predicted_value / right_value) * 100}}%')
                if predicted_value > 0:
                    logger.info(f'预测正确{{predicted_value}}张图片')
            end_time = time.time()
            logger.info(f'已识别{{right_value}}张图片')
            logger.info(f'识别时间为{{end_time - start_time}}s')
            return Image.fromarray(image_object[0] * 255)

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
                return {{'times': str(time.time() - start_time)}}
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
                data = {{'data': [f'data:image;base64,{{base64.b64encode(image_bytes).decode()}}']}}
                response = requests.post('http://127.0.0.1:7860/api/predict/', json=data).json()
                result = json.loads(response.get('data')[0].get('label'))
                predicted_class = result.get('result')
                recognition_rate = result.get('recognition_rate')
                recognition_rate = float(recognition_rate.replace('%', '')) / 100
                recognition_rate_list.append(recognition_rate)
                label = {{"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}}
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
                        # data = {{'data': [f'data:image;base64,{{base64.b64encode(image_bytes).decode()}}']}}
                        # response = requests.post('http://127.0.0.1:7860/api/predict/', json=data).json()
                        # result = json.loads(response.get('data')[0].get('label'))
                        # predicted_class = result.get('result')
                        # recognition_rate = result.get('recognition_rate')
                        # recognition_rate = float(recognition_rate.replace('%', '')) / 100
                        # recognition_rate_list.append(recognition_rate)
                        # label = {{"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}}
                        # result_list.append(label)
                    else:
                        label = {{"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}}
                        result_list.append(label)

            recognition_rate = self.recognition_probability(recognition_rate_list)
            end_time = time.time()
            times = end_time - start_time
            return {{'result': str(result_list), 'recognition_rate': str(round(recognition_rate * 100, 2)) + '%',
                    'times': str(times)}}

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
                feed_dict={{
                    self.model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    KT.learning_phase(): 0
                }})

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

                label = {{"label": predicted_class, "xmax": top, "ymax": left, "xmin": bottom, "ymin": right}}
                result_list.append(label)
                recognition_rate_list.append(score)

            recognition_rate = self.recognition_probability(recognition_rate_list)
            end_time = time.time()
            times = end_time - start_time
            return {{'result': str(result_list), 'recognition_rate': str(round(recognition_rate * 100, 2)) + '%',
                    'times': str(times)}}

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
            return {{'result': str(result), 'recognition_rate': str(round(recognition_rate * 100, 2)) + '%',
                    'times': str(times)}}


def cheak_path(path):
    number = 0
    while True:
        if os.path.exists(path):
            paths, name = os.path.split(path)
            name, mix = os.path.splitext(name)
            number = number + 1
            name = re.split('_', name)[0]
            name = name + f'_V{{number}}.0'
            path = os.path.join(paths, name + mix)
            return path
        if not os.path.exists(path):
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


def gen_sample_by_captcha(work_path, project_name):
    return f"""# -*- coding: UTF-8 -*-
import os
import time
import json
import random
from tqdm import tqdm
from captcha.image import ImageCaptcha
from concurrent.futures import ThreadPoolExecutor


def gen_special_img(text, file_path, width, height):
    # 生成img文件
    generator = ImageCaptcha(width=width, height=height)  # 指定大小
    img = generator.generate_image(text)  # 生成图片
    img.save(file_path)  # 保存图片


def gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height):
    # 判断文件夹是否存在
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for _ in tqdm(enumerate(range(count)), desc='Generate captcha image'):
        text = ""
        for _ in range(random.choice(char_count)):
            text += random.choice(characters)

        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{{}}_{{}}.{{}}".format(text, timec, image_suffix))
        gen_special_img(text, p, width, height)

        # logger.debug("Generate captcha image => {{}}".format(index + 1))


def main():
    with open("captcha_config.json", "r") as f:
        config = json.load(f)
    # 配置参数
    train_dir = config["train_dir"]
    validation_dir = config["validation_dir"]
    test_dir = config["test_dir"]
    image_suffix = config["image_suffix"]  # 图片储存后缀
    characters = config["characters"]  # 图片上显示的字符集 # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    count = config["count"]  # 生成多少张样本
    char_count = config["char_count"]  # 图片上的字符数量

    # 设置图片高度和宽度
    width = config["width"]
    height = config["height"]

    with ThreadPoolExecutor(max_workers=3) as t:
        t.submit(gen_ima_by_batch, train_dir, image_suffix, characters, count, char_count, width, height)
        t.submit(gen_ima_by_batch, validation_dir, image_suffix, characters, count, char_count, width, height)
        t.submit(gen_ima_by_batch, test_dir, image_suffix, characters, count, char_count, width, height)


if __name__ == '__main__':
    main()

"""


def init_working_space(work_path, project_name):
    return f"""# 检查项目路径
import os
import shutil
from loguru import logger
from {work_path}.{project_name}.settings import WEIGHT
from {work_path}.{project_name}.settings import BASIC_PATH
from {work_path}.{project_name}.settings import TRAIN_PATH
from {work_path}.{project_name}.settings import VALIDATION_PACK_PATH
from {work_path}.{project_name}.settings import TEST_PATH
from {work_path}.{project_name}.settings import TRAIN_PACK_PATH
from {work_path}.{project_name}.settings import MODEL_PATH
from {work_path}.{project_name}.settings import VALIDATION_PATH
from {work_path}.{project_name}.settings import LABEL_PATH
from {work_path}.{project_name}.settings import APP_MODEL_PATH
from {work_path}.{project_name}.settings import CHECKPOINT_PATH


def chrak_path():
    paths = [TEST_PATH, TRAIN_PATH, VALIDATION_PATH, TRAIN_PACK_PATH,
             VALIDATION_PACK_PATH, MODEL_PATH, os.path.join(BASIC_PATH, 'logs'),
             os.path.join(BASIC_PATH, 'CSVLogger'), CHECKPOINT_PATH, WEIGHT,
             LABEL_PATH, APP_MODEL_PATH]
    for i in paths:
        if not os.path.exists(i):
            os.mkdir(i)


def del_file():
    path = [os.path.join(BASIC_PATH, 'CSVLogger'),
            os.path.join(BASIC_PATH, 'logs'), CHECKPOINT_PATH]
    for i in path:
        try:
            shutil.rmtree(i)
        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    del_file()
    chrak_path()

"""


def models(work_path, project_name):
    return f"""import os
import glob
import math
import json
import copy
import random
import collections
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from loguru import logger
from functools import wraps
from six.moves import xrange
from functools import reduce
import xml.etree.ElementTree as ET
from tensorflow.keras import backend as K
from adabelief_tf import AdaBeliefOptimizer
from einops.layers.tensorflow import Rearrange
from {work_path}.{project_name}.settings import LR
from {work_path}.{project_name}.settings import PHI
from {work_path}.{project_name}.settings import MODE
from {work_path}.{project_name}.settings import WEIGHT
from {work_path}.{project_name}.settings import MAX_BOXES
from {work_path}.{project_name}.settings import LABEL_PATH
from {work_path}.{project_name}.settings import IMAGE_WIDTH
from {work_path}.{project_name}.settings import IMAGE_SIZES
from {work_path}.{project_name}.settings import ANCHORS_PATH
from {work_path}.{project_name}.settings import NUMBER_CLASSES_FILE
from {work_path}.{project_name}.settings import IMAGE_HEIGHT
from {work_path}.{project_name}.settings import CAPTCHA_LENGTH
from {work_path}.{project_name}.settings import IMAGE_CHANNALS
from {work_path}.{project_name}.settings import LABEL_SMOOTHING

inputs_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {{
    'class_name': 'VarianceScaling',
    'config': {{
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }}
}}

DENSE_KERNEL_INITIALIZER = {{
    'class_name': 'VarianceScaling',
    'config': {{
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }}
}}


########################################
## 自定义层与激活函数
########################################

class Mish_Activation(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Mish_Activation, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish_Activation'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


class SwitchNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 final_gamma=False,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 mean_weights_initializer='ones',
                 variance_weights_initializer='ones',
                 moving_mean_initializer='ones',
                 moving_variance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 mean_weights_regularizer=None,
                 variance_weights_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 mean_weights_constraints=None,
                 variance_weights_constraints=None,
                 **kwargs):
        super(SwitchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        if final_gamma:
            self.gamma_initializer = tf.keras.initializers.get('zeros')
        else:
            self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.mean_weights_initializer = tf.keras.initializers.get(mean_weights_initializer)
        self.variance_weights_initializer = tf.keras.initializers.get(variance_weights_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.mean_weights_regularizer = tf.keras.regularizers.get(mean_weights_regularizer)
        self.variance_weights_regularizer = tf.keras.regularizers.get(variance_weights_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.mean_weights_constraints = tf.keras.constraints.get(mean_weights_constraints)
        self.variance_weights_constraints = tf.keras.constraints.get(variance_weights_constraints)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape),
                                                    axes={{self.axis: dim}})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name='beta',
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)
        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)

        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)

        self.mean_weights = self.add_weight(
            shape=(3,),
            name='mean_weights',
            initializer=self.mean_weights_initializer,
            regularizer=self.mean_weights_regularizer,
            constraint=self.mean_weights_constraints)

        self.variance_weights = self.add_weight(
            shape=(3,),
            name='variance_weights',
            initializer=self.variance_weights_initializer,
            regularizer=self.variance_weights_regularizer,
            constraint=self.variance_weights_constraints)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]

        if self.axis != 0:
            del reduction_axes[0]

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        mean_instance = K.mean(inputs, reduction_axes, keepdims=True)
        variance_instance = K.var(inputs, reduction_axes, keepdims=True)

        mean_layer = K.mean(mean_instance, self.axis, keepdims=True)
        temp = variance_instance + K.square(mean_instance)
        variance_layer = K.mean(temp, self.axis, keepdims=True) - K.square(mean_layer)

        def training_phase():
            mean_batch = K.mean(mean_instance, axis=0, keepdims=True)
            variance_batch = K.mean(temp, axis=0, keepdims=True) - K.square(mean_batch)

            mean_batch_reshaped = K.flatten(mean_batch)
            variance_batch_reshaped = K.flatten(variance_batch)

            if K.backend() != 'cntk':
                sample_size = K.prod([K.shape(inputs)[axis]
                                      for axis in reduction_axes])
                sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

                # sample variance - unbiased estimator of population variance
                variance_batch_reshaped *= sample_size / (sample_size - (1.0 + self.epsilon))

            self.add_update([K.moving_average_update(self.moving_mean,
                                                     mean_batch_reshaped,
                                                     self.momentum),
                             K.moving_average_update(self.moving_variance,
                                                     variance_batch_reshaped,
                                                     self.momentum)],
                            inputs)

            return normalize_func(mean_batch, variance_batch)

        def inference_phase():
            mean_batch = self.moving_mean
            variance_batch = self.moving_variance

            return normalize_func(mean_batch, variance_batch)

        def normalize_func(mean_batch, variance_batch):
            mean_batch = K.reshape(mean_batch, broadcast_shape)
            variance_batch = K.reshape(variance_batch, broadcast_shape)

            mean_weights = K.softmax(self.mean_weights, axis=0)
            variance_weights = K.softmax(self.variance_weights, axis=0)

            mean = (mean_weights[0] * mean_instance +
                    mean_weights[1] * mean_layer +
                    mean_weights[2] * mean_batch)

            variance = (variance_weights[0] * variance_instance +
                        variance_weights[1] * variance_layer +
                        variance_weights[2] * variance_batch)

            outputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                outputs = outputs + broadcast_beta

            return outputs

        if training in {{0, False}}:
            return inference_phase()

        return K.in_train_phase(training_phase,
                                inference_phase,
                                training=training)

    def get_config(self):
        config = {{
            'axis': self.axis,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'mean_weights_initializer': tf.keras.initializers.serialize(self.mean_weights_initializer),
            'variance_weights_initializer': tf.keras.initializers.serialize(self.variance_weights_initializer),
            'moving_mean_initializer': tf.keras.initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': tf.keras.initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'mean_weights_regularizer': tf.keras.regularizers.serialize(self.mean_weights_regularizer),
            'variance_weights_regularizer': tf.keras.regularizers.serialize(self.variance_weights_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint),
            'mean_weights_constraints': tf.keras.constraints.serialize(self.mean_weights_constraints),
            'variance_weights_constraints': tf.keras.constraints.serialize(self.variance_weights_constraints),
        }}
        base_config = super(SwitchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


tf.keras.utils.get_custom_objects().update({{'Mish_Activation': Mish_Activation(mish)}})


class FRN(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=1e-6,
                 learnable_epsilon=False,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 epsilon_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 epsilon_constraint=None,
                 **kwargs):
        super(FRN, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.epsilon_regularizer = tf.keras.regularizers.get(epsilon_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.epsilon_constraint = tf.keras.constraints.get(epsilon_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape),
                                                    axes={{self.axis: dim}})
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)
        self.epsilon_l = self.add_weight(shape=(1,),
                                         name='epsilon_l',
                                         initializer=tf.keras.initializers.Constant(self.epsilon),
                                         regularizer=self.epsilon_regularizer,
                                         constraint=self.epsilon_constraint,
                                         trainable=self.learnable_epsilon)

        self.built = True

    def call(self, x, **kwargs):
        nu2 = tf.reduce_mean(tf.square(x), axis=list(range(1, x.shape.ndims - 1)), keepdims=True)

        # Perform FRN.
        x = x * tf.math.rsqrt(nu2 + tf.abs(self.epsilon_l))

        return self.gamma * x + self.beta

    def get_config(self):
        config = {{
            'epsilon': self.epsilon,
            'learnable_epsilon': self.learnable_epsilon,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'epsilon_regularizer': tf.keras.regularizers.serialize(self.epsilon_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint),
            'epsilon_constraint': tf.keras.constraints.serialize(self.epsilon_constraint),
        }}
        base_config = super(FRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class TLU(tf.keras.layers.Layer):

    def __init__(self,
                 axis=-1,
                 tau_initializer='zeros',
                 tau_regularizer=None,
                 tau_constraint=None,
                 **kwargs):
        super(TLU, self).__init__(**kwargs)
        self.axis = axis
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau_regularizer = tf.keras.regularizers.get(tau_regularizer)
        self.tau_constraint = tf.keras.constraints.get(tau_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape),
                                                    axes={{self.axis: dim}})
        shape = (dim,)

        self.tau = self.add_weight(shape=shape,
                                   name='tau',
                                   initializer=self.tau_initializer,
                                   regularizer=self.tau_regularizer,
                                   constraint=self.tau_constraint)

        self.built = True

    def call(self, x, **kwargs):
        return tf.maximum(x, self.tau)

    def get_config(self):
        config = {{
            'tau_initializer': tf.keras.initializers.serialize(self.tau_initializer),
            'tau_regularizer': tf.keras.regularizers.serialize(self.tau_regularizer),
            'tau_constraint': tf.keras.constraints.serialize(self.tau_constraint)
        }}
        base_config = super(TLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


@wraps(tf.keras.layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {{'kernel_regularizer': tf.keras.regularizers.l2(5e-4)}}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def ctc_lambda_func(args):
    x, labels, input_len, label_len = args
    return K.ctc_batch_cost(labels, x, input_len, label_len)


class wBiFPNAdd(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({{
            'epsilon': self.epsilon
        }})
        return config


class PriorProbability(tf.keras.initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {{
            'probability': self.probability
        }}

    def __call__(self, shape, dtype=None):
        result = np.ones(shape) * -math.log((1 - self.probability) / self.probability)
        return result


class BoxNet(object):
    def __init__(self, width, depth, num_anchors=9, name='box_net', **kwargs):
        self.name = name
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        options = {{
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
            'depthwise_initializer': tf.keras.initializers.VarianceScaling(),
            'pointwise_initializer': tf.keras.initializers.VarianceScaling(),
        }}

        self.convs = [tf.keras.layers.SeparableConv2D(filters=width, **options) for i in range(depth)]
        self.head = tf.keras.layers.SeparableConv2D(filters=num_anchors * 4, **options)

        self.bns = [
            [tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3, name=f'{{self.name}}/box-{{i}}-bn-{{j}}') for j
             in range(3, 8)] for i in range(depth)]

        self.relu = tf.keras.layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = tf.keras.layers.Reshape((-1, 4))

    def call(self, inputs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        return outputs


class ClassNet(object):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, name='class_net', **kwargs):
        self.name = name
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        options = {{
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'depthwise_initializer': tf.keras.initializers.VarianceScaling(),
            'pointwise_initializer': tf.keras.initializers.VarianceScaling(),
        }}

        self.convs = [
            tf.keras.layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{{self.name}}/class-{{i}}',
                                            **options)
            for i in range(depth)]
        self.head = tf.keras.layers.SeparableConv2D(filters=num_classes * num_anchors,
                                                    bias_initializer=PriorProbability(probability=0.01),
                                                    name=f'{{self.name}}/class-predict', **options)

        self.bns = [
            [tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3, name=f'{{self.name}}/class-{{i}}-bn-{{j}}') for j
             in range(3, 8)]
            for i in range(depth)]

        self.relu = tf.keras.layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = tf.keras.layers.Reshape((-1, num_classes))
        self.activation = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        return outputs


class AnchorParameters(object):
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array([0.5, 1, 2], tf.keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], tf.keras.backend.floatx()),
)


class lambdalayer(object):
    @staticmethod
    def exists(val):
        return val is not None

    @staticmethod
    def default(val, d):
        return val if lambdalayer.exists(val) else d

    @staticmethod
    def calc_rel_pos(n):
        pos = tf.stack(tf.meshgrid(tf.range(n), tf.range(n), indexing='ij'))
        pos = Rearrange('n i j -> (i j) n')(pos)  # [n*n, 2] pos[n] = (i, j)
        rel_pos = pos[None, :] - pos[:, None]  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
        rel_pos += n - 1  # shift value range from [-n+1, n-1] to [0, 2n-2]
        return rel_pos


class LambdaLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            dim_k,
            n=None,
            r=None,
            heads=4,
            dim_out=None,
            dim_u=1):
        super(LambdaLayer, self).__init__()

        self.out_dim = dim_out
        self.u = dim_u  # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        self.dim_v = dim_out // heads
        self.dim_k = dim_k
        self.heads = heads

        self.to_q = tf.keras.layers.Conv2D(self.dim_k * heads, 1, use_bias=False)
        self.to_k = tf.keras.layers.Conv2D(self.dim_k * dim_u, 1, use_bias=False)
        self.to_v = tf.keras.layers.Conv2D(self.dim_v * dim_u, 1, use_bias=False)

        self.norm_q = tf.keras.layers.BatchNormalization()
        self.norm_v = tf.keras.layers.BatchNormalization()

        self.local_contexts = lambdalayer.exists(r)
        if lambdalayer.exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = tf.keras.layers.Conv3D(dim_k, (1, r, r), padding='same')
        else:
            assert lambdalayer.exists(n), 'You must specify the window length (n = h = w)'
            rel_length = 2 * n - 1
            self.rel_pos_emb = self.add_weight(name='pos_emb',
                                               shape=(rel_length, rel_length, dim_k, dim_u),
                                               initializer=tf.keras.initializers.random_normal,
                                               trainable=True)
            self.rel_pos = lambdalayer.calc_rel_pos(n)

    def call(self, x, **kwargs):
        b, hh, ww, c, u, h = *x.get_shape().as_list(), self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = Rearrange('b hh ww (h k) -> b h k (hh ww)', h=h)(q)
        k = Rearrange('b hh ww (u k) -> b u k (hh ww)', u=u)(k)
        v = Rearrange('b hh ww (u v) -> b u v (hh ww)', u=u)(v)

        k = tf.nn.softmax(k)

        Lc = tf.einsum('b u k m, b u v m -> b k v', k, v)
        Yc = tf.einsum('b h k n, b k v -> b n h v', q, Lc)

        if self.local_contexts:
            v = Rearrange('b u v (hh ww) -> b v hh ww u', hh=hh, ww=ww)(v)
            Lp = self.pos_conv(v)
            Lp = Rearrange('b v h w k -> b v k (h w)')(Lp)
            Yp = tf.einsum('b h k n, b v k n -> b n h v', q, Lp)
        else:
            rel_pos_emb = tf.gather_nd(self.rel_pos_emb, self.rel_pos)
            Lp = tf.einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = tf.einsum('b h k n, b n k v -> b n h v', q, Lp)

        Y = Yc + Yp
        out = Rearrange('b (hh ww) h v -> b hh ww (h v)', hh=hh, ww=ww)(Y)
        return out

    def compute_output_shape(self, input_shape):
        return (*input_shape[:2], self.out_dim)

    def get_config(self):
        config = {{'output_dim': (*self.input_shape[:2], self.out_dim)}}
        base_config = super(LambdaLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DyReLU(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = tf.keras.layers.Dense(
            channels // reduction,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform"))
        self.relu = tf.nn.relu
        self.fc2 = tf.keras.layers.Dense(
            2 * k * channels,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform"))
        self.sigmoid = tf.math.sigmoid

        self.lambdas = tf.constant([1.] * k + [0.5] * k, dtype=tf.float32)
        self.init_v = tf.constant([1.] + [0.] * (2 * k - 1), dtype=tf.float32)

    def get_relu_coefs(self, x):
        theta = tf.reduce_mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = tf.reduce_mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)
        relu_coefs = tf.reshape(theta, [-1, self.channels, 2 * self.k]) * self.lambdas + self.init_v

        # BxCxHxW -> HxWxBxCx1
        x_perm = tf.expand_dims(tf.transpose(x, [2, 3, 0, 1]), axis=-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # HxWxBxCx2 -> BxCxHxW
        result = tf.transpose(tf.reduce_max(output, axis=-1), [2, 3, 0, 1])
        return result

    def get_config(self):
        config = super(DyReLU, self).get_config()
        return config


class DropBlock(tf.keras.layers.Layer):
    # drop機率、block size
    def __init__(self, drop_rate=0.2, block_size=3, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.rate = drop_rate
        self.block_size = block_size

    def call(self, inputs, training=None):
        b = tf.shape(inputs)[0]

        random_tensor = tf.random.uniform(shape=[b, self.m_h, self.m_w, self.c]) + self.bernoulli_rate
        binary_tensor = tf.floor(random_tensor)
        binary_tensor = tf.pad(binary_tensor, [[0, 0],
                                               [self.block_size // 2, self.block_size // 2],
                                               [self.block_size // 2, self.block_size // 2],
                                               [0, 0]])
        binary_tensor = tf.nn.max_pool(binary_tensor,
                                       [1, self.block_size, self.block_size, 1],
                                       [1, 1, 1, 1],
                                       'SAME')
        binary_tensor = 1 - binary_tensor
        inputs = tf.math.divide(inputs, (1 - self.rate)) * binary_tensor
        return inputs

    def get_config(self):
        config = super(DropBlock, self).get_config()
        return config

    def build(self, input_shape):
        self.b, self.h, self.w, self.c = input_shape.as_list()

        self.m_h = self.h - (self.block_size // 2) * 2
        self.m_w = self.w - (self.block_size // 2) * 2
        self.bernoulli_rate = (self.rate * self.h * self.w) / (self.m_h * self.m_w * self.block_size ** 2)


class GroupedConv2D(object):
    def __init__(self, filters, kernel_size, use_keras=True, **kwargs):
        self._groups = len(kernel_size)
        self._channel_axis = -1

        self._convs = []
        splits = self._split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(self._get_conv2d(splits[i], kernel_size[i], use_keras, **kwargs))

    def _get_conv2d(self, filters, kernel_size, use_keras, **kwargs):
        if use_keras:
            return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)
        else:
            return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        if tf.__version__ < "2.0.0":
            filters = inputs.shape[self._channel_axis].value
        else:
            filters = inputs.shape[self._channel_axis]
        splits = self._split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x


class SCConv(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, padding='same', dilation=1, groups=1):
        super(SCConv, self).__init__()
        self.filters = filters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x):
        identity = x
        k2 = tf.keras.layers.AveragePooling2D(pool_size=4, strides=4)(x)
        k2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.stride, padding=self.padding,
                                    dilation_rate=self.dilation, groups=self.groups, use_bias=False)(k2)
        k2 = FRN()(k2)
        k3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.stride, padding=self.padding,
                                    dilation_rate=self.dilation, groups=self.groups, use_bias=False)(x)
        k3 = FRN()(k3)
        k4 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.stride, padding=self.padding,
                                    dilation_rate=self.dilation, groups=self.groups, use_bias=False)
        out = tf.keras.layers.UpSampling2D(size=(4, 4))(k2)
        out = tf.keras.layers.add([identity, out])
        out = tf.nn.sigmoid(out)
        out = tf.keras.layers.multiply([k3, out])
        out = k4(out)
        out = FRN()(out)
        return out

    def get_config(self):
        config = super(SCConv, self).get_config()
        return config


class Transformer_mask(object):
    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, d_model):
        angle_rads = Transformer_mask.get_angles(np.arange(position)[:, np.newaxis],
                                                 np.arange(d_model)[np.newaxis, :],
                                                 d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    @staticmethod
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    @staticmethod
    def point_wise_feed_forward_network(d_model, dim_feedforward, rate=0.1, activation='relu'):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation=activation),  # (batch_size, seq_len, dim_feedforward)
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.d_model = d_model

        assert d_model % self.nhead == 0

        self.depth = d_model // self.nhead

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.nhead, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, nhead, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, nhead, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, nhead, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, nhead, seq_len_q, depth)
        # attention_weights.shape == (batch_size, nhead, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = Transformer_mask.scaled_dot_product_attention(
            q, k, v, mask=mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, nhead, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, rate=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, nhead)
        self.ffn = Transformer_mask.point_wise_feed_forward_network(d_model, dim_feedforward, rate,
                                                                    activation=activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dim_feedforward, rate=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        # self.ffn = point_wise_feed_forward_network(d_model, dim_feedforward,rate)
        self.ffn = Transformer_mask.point_wise_feed_forward_network(d_model, dim_feedforward, rate,
                                                                    activation=activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, rate=0.1, activation='relu'):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, nhead, dim_feedforward, rate=rate, activation=activation)
                           for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None):
        # print('Encoder',x.shape)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=mask)

        x = self.layernorm(x)
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, rate=0.1, activation='relu'):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dim_feedforward, rate=rate, activation=activation)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output,
             look_ahead_mask, padding_mask):
        # seq_len = tf.shape(x)[1]
        attention_weights = {{}}

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output,
                                                   look_ahead_mask, padding_mask)

        attention_weights['decoder_layer{{}}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{{}}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class GhostModule(tf.keras.layers.Layer):
    def __init__(self, out, ratio, convkernel, dwkernel):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        self.out = out
        self.conv_out_channel = math.ceil(self.out * 1.0 / ratio)
        self.conv = tf.keras.layers.Conv2D(int(self.conv_out_channel), (convkernel, convkernel), use_bias=False,
                                           strides=(1, 1), padding='same', activation=None)
        self.depthconv = tf.keras.layers.DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                                                         depth_multiplier=ratio - 1, activation=None)
        self.slice = tf.keras.layers.Lambda(self._return_slices,
                                            arguments={{'channel': int(self.out - self.conv_out_channel)}})
        self.concat = tf.keras.layers.Concatenate()

    @staticmethod
    def _return_slices(x, channel):
        return x[:, :, :, :channel]

    def call(self, inputs):
        x = self.conv(inputs)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = self.slice(dw)
        output = self.concat([x, dw])
        return output


class SEModule(tf.keras.layers.Layer):

    def __init__(self, filters, ratio):
        super(SEModule, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Lambda(self._reshape)
        self.conv1 = tf.keras.layers.Conv2D(int(filters / ratio), (1, 1), strides=(1, 1), padding='same',
                                            use_bias=False, activation=None)
        self.conv2 = tf.keras.layers.Conv2D(int(filters), (1, 1), strides=(1, 1), padding='same',
                                            use_bias=False, activation=None)
        self.relu = tf.keras.layers.Activation('relu')
        self.hard_sigmoid = tf.keras.layers.Activation('hard_sigmoid')

    @staticmethod
    def _reshape(x):
        return tf.keras.layers.Reshape((1, 1, int(x.shape[1])))(x)

    @staticmethod
    def _excite(x, excitation):
        return x * excitation

    def call(self, inputs):
        x = self.reshape(self.pooling(inputs))
        x = self.relu(self.conv1(x))
        excitation = self.hard_sigmoid(self.conv2(x))
        x = tf.keras.layers.Lambda(self._excite, arguments={{'excitation': excitation}})(inputs)
        return x


class GBNeck(tf.keras.layers.Layer):

    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(GBNeck, self).__init__()
        self.strides = strides
        self.use_se = use_se
        self.conv = tf.keras.layers.Conv2D(out, (1, 1), strides=(1, 1), padding='same',
                                           activation=None, use_bias=False)
        self.relu = tf.keras.layers.Activation('relu')
        self.depthconv1 = tf.keras.layers.DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio - 1,
                                                          activation=None, use_bias=False)
        self.depthconv2 = tf.keras.layers.DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio - 1,
                                                          activation=None, use_bias=False)
        for i in range(5):
            setattr(self, f"batchnorm{{i + 1}}", tf.keras.layers.BatchNormalization())
        self.ghost1 = GhostModule(exp, ratio, 1, 3)
        self.ghost2 = GhostModule(out, ratio, 1, 3)
        self.se = SEModule(exp, ratio)

    def call(self, inputs):
        x = self.batchnorm1(self.depthconv1(inputs))
        x = self.batchnorm2(self.conv(x))

        y = self.relu(self.batchnorm3(self.ghost1(inputs)))
        if self.strides > 1:
            y = self.relu(self.batchnorm4(self.depthconv2(y)))
        if self.use_se:
            y = self.se(y)
        y = self.batchnorm5(self.ghost2(y))
        return tf.keras.layers.add([x, y])

    def get_config(self):
        config = super(GBNeck, self).get_config()
        return config


class Yolo_Loss(object):

    @staticmethod
    def _smooth_labels(y_true, label_smoothing):
        num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
        label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    @staticmethod
    def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
        num_anchors = len(anchors)
        # [1, 1, 1, num_anchors, 2]
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        # 获得x，y的网格
        # (13, 13, 1, 2)
        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        # (batch_size,13,13,3,85)
        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # 将预测值调成真实值
        # box_xy对应框的中心点
        # box_wh对应框的宽和高
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        # 在计算loss的时候返回如下参数
        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    @staticmethod
    def box_ciou(b1, b2):
        # 求出预测框左上角右下角
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # 求出真实框左上角右下角
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 求真实框和预测框所有的iou
        intersect_mins = K.maximum(b1_mins, b2_mins)
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / K.maximum(union_area, K.epsilon())

        # 计算中心的差距
        center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
        # 找到包裹两个框的最小框的左上角和右下角
        enclose_mins = K.minimum(b1_mins, b2_mins)
        enclose_maxes = K.maximum(b1_maxes, b2_maxes)
        enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
        # 计算对角线距离
        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
        ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())

        v = 4 * K.square(
            tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0],
                                                                                                K.maximum(
                                                                                                    b2_wh[
                                                                                                        ..., 1],
                                                                                                    K.epsilon()))) / (
                    math.pi * math.pi)
        alpha = v / K.maximum((1.0 - iou + v), K.epsilon())
        ciou = ciou - alpha * v

        ciou = K.expand_dims(ciou, -1)
        return ciou

    @staticmethod
    def box_iou(b1, b2):
        # 13,13,3,1,4
        # 计算左上角的坐标和右下角的坐标
        b1 = K.expand_dims(b1, -2)
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        # 1,n,4
        # 计算左上角和右下角的坐标
        b2 = K.expand_dims(b2, 0)
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 计算重合面积
        intersect_mins = K.maximum(b1_mins, b2_mins)
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area)

        return iou

    @staticmethod
    def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False):
        # 一共有三层
        num_layers = len(anchors) // 3

        # 将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
        # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
        # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,255),(m,26,26,255),(m,52,52,255)。
        y_true = args[num_layers:]
        yolo_outputs = args[:num_layers]

        # 先验框
        # 678为142,110,  192,243,  459,401
        # 345为36,75,  76,55,  72,146
        # 012为12,16,  19,36,  40,28
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        # 得到input_shpae为608,608
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

        loss = 0

        # 取出每一张图片
        # m的值就是batch_size
        m = K.shape(yolo_outputs[0])[0]
        mf = K.cast(m, K.dtype(yolo_outputs[0]))

        # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
        # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,255),(m,26,26,255),(m,52,52,255)。
        for l in range(num_layers):
            # 以第一个特征层(m,13,13,3,85)为例子
            # 取出该特征层中存在目标的点的位置。(m,13,13,3,1)
            object_mask = y_true[l][..., 4:5]
            # 取出其对应的种类(m,13,13,3,80)
            true_class_probs = y_true[l][..., 5:]
            if label_smoothing:
                true_class_probs = Yolo_Loss._smooth_labels(true_class_probs, label_smoothing)

            # 将yolo_outputs的特征层输出进行处理
            # grid为网格结构(13,13,1,2)，raw_pred为尚未处理的预测结果(m,13,13,3,85)
            # 还有解码后的xy，wh，(m,13,13,3,2)
            grid, raw_pred, pred_xy, pred_wh = Yolo_Loss.yolo_head(yolo_outputs[l],
                                                                   anchors[anchor_mask[l]], num_classes, input_shape,
                                                                   calc_loss=True)

            # 这个是解码后的预测的box的位置
            # (m,13,13,3,4)
            pred_box = K.concatenate([pred_xy, pred_wh])

            # 找到负样本群组，第一步是创建一个数组，[]
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, 'bool')

            # 对每一张图片计算ignore_mask
            def loop_body(b, ignore_mask):
                # 取出第b副图内，真实存在的所有的box的参数
                # n,4
                true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                # 计算预测结果与真实情况的iou
                # pred_box为13,13,3,4
                # 计算的结果是每个pred_box和其它所有真实框的iou
                # 13,13,3,n
                iou = Yolo_Loss.box_iou(pred_box[b], true_box)

                # 13,13,3
                best_iou = K.max(iou, axis=-1)

                # 如果某些预测框和真实框的重合程度大于0.5，则忽略。
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
                return b + 1, ignore_mask

            # 遍历所有的图片
            _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

            # 将每幅图的内容压缩，进行处理
            ignore_mask = ignore_mask.stack()
            # (m,13,13,3,1)
            ignore_mask = K.expand_dims(ignore_mask, -1)

            box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

            # Calculate ciou loss as location loss
            raw_true_box = y_true[l][..., 0:4]
            ciou = Yolo_Loss.box_ciou(pred_box, raw_true_box)
            ciou_loss = object_mask * box_loss_scale * (1 - ciou)
            ciou_loss = K.sum(ciou_loss) / mf
            location_loss = ciou_loss

            # 如果该位置本来有框，那么计算1与置信度的交叉熵
            # 如果该位置本来没有框，而且满足best_iou<ignore_thresh，则被认定为负样本
            # best_iou<ignore_thresh用于限制负样本数量
            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (
                    1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                             from_logits=True) * ignore_mask

            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

            confidence_loss = K.sum(confidence_loss) / mf
            class_loss = K.sum(class_loss) / mf
            location_loss = K.sum(location_loss) / mf
            confidence_loss = K.sum(confidence_loss) / mf
            class_loss = K.sum(class_loss) / mf
            loss += location_loss + confidence_loss + class_loss
        loss = K.expand_dims(loss, axis=-1)
        return loss


class YOLO_anchors(object):
    @staticmethod
    def get_anchors():
        if not os.path.exists(ANCHORS_PATH):
            SIZE = IMAGE_HEIGHT
            if MODE == 'YOLO':
                anchors_num = 9
            elif MODE == 'YOLO_TINY':
                anchors_num = 6
            else:
                raise ValueError('anchors_num error')
            data = YOLO_anchors.load_data()
            # out = YOLO_anchors.kmeans(data, anchors_num)
            out = AnchorGenerator(anchors_num).generate_anchor(data)
            out = out[np.argsort(out[:, 0])]
            data = out * SIZE
            row = np.shape(data)[0]
            anchors = []
            for i in range(row):
                x_y = [int(data[i][0]), int(data[i][1])]
                anchors.append(x_y)
            save_dict = {{'anchors': anchors}}
            with open(ANCHORS_PATH, 'w', encoding='utf-8') as f:
                f.write(json.dumps(save_dict, ensure_ascii=False))
            return np.array(anchors, dtype=np.float).reshape(-1, 2)
        else:
            with open(ANCHORS_PATH, 'r', encoding='utf-8') as f:
                anchors = json.loads(f.read()).get('anchors')
            return np.array(anchors, dtype=np.float).reshape(-1, 2)

    @staticmethod
    def load_data():
        data = []
        # 对于每一个xml都寻找box
        try:
            label_list = glob.glob(f'{{LABEL_PATH}}\*\*.xml')
        except:
            label_list = glob.glob(f'{{LABEL_PATH}}\*.xml')
        for xml_file in label_list:
            tree = ET.parse(xml_file)
            height = int(tree.findtext('./size/height'))
            width = int(tree.findtext('./size/width'))
            # 对于每一个目标都获得它的宽高
            for obj in tree.iter('object'):
                xmin = np.float64(int(float(obj.findtext('bndbox/xmin'))) / width)
                ymin = np.float64(int(float(obj.findtext('bndbox/ymin'))) / height)
                xmax = np.float64(int(float(obj.findtext('bndbox/xmax'))) / width)
                ymax = np.float64(int(float(obj.findtext('bndbox/ymax'))) / height)
                # 得到宽高
                data.append([xmax - xmin, ymax - ymin])
        return np.array(data)

    @staticmethod
    def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
        num_anchors = len(anchors)
        # [1, 1, 1, num_anchors, 2]
        feats = tf.convert_to_tensor(feats)
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        # 获得x，y的网格
        # (13, 13, 1, 2)
        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        # (batch_size,13,13,3,85)
        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # 将预测值调成真实值
        # box_xy对应框的中心点
        # box_wh对应框的宽和高
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        # 在计算loss的时候返回如下参数
        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    @staticmethod
    def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        boxes *= K.concatenate([image_shape, image_shape])
        return boxes

    @staticmethod
    def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
        # 将预测值调成真实值
        # box_xy对应框的中心点
        # box_wh对应框的宽和高
        # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,80
        box_xy, box_wh, box_confidence, box_class_probs = YOLO_anchors.yolo_head(feats, anchors, num_classes,
                                                                                 input_shape)
        # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
        boxes = YOLO_anchors.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        # 获得得分和box
        boxes = K.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = K.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores

    @staticmethod
    def yolo_eval(yolo_outputs,
                  anchors,
                  num_classes,
                  image_shape,
                  max_boxes=MAX_BOXES,
                  score_threshold=.6,
                  iou_threshold=.5,
                  eager=False):
        if eager:
            image_shape = K.reshape(yolo_outputs[-1], [-1])
            num_layers = len(yolo_outputs) - 1
        else:
            # 获得特征层的数量
            num_layers = len(yolo_outputs)
        # 特征层1对应的anchor是678
        # 特征层2对应的anchor是345
        # 特征层3对应的anchor是012
        anchor_mask = [[3, 4, 5], [1, 2, 3]]

        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []
        # 对每个特征层进行处理
        for l in range(num_layers):
            _boxes, _box_scores = YOLO_anchors.yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]],
                                                                     num_classes,
                                                                     input_shape,
                                                                     image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 将每个特征层的结果进行堆叠
        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            # 取出所有box_scores >= score_threshold的框，和成绩
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

            # 非极大抑制，去掉box重合程度高的那一些
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

            # 获取非极大抑制后的结果
            # 下列三个分别是
            # 框的位置，得分与种类
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_


class Efficientdet_Loss(object):
    @staticmethod
    def focal(alpha=0.25, gamma=2.0):
        def _focal(y_true, y_pred):
            # y_true [batch_size, num_anchor, num_classes+1]
            # y_pred [batch_size, num_anchor, num_classes]
            labels = y_true[:, :, :-1]
            anchor_state = y_true[:, :, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
            classification = y_pred

            # 找出存在目标的先验框
            indices_for_object = tf.where(tf.keras.backend.equal(anchor_state, 1))
            labels_for_object = tf.gather_nd(labels, indices_for_object)
            classification_for_object = tf.gather_nd(classification, indices_for_object)

            # 计算每一个先验框应该有的权重
            alpha_factor_for_object = tf.keras.backend.ones_like(labels_for_object) * alpha
            alpha_factor_for_object = tf.where(tf.keras.backend.equal(labels_for_object, 1), alpha_factor_for_object,
                                               1 - alpha_factor_for_object)
            focal_weight_for_object = tf.where(tf.keras.backend.equal(labels_for_object, 1),
                                               1 - classification_for_object, classification_for_object)
            focal_weight_for_object = alpha_factor_for_object * focal_weight_for_object ** gamma

            # 将权重乘上所求得的交叉熵
            cls_loss_for_object = focal_weight_for_object * tf.keras.backend.binary_crossentropy(labels_for_object,
                                                                                                 classification_for_object)

            # 找出实际上为背景的先验框
            indices_for_back = tf.where(tf.keras.backend.equal(anchor_state, 0))
            labels_for_back = tf.gather_nd(labels, indices_for_back)
            classification_for_back = tf.gather_nd(classification, indices_for_back)

            # 计算每一个先验框应该有的权重
            alpha_factor_for_back = tf.keras.backend.ones_like(labels_for_back) * (1 - alpha)
            focal_weight_for_back = classification_for_back
            focal_weight_for_back = alpha_factor_for_back * focal_weight_for_back ** gamma

            # 将权重乘上所求得的交叉熵
            cls_loss_for_back = focal_weight_for_back * tf.keras.backend.binary_crossentropy(labels_for_back,
                                                                                             classification_for_back)

            # 标准化，实际上是正样本的数量
            normalizer = tf.where(tf.keras.backend.equal(anchor_state, 1))
            normalizer = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
            normalizer = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

            # 将所获得的loss除上正样本的数量
            cls_loss_for_object = tf.keras.backend.sum(cls_loss_for_object)
            cls_loss_for_back = tf.keras.backend.sum(cls_loss_for_back)

            # 总的loss
            loss = (cls_loss_for_object + cls_loss_for_back) / normalizer

            return loss

        return _focal

    @staticmethod
    def smooth_l1(sigma=3.0):
        sigma_squared = sigma ** 2

        def _smooth_l1(y_true, y_pred):
            regression = y_pred
            regression_target = y_true[:, :, :-1]
            anchor_state = y_true[:, :, -1]

            indices = tf.where(tf.keras.backend.equal(anchor_state, 1))
            regression = tf.gather_nd(regression, indices)
            regression_target = tf.gather_nd(regression_target, indices)

            # compute smooth L1 loss
            # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
            #        |x| - 0.5 / sigma / sigma    otherwise
            regression_diff = regression - regression_target
            regression_diff = tf.keras.backend.abs(regression_diff)
            regression_loss = tf.where(
                tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
                0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
                regression_diff - 0.5 / sigma_squared
            )

            # compute the normalizer: the number of positive anchors
            normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
            normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
            return tf.keras.backend.sum(regression_loss) / normalizer / 4

        return _smooth_l1


class Efficientdet_anchors(object):
    @staticmethod
    def get_swish():
        def swish(x):
            return x * tf.keras.backend.sigmoid(x)

        return swish

    @staticmethod
    def get_dropout():
        class FixedDropout(tf.keras.layers.Dropout):
            def _get_noise_shape(self, inputs):
                if self.noise_shape is None:
                    return self.noise_shape

                symbolic_shape = tf.keras.backend.shape(inputs)
                noise_shape = [symbolic_shape[axis] if shape is None else shape
                               for axis, shape in enumerate(self.noise_shape)]
                return tuple(noise_shape)

        return FixedDropout

    @staticmethod
    def round_filters(filters, width_coefficient, depth_divisor):
        filters *= width_coefficient
        new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
        new_filters = max(depth_divisor, new_filters)
        if new_filters < 0.9 * filters:
            new_filters += depth_divisor
        return int(new_filters)

    @staticmethod
    def round_repeats(repeats, depth_coefficient):
        return int(math.ceil(depth_coefficient * repeats))

    @staticmethod
    def mb_conv_block(inputs, block_args, activation, drop_rate=None):
        has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        bn_axis = 3

        Dropout = Efficientdet_anchors.get_dropout()

        filters = block_args.input_filters * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            x = tf.keras.layers.Conv2D(filters, 1,
                                       padding='same',
                                       use_bias=False,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)
            x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
            x = tf.keras.layers.Activation(activation)(x)
        else:
            x = inputs

        x = tf.keras.layers.DepthwiseConv2D(block_args.kernel_size,
                                            strides=block_args.strides,
                                            padding='same',
                                            use_bias=False,
                                            depthwise_initializer=CONV_KERNEL_INITIALIZER)(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = tf.keras.layers.Activation(activation)(x)

        if has_se:
            num_reduced_filters = max(1, int(
                block_args.input_filters * block_args.se_ratio
            ))
            se_tensor = tf.keras.layers.GlobalAveragePooling2D()(x)

            target_shape = (1, 1, filters) if tf.keras.backend.image_data_format() == 'channels_last' else (
                filters, 1, 1)
            se_tensor = tf.keras.layers.Reshape(target_shape)(se_tensor)
            se_tensor = tf.keras.layers.Conv2D(num_reduced_filters, 1,
                                               activation=activation,
                                               padding='same',
                                               use_bias=True,
                                               kernel_initializer=CONV_KERNEL_INITIALIZER)(se_tensor)
            se_tensor = tf.keras.layers.Conv2D(filters, 1,
                                               activation='sigmoid',
                                               padding='same',
                                               use_bias=True,
                                               kernel_initializer=CONV_KERNEL_INITIALIZER)(se_tensor)
            if tf.keras.backend.backend() == 'theano':
                pattern = ([True, True, True, False] if tf.keras.backend.image_data_format() == 'channels_last'
                           else [True, False, True, True])
                se_tensor = tf.keras.layers.Lambda(
                    lambda x: tf.keras.backend.pattern_broadcast(x, pattern))(se_tensor)
            x = tf.keras.layers.multiply([x, se_tensor])

        # Output phase
        x = tf.keras.layers.Conv2D(block_args.output_filters, 1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
        if block_args.id_skip and all(
                s == 1 for s in block_args.strides
        ) and block_args.input_filters == block_args.output_filters:
            if drop_rate and (drop_rate > 0):
                x = Dropout(drop_rate,
                            noise_shape=(None, 1, 1, 1))(x)
            x = tf.keras.layers.add([x, inputs])

        return x

    @staticmethod
    def iou(b1, b2):
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

    @staticmethod
    def generate_anchors(base_size=16, ratios=None, scales=None):
        if ratios is None:
            ratios = AnchorParameters.default.ratios

        if scales is None:
            scales = AnchorParameters.default.scales

        num_anchors = len(ratios) * len(scales)

        anchors = np.zeros((num_anchors, 4))

        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        areas = anchors[:, 2] * anchors[:, 3]

        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    @staticmethod
    def shift(shape, stride, anchors):
        shift_x = (np.arange(0, shape[1], dtype=tf.keras.backend.floatx()) + 0.5) * stride
        shift_y = (np.arange(0, shape[0], dtype=tf.keras.backend.floatx()) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift_x = np.reshape(shift_x, [-1])
        shift_y = np.reshape(shift_y, [-1])

        shifts = np.stack([
            shift_x,
            shift_y,
            shift_x,
            shift_y
        ], axis=0)

        shifts = np.transpose(shifts)
        number_of_anchors = np.shape(anchors)[0]

        k = np.shape(shifts)[0]

        shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]),
                                                                                    tf.keras.backend.floatx())
        shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])

        return shifted_anchors

    @staticmethod
    def get_anchors(image_size):
        border = image_size
        features = [image_size / 8, image_size / 16, image_size / 32, image_size / 64, image_size / 128]
        shapes = []
        for feature in features:
            shapes.append(feature)
        all_anchors = []
        for i in range(5):
            anchors = Efficientdet_anchors.generate_anchors(AnchorParameters.default.sizes[i])
            shifted_anchors = Efficientdet_anchors.shift([shapes[i], shapes[i]], AnchorParameters.default.strides[i],
                                                         anchors)
            all_anchors.append(shifted_anchors)

        all_anchors = np.concatenate(all_anchors, axis=0)
        all_anchors = all_anchors / border
        return all_anchors

    @staticmethod
    def SeparableConvBlock(num_channels, kernel_size, strides):
        f1 = tf.keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                             use_bias=True)
        f2 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

    @staticmethod
    def build_wBiFPN(features, num_channels, id):
        if id == 0:
            _, _, C3, C4, C5 = features
            # 第一次BIFPN需要 下采样 与 降通道 获得 p3_in p4_in p5_in p6_in p7_in
            # -----------------------------下采样 与 降通道----------------------------#
            P3_in = C3
            P3_in = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P3_in)
            P3_in = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P3_in)

            P4_in = C4
            P4_in_1 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P4_in)
            P4_in_1 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P4_in_1)
            P4_in_2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P4_in)
            P4_in_2 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P4_in_2)

            P5_in = C5
            P5_in_1 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P5_in)
            P5_in_1 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P5_in_1)
            P5_in_2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P5_in)
            P5_in_2 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P5_in_2)

            P6_in = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(C5)
            P6_in = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P6_in)
            P6_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_in)

            P7_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_in)
            # -------------------------------------------------------------------------#

            # --------------------------构建BIFPN的上下采样循环-------------------------#
            P7_U = tf.keras.layers.UpSampling2D()(P7_in)
            P6_td = wBiFPNAdd()([P6_in, P7_U])
            P6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
            P6_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_td)

            P6_U = tf.keras.layers.UpSampling2D()(P6_td)
            P5_td = wBiFPNAdd()([P5_in_1, P6_U])
            P5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
            P5_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_td)

            P5_U = tf.keras.layers.UpSampling2D()(P5_td)
            P4_td = wBiFPNAdd()([P4_in_1, P5_U])
            P4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
            P4_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_td)

            P4_U = tf.keras.layers.UpSampling2D()(P4_td)
            P3_out = wBiFPNAdd()([P3_in, P4_U])
            P3_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
            P3_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P3_out)

            P3_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = wBiFPNAdd()([P4_in_2, P4_td, P3_D])
            P4_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
            P4_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P4_out)

            P4_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = wBiFPNAdd()([P5_in_2, P5_td, P4_D])
            P5_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
            P5_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P5_out)

            P5_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = wBiFPNAdd()([P6_in, P6_td, P5_D])
            P6_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
            P6_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P6_out)

            P6_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = wBiFPNAdd()([P7_in, P6_D])
            P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
            P7_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P7_out)

        else:
            P3_in, P4_in, P5_in, P6_in, P7_in = features
            P7_U = tf.keras.layers.UpSampling2D()(P7_in)
            P6_td = wBiFPNAdd()([P6_in, P7_U])
            P6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
            P6_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_td)

            P6_U = tf.keras.layers.UpSampling2D()(P6_td)
            P5_td = wBiFPNAdd()([P5_in, P6_U])
            P5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
            P5_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_td)

            P5_U = tf.keras.layers.UpSampling2D()(P5_td)
            P4_td = wBiFPNAdd()([P4_in, P5_U])
            P4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
            P4_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_td)

            P4_U = tf.keras.layers.UpSampling2D()(P4_td)
            P3_out = wBiFPNAdd()([P3_in, P4_U])
            P3_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
            P3_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P3_out)

            P3_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = wBiFPNAdd()([P4_in, P4_td, P3_D])
            P4_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
            P4_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P4_out)

            P4_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = wBiFPNAdd()([P5_in, P5_td, P4_D])
            P5_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
            P5_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P5_out)

            P5_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = wBiFPNAdd()([P6_in, P6_td, P5_D])
            P6_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
            P6_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P6_out)

            P6_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = wBiFPNAdd()([P7_in, P6_D])
            P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
            P7_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P7_out)

        return [P3_out, P4_out, P5_out, P6_out, P7_out]

    @staticmethod
    def build_BiFPN(features, num_channels, id):
        if id == 0:
            # 第一次BIFPN需要 下采样 与 降通道 获得 p3_in p4_in p5_in p6_in p7_in
            # -----------------------------下采样 与 降通道----------------------------#
            _, _, C3, C4, C5 = features
            P3_in = C3
            P3_in = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P3_in)
            P3_in = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P3_in)

            P4_in = C4
            P4_in_1 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P4_in)
            P4_in_1 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P4_in_1)
            P4_in_2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P4_in)
            P4_in_2 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P4_in_2)

            P5_in = C5
            P5_in_1 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P5_in)
            P5_in_1 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P5_in_1)
            P5_in_2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(P5_in)
            P5_in_2 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P5_in_2)

            P6_in = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(C5)
            P6_in = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(P6_in)
            P6_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_in)

            P7_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_in)
            # -------------------------------------------------------------------------#

            # --------------------------构建BIFPN的上下采样循环-------------------------#
            P7_U = tf.keras.layers.UpSampling2D()(P7_in)
            P6_td = tf.keras.layers.Add()([P6_in, P7_U])
            P6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
            P6_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_td)

            P6_U = tf.keras.layers.UpSampling2D()(P6_td)
            P5_td = tf.keras.layers.Add()([P5_in_1, P6_U])
            P5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
            P5_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_td)

            P5_U = tf.keras.layers.UpSampling2D()(P5_td)
            P4_td = tf.keras.layers.Add()([P4_in_1, P5_U])
            P4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
            P4_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_td)

            P4_U = tf.keras.layers.UpSampling2D()(P4_td)
            P3_out = tf.keras.layers.Add()([P3_in, P4_U])
            P3_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
            P3_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P3_out)

            P3_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = tf.keras.layers.Add()([P4_in_2, P4_td, P3_D])
            P4_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
            P4_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P4_out)

            P4_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = tf.keras.layers.Add()([P5_in_2, P5_td, P4_D])
            P5_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
            P5_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P5_out)

            P5_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = tf.keras.layers.Add()([P6_in, P6_td, P5_D])
            P6_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
            P6_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P6_out)

            P6_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = tf.keras.layers.Add()([P7_in, P6_D])
            P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
            P7_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P7_out)

        else:
            P3_in, P4_in, P5_in, P6_in, P7_in = features
            P7_U = tf.keras.layers.UpSampling2D()(P7_in)
            P6_td = tf.keras.layers.Add()([P6_in, P7_U])
            P6_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
            P6_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P6_td)

            P6_U = tf.keras.layers.UpSampling2D()(P6_td)
            P5_td = tf.keras.layers.Add()([P5_in, P6_U])
            P5_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
            P5_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P5_td)

            P5_U = tf.keras.layers.UpSampling2D()(P5_td)
            P4_td = tf.keras.layers.Add()([P4_in, P5_U])
            P4_td = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
            P4_td = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(P4_td)

            P4_U = tf.keras.layers.UpSampling2D()(P4_td)
            P3_out = tf.keras.layers.Add()([P3_in, P4_U])
            P3_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
            P3_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P3_out)

            P3_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
            P4_out = tf.keras.layers.Add()([P4_in, P4_td, P3_D])
            P4_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
            P4_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P4_out)

            P4_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
            P5_out = tf.keras.layers.Add()([P5_in, P5_td, P4_D])
            P5_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
            P5_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P5_out)

            P5_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
            P6_out = tf.keras.layers.Add()([P6_in, P6_td, P5_D])
            P6_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
            P6_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P6_out)

            P6_D = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
            P7_out = tf.keras.layers.Add()([P7_in, P6_D])
            P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
            P7_out = Efficientdet_anchors.SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1)(
                P7_out)
        return [P3_out, P4_out, P5_out, P6_out, P7_out]


class AnchorGenerator:
    def __init__(self, cluster_number):
        self.cluster_number = cluster_number

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))
        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))
        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)
        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def generator(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        last_nearest = np.zeros((box_number,))
        clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest
        return clusters

    def generate_anchor(self, boxes):
        result = self.generator(boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        logger.debug("Accuracy: {{:.2f}}%".format(self.avg_iou(boxes, result) * 100))
        return result


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=-1,
                 reduction=tf.keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index
        )
        return tf.reduce_mean(loss)


class WordAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='word_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32,
                                     initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32,
                                     initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        b = tf.shape(y_true)[0]
        max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        y_true = tf.sparse.reset_shape(y_true, [b, max_width])
        y_pred = tf.sparse.reset_shape(decoded[0], [b, max_width])
        y_true = tf.sparse.to_dense(y_true, default_value=-1)
        y_pred = tf.sparse.to_dense(y_pred, default_value=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        values = tf.math.reduce_any(tf.math.not_equal(y_true, y_pred), axis=1)
        values = tf.cast(values, tf.int32)
        values = tf.reduce_sum(values)
        self.total.assign_add(b)
        self.count.assign_add(b - values)

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)


class Settings(object):
    @staticmethod
    def settings():
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            n_class = len(json.loads(f.read()))
        return n_class + 1

    @staticmethod
    def settings_num_classes():
        with open(NUMBER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            n_class = len(json.loads(f.read()))
        return n_class


class Yolov3_block(object):
    @staticmethod
    def darknetconv(x, filters, size, strides=1, batch_norm=True):
        if strides == 1:
            padding = 'same'
        else:
            x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
            padding = 'valid'
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding,
                                   use_bias=not batch_norm, kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    @staticmethod
    def darknetresidual(x, filters):
        prev = x
        x = Yolov3_block.darknetconv(x, filters // 2, 1)
        x = Yolov3_block.darknetconv(x, filters, 3)
        x = tf.keras.layers.Add()([prev, x])
        return x

    @staticmethod
    def darknetblock(x, filters, blocks):
        x = Yolov3_block.darknetconv(x, filters, 3, strides=2)
        for _ in range(blocks):
            x = Yolov3_block.darknetresidual(x, filters)
        return x

    @staticmethod
    def darknet(inputs):
        x = inputs
        x = Yolov3_block.darknetconv(x, 32, 3)
        x = Yolov3_block.darknetconv(x, 64, 1)
        x = Yolov3_block.darknetconv(x, 128, 2)
        x = x_36 = Yolov3_block.darknetblock(x, 256, 8)
        x = x_61 = Yolov3_block.darknetblock(x, 512, 8)
        x = Yolov3_block.darknetblock(x, 1024, 4)
        return x_36, x_61, x

    @staticmethod
    def darknettiny(inputs):
        x = inputs
        x = Yolov3_block.darknetconv(x, 16, 3)
        x = tf.keras.layers.MaxPooling2D(2, 2, 'same')(x)
        x = Yolov3_block.darknetconv(x, 32, 3)
        x = tf.keras.layers.MaxPooling2D(2, 2, 'same')(x)
        x = Yolov3_block.darknetconv(x, 64, 3)
        x = tf.keras.layers.MaxPooling2D(2, 2, 'same')(x)
        x = Yolov3_block.darknetconv(x, 128, 3)
        x = tf.keras.layers.MaxPooling2D(2, 2, 'same')(x)
        x = x_8 = Yolov3_block.darknetconv(x, 256, 3)
        x = tf.keras.layers.MaxPooling2D(2, 2, 'same')(x)
        x = Yolov3_block.darknetconv(x, 512, 3)
        x = tf.keras.layers.MaxPooling2D(2, 1, 'same')(x)
        x = Yolov3_block.darknetconv(x, 1024, 3)
        return x_8, x

    @staticmethod
    def yoloconv(filters):
        def yolo_conv(x_in):
            if isinstance(x_in, tuple):
                inputs = tf.keras.layers.Input(x_in[0].shape[1:]), tf.keras.layers.Input(x_in[1].shape[1:])
                x, x_skip = inputs
                x = Yolov3_block.darknetconv(x, filters, 1)
                x = tf.keras.layers.UpSampling2D(2)(x)
                x = tf.keras.layers.Concatenate()([x, x_skip])
            else:
                x = inputs = tf.keras.layers.Input(x_in.shape[1:])

            x = Yolov3_block.darknetconv(x, filters, 1)
            x = Yolov3_block.darknetconv(x, filters * 2, 3)
            x = Yolov3_block.darknetconv(x, filters, 1)
            x = Yolov3_block.darknetconv(x, filters * 2, 3)
            x = Yolov3_block.darknetconv(x, filters, 1)
            return tf.keras.Model(inputs, x)(x_in)

        return yolo_conv

    @staticmethod
    def yoloconvtiny(filters):
        def yolo_conv(x_in):
            if isinstance(x_in, tuple):
                inputs = tf.keras.layers.Input(x_in[0].shape[1:]), tf.keras.layers.Input(x_in[1].shape[1:])
                x, x_skip = inputs
                x = Yolov3_block.darknetconv(x, filters, 1)
                x = tf.keras.layers.UpSampling2D(2)(x)
                x = tf.keras.layers.Concatenate()([x, x_skip])
            else:
                x = inputs = tf.keras.layers.Input(x_in.shape[1:])
                x = Yolov3_block.darknetconv(x, filters, 1)
            return tf.keras.Model(inputs, x)(x_in)

        return yolo_conv

    @staticmethod
    def yolo_output(filters, anchors, classes):
        def yolo_output(x_in):
            x = inputs = tf.keras.layers.Input(x_in.shape[1:])
            x = Yolov3_block.darknetconv(x, filters * 2, 3)
            x = Yolov3_block.darknetconv(x, anchors * (classes + 5), 1, batch_norm=False)
            x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                                                anchors, classes + 5)))(x)
            return tf.keras.Model(inputs, x)(x_in)

        return yolo_output


class Yolov3_losses(object):
    @staticmethod
    def yolo_boxes(pred, anchors, classes):
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, classes), axis=-1)
        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss
        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors
        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
        return bbox, objectness, class_probs, pred_box

    @staticmethod
    def yolo_nms(outputs, anchors, masks, classes):
        b, c, t = [], [], []
        for o in outputs:
            b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
            c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
            t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
        bbox = tf.concat(b, axis=1)
        confidence = tf.concat(c, axis=1)
        class_probs = tf.concat(t, axis=1)
        scores = confidence * class_probs
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(
                scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=0.5,
            score_threshold=0.5
        )
        return boxes, scores, classes, valid_detections

    @staticmethod
    def broadcast_iou(box_1, box_2):
        box_1 = tf.expand_dims(box_1, -2)
        box_2 = tf.expand_dims(box_2, 0)
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)
        int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                           tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                           tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
        return int_area / (box_1_area + box_2_area - int_area)

    @staticmethod
    def yololoss(anchors, classes=80, ignore_thresh=0.5):
        def yolo_loss(y_true, y_pred):
            pred_box, pred_obj, pred_class, pred_xywh = Yolov3_losses.yolo_boxes(
                y_pred, anchors, classes)
            pred_xy = pred_xywh[..., 0:2]
            pred_wh = pred_xywh[..., 2:4]
            true_box, true_obj, true_class_idx = tf.split(
                y_true, (4, 1, 1), axis=-1)
            true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
            true_wh = true_box[..., 2:4] - true_box[..., 0:2]
            box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
            grid_size = tf.shape(y_true)[1]
            grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
            grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
            true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
            true_wh = tf.math.log(true_wh / anchors)
            true_wh = tf.where(tf.math.is_inf(true_wh),
                               tf.zeros_like(true_wh), true_wh)
            obj_mask = tf.squeeze(true_obj, -1)
            best_iou = tf.map_fn(
                lambda x: tf.reduce_max(Yolov3_losses.broadcast_iou(x[0], tf.boolean_mask(
                    x[1], tf.cast(x[2], tf.bool))), axis=-1),
                (pred_box, true_box, obj_mask),
                tf.float32)
            ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
            xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
            obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
            class_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(
                true_class_idx, pred_class)
            xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
            wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
            obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
            class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
            return xy_loss + wh_loss + obj_loss + class_loss

        return yolo_loss


# Transformer
class Transformer(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, rate=0.1,
                 activation="relu"):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_encoder_layers, d_model, nhead, dim_feedforward,
                               rate)

        self.decoder = Decoder(num_decoder_layers, d_model, nhead, dim_feedforward,
                               rate)

    def call(self, inp, tar, enc_padding_mask=None,
             look_ahead_mask=None, dec_padding_mask=None):
        enc_output = self.encoder(inp, mask=enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)

        return dec_output, attention_weights


########################################
# 分割线
########################################


########################################
## 模型定义
########################################


class Yolo_model(object):
    @staticmethod
    def DarknetConv2D_BN_Mish(*args, **kwargs):
        no_bias_kwargs = {{'use_bias': False}}
        no_bias_kwargs.update(kwargs)
        return compose(
            DarknetConv2D(*args, **no_bias_kwargs),
            tf.keras.layers.BatchNormalization(),
            # tfa.layers.GroupNormalization(),
            Mish())

    @staticmethod
    def DarknetConv2D_BN_Leaky(*args, **kwargs):
        no_bias_kwargs = {{'use_bias': False}}
        no_bias_kwargs.update(kwargs)
        return compose(
            DarknetConv2D(*args, **no_bias_kwargs),
            tf.keras.layers.BatchNormalization(),
            # tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1))

    @staticmethod
    def resblock_body(x, num_filters, num_blocks, all_narrow=True):
        # 进行长和宽的压缩
        preconv1 = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        preconv1 = Yolo_model.DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(preconv1)

        # 生成一个大的残差边
        shortconv = Yolo_model.DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)

        # 主干部分的卷积
        mainconv = Yolo_model.DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
        # 1x1卷积对通道数进行整合->3x3卷积提取特征，使用残差结构
        for i in range(num_blocks):
            y = compose(
                Yolo_model.DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
                Yolo_model.DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3)))(mainconv)
            mainconv = tf.keras.layers.Add()([mainconv, y])
        # 1x1卷积后和残差边堆叠
        postconv = Yolo_model.DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(mainconv)
        route = tf.keras.layers.Concatenate()([postconv, shortconv])

        # 最后对通道数进行整合
        return Yolo_model.DarknetConv2D_BN_Mish(num_filters, (1, 1))(route)

    @staticmethod
    def darknet_body(x):
        x = Yolo_model.DarknetConv2D_BN_Mish(32, (3, 3))(x)
        x = Yolo_model.resblock_body(x, 64, 1, False)
        x = Yolo_model.resblock_body(x, 128, 2)
        x = Yolo_model.resblock_body(x, 256, 8)
        feat1 = x
        x = Yolo_model.resblock_body(x, 512, 8)
        feat2 = x
        x = Yolo_model.resblock_body(x, 1024, 4)
        feat3 = x
        return feat1, feat2, feat3

    @staticmethod
    def make_five_convs(x, num_filters):
        # 五次卷积
        x = Yolo_model.DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
        x = Yolo_model.DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
        x = Yolo_model.DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
        x = Yolo_model.DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
        x = Yolo_model.DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
        return x

    @staticmethod
    def yolo_body(inputs, num_anchors, num_classes):
        # 生成darknet53的主干模型
        feat1, feat2, feat3 = Yolo_model.darknet_body(inputs)

        # 第一个特征层
        # y1=(batch_size,13,13,3,85)
        P5 = Yolo_model.DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
        P5 = Yolo_model.DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
        P5 = Yolo_model.DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
        # 使用了SPP结构，即不同尺度的最大池化后堆叠。
        maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
        maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
        maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
        P5 = tf.keras.layers.Concatenate()([maxpool1, maxpool2, maxpool3, P5])
        P5 = Yolo_model.DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
        P5 = Yolo_model.DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
        P5 = Yolo_model.DarknetConv2D_BN_Leaky(512, (1, 1))(P5)

        P5_upsample = compose(Yolo_model.DarknetConv2D_BN_Leaky(256, (1, 1)), tf.keras.layers.UpSampling2D(2))(P5)

        P4 = Yolo_model.DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
        P4 = tf.keras.layers.Concatenate()([P4, P5_upsample])
        P4 = Yolo_model.make_five_convs(P4, 256)

        P4_upsample = compose(Yolo_model.DarknetConv2D_BN_Leaky(128, (1, 1)), tf.keras.layers.UpSampling2D(2))(P4)

        P3 = Yolo_model.DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
        P3 = tf.keras.layers.Concatenate()([P3, P4_upsample])
        P3 = Yolo_model.make_five_convs(P3, 128)

        P3_output = Yolo_model.DarknetConv2D_BN_Leaky(256, (3, 3))(P3)
        P3_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P3_output)

        # 38x38 output
        P3_downsample = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(P3)
        P3_downsample = Yolo_model.DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(P3_downsample)
        P4 = tf.keras.layers.Concatenate()([P3_downsample, P4])
        P4 = Yolo_model.make_five_convs(P4, 256)

        P4_output = Yolo_model.DarknetConv2D_BN_Leaky(512, (3, 3))(P4)
        P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)

        # 19x19 output
        P4_downsample = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(P4)
        P4_downsample = Yolo_model.DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(P4_downsample)
        P5 = tf.keras.layers.Concatenate()([P4_downsample, P5])
        P5 = Yolo_model.make_five_convs(P5, 512)

        P5_output = Yolo_model.DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
        P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)

        return tf.keras.Model(inputs, [P5_output, P4_output, P3_output])


class Yolo_tiny_model(object):
    @staticmethod
    def route_group(input_layer, groups, group_id):
        # 对通道数进行均等分割，我们取第二部分
        convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
        return convs[group_id]

    @staticmethod
    def resblock_body(x, num_filters):
        # 特征整合
        x = Yolo_model.DarknetConv2D_BN_Leaky(num_filters, (3, 3))(x)
        # 残差边route
        route = x
        # 通道分割
        x = tf.keras.layers.Lambda(Yolo_tiny_model.route_group, arguments={{'groups': 2, 'group_id': 1}})(x)
        x = Yolo_model.DarknetConv2D_BN_Leaky(int(num_filters / 2), (3, 3))(x)

        # 小残差边route1
        route_1 = x
        x = Yolo_model.DarknetConv2D_BN_Leaky(int(num_filters / 2), (3, 3))(x)
        # 堆叠
        x = tf.keras.layers.Concatenate()([x, route_1])

        x = Yolo_model.DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
        # 第三个resblockbody会引出来一个有效特征层分支
        feat = x
        # 连接
        x = tf.keras.layers.Concatenate()([route, x])
        x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], )(x)

        # 最后对通道数进行整合
        return x, feat

    @staticmethod
    def darknet_body(x):
        # 进行长和宽的压缩
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        # 416,416,3 -> 208,208,32
        x = Yolo_model.DarknetConv2D_BN_Leaky(32, (3, 3), strides=(2, 2))(x)

        # 进行长和宽的压缩
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        # 208,208,32 -> 104,104,64
        x = Yolo_model.DarknetConv2D_BN_Leaky(64, (3, 3), strides=(2, 2))(x)
        # 104,104,64 -> 52,52,128
        x, _ = Yolo_tiny_model.resblock_body(x, num_filters=64)
        # 52,52,128 -> 26,26,256
        x, _ = Yolo_tiny_model.resblock_body(x, num_filters=128)
        # 26,26,256 -> 13,13,512
        # feat1的shape = 26,26,256
        x, feat1 = Yolo_tiny_model.resblock_body(x, num_filters=256)

        x = Yolo_model.DarknetConv2D_BN_Leaky(512, (3, 3))(x)

        feat2 = x
        return feat1, feat2

    @staticmethod
    def yolo_body(inputs, num_anchors, num_classes):
        # 生成darknet53的主干模型
        # 首先我们会获取到两个有效特征层
        # feat1 26x26x256
        # feat2 13x13x512
        feat1, feat2 = Yolo_tiny_model.darknet_body(inputs)
        # 13x13x512 -> 13x13x256
        P5 = Yolo_model.DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)

        P5_output = Yolo_model.DarknetConv2D_BN_Leaky(512, (3, 3))(P5)
        P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)

        # Conv+UpSampling2D 13x13x256 -> 26x26x128
        P5_upsample = compose(Yolo_model.DarknetConv2D_BN_Leaky(128, (1, 1)), tf.keras.layers.UpSampling2D(2))(P5)

        # 26x26x(128+256) 26x26x384
        P4 = tf.keras.layers.Concatenate()([feat1, P5_upsample])

        P4_output = Yolo_model.DarknetConv2D_BN_Leaky(256, (3, 3))(P4)
        P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)
        return tf.keras.Model(inputs, [P5_output, P4_output])

    @staticmethod
    def yolo_body_ghostdet(inputs, num_anchors, num_classes):
        x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation=None, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = GBNeck(dwkernel=3, strides=1, exp=16, out=16, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=3, strides=2, exp=48, out=24, ratio=2, use_se=False)(x)  # 208
        x = GBNeck(dwkernel=3, strides=1, exp=72, out=24, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=5, strides=2, exp=72, out=40, ratio=2, use_se=True)(x)  # 104
        x = GBNeck(dwkernel=5, strides=1, exp=120, out=40, ratio=2, use_se=True)(x)
        feat1 = GBNeck(dwkernel=3, strides=2, exp=240, out=256, ratio=2, use_se=False)(x)  # 26
        feat2 = GBNeck(dwkernel=3, strides=1, exp=200, out=80, ratio=2, use_se=False)(feat1)
        feat2 = GBNeck(dwkernel=3, strides=1, exp=184, out=80, ratio=2, use_se=False)(feat2)
        feat2 = GBNeck(dwkernel=3, strides=1, exp=184, out=80, ratio=2, use_se=False)(feat2)
        feat2 = GBNeck(dwkernel=3, strides=1, exp=480, out=112, ratio=2, use_se=True)(feat2)
        feat2 = GBNeck(dwkernel=3, strides=1, exp=672, out=112, ratio=2, use_se=True)(feat2)
        feat2 = GBNeck(dwkernel=5, strides=2, exp=672, out=160, ratio=2, use_se=True)(feat2)  # 13
        feat2 = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=False)(feat2)
        feat2 = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=True)(feat2)
        feat2 = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=False)(feat2)
        feat2 = GBNeck(dwkernel=5, strides=1, exp=960, out=512, ratio=2, use_se=True)(feat2)
        # logger.debug(feat1)
        # logger.debug(feat2)

        P5 = Yolo_model.DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)

        P5_output = Yolo_model.DarknetConv2D_BN_Leaky(512, (3, 3))(P5)
        P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)

        # Conv+UpSampling2D 13x13x256 -> 26x26x128
        P5_upsample = compose(Yolo_model.DarknetConv2D_BN_Leaky(128, (1, 1)), tf.keras.layers.UpSampling2D(2))(P5)

        # 26x26x(128+256) 26x26x384
        P4 = tf.keras.layers.Concatenate()([feat1, P5_upsample])

        P4_output = Yolo_model.DarknetConv2D_BN_Leaky(256, (3, 3))(P4)
        P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)
        return tf.keras.Model(inputs, [P5_output, P4_output])


# object_detection
class GhostNet_det(object):

    @staticmethod
    def ghostnet(x):
        x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x1 = GBNeck(dwkernel=3, strides=1, exp=16, out=16, ratio=2, use_se=False)(x)
        x2 = GBNeck(dwkernel=3, strides=2, exp=48, out=24, ratio=2, use_se=False)(x1)
        x2 = GBNeck(dwkernel=3, strides=1, exp=72, out=24, ratio=2, use_se=False)(x2)
        x3 = GBNeck(dwkernel=5, strides=2, exp=72, out=40, ratio=2, use_se=True)(x2)
        x3 = GBNeck(dwkernel=5, strides=1, exp=120, out=40, ratio=2, use_se=True)(x3)
        x4 = GBNeck(dwkernel=3, strides=2, exp=240, out=80, ratio=2, use_se=False)(x3)
        x4 = GBNeck(dwkernel=3, strides=1, exp=200, out=80, ratio=2, use_se=False)(x4)
        x4 = GBNeck(dwkernel=3, strides=1, exp=184, out=80, ratio=2, use_se=False)(x4)
        x4 = GBNeck(dwkernel=3, strides=1, exp=184, out=80, ratio=2, use_se=False)(x4)
        x4 = GBNeck(dwkernel=3, strides=1, exp=480, out=112, ratio=2, use_se=True)(x4)
        x4 = GBNeck(dwkernel=3, strides=1, exp=672, out=112, ratio=2, use_se=True)(x4)
        x5 = GBNeck(dwkernel=5, strides=2, exp=672, out=160, ratio=2, use_se=True)(x4)
        x5 = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=False)(x5)
        x5 = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=True)(x5)
        x5 = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=False)(x5)
        x5 = GBNeck(dwkernel=5, strides=1, exp=960, out=320, ratio=2, use_se=True)(x5)
        x1 = tf.keras.layers.Activation('relu')(x1)
        x2 = tf.keras.layers.Activation('relu')(x2)
        x3 = tf.keras.layers.Activation('relu')(x3)
        x4 = tf.keras.layers.Activation('relu')(x4)
        x5 = tf.keras.layers.Activation('relu')(x5)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x5 = tf.keras.layers.BatchNormalization()(x5)
        return x1, x2, x3, x4, x5

    @staticmethod
    def ghostdet(x):
        x1, x2, x3, x4, x5 = x
        x1 = GBNeck(dwkernel=5, strides=1, exp=72, out=64, ratio=2, use_se=True)(x1)
        x2 = GBNeck(dwkernel=5, strides=1, exp=72, out=64, ratio=2, use_se=True)(x2)
        x3 = GBNeck(dwkernel=5, strides=1, exp=72, out=64, ratio=2, use_se=True)(x3)
        x4 = GBNeck(dwkernel=5, strides=1, exp=72, out=64, ratio=2, use_se=True)(x4)
        x5 = GBNeck(dwkernel=5, strides=1, exp=72, out=64, ratio=2, use_se=True)(x5)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x5 = tf.keras.layers.BatchNormalization()(x5)
        return x1, x2, x3, x4, x5


# Efficientdet
def Efficientdet(width_coefficient, depth_coefficient, default_resolution, dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS, inputs=None):
    features = []
    img_input = inputs

    bn_axis = 3
    activation = Efficientdet_anchors.get_swish()

    x = img_input
    x = tf.keras.layers.Conv2D(Efficientdet_anchors.round_filters(32, width_coefficient, depth_divisor), 3,
                               strides=(2, 2),
                               padding='same',
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = tf.keras.layers.Activation(activation)(x)

    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0

        block_args = block_args._replace(
            input_filters=Efficientdet_anchors.round_filters(block_args.input_filters,
                                                             width_coefficient, depth_divisor),
            output_filters=Efficientdet_anchors.round_filters(block_args.output_filters,
                                                              width_coefficient, depth_divisor),
            num_repeat=Efficientdet_anchors.round_repeats(block_args.num_repeat, depth_coefficient))

        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = Efficientdet_anchors.mb_conv_block(x, block_args,
                                               activation=activation,
                                               drop_rate=drop_rate)
        block_num += 1
        if block_args.num_repeat > 1:

            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])

            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                x = Efficientdet_anchors.mb_conv_block(x, block_args,
                                                       activation=activation,
                                                       drop_rate=drop_rate)
                block_num += 1
        if idx < len(blocks_args) - 1 and blocks_args[idx + 1].strides[0] == 2:
            features.append(x)
        elif idx == len(blocks_args) - 1:
            features.append(x)
    return features


class Model_Structure(object):
    @staticmethod
    def densenet_dense_block(x, blocks, name):
        for i in range(blocks):
            x = Model_Structure.densenet_conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    @staticmethod
    def densenet_transition_block(x, reduction, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
            x)
        x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
        x = tf.keras.layers.Conv2D(
            int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            1,
            use_bias=False,
            name=name + '_conv')(
            x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    @staticmethod
    def densenet_conv_block(x, growth_rate, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x1 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
            x)
        x1 = tf.keras.layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
            x1)
        x1 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
            x1)
        x1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    @staticmethod
    def efficientnet_block(inputs,
                           activation='swish',
                           drop_rate=0.,
                           name='',
                           filters_in=32,
                           filters_out=16,
                           kernel_size=3,
                           strides=1,
                           expand_ratio=1,
                           se_ratio=0.,
                           id_skip=True):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        # Expansion phase
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = tf.keras.layers.Conv2D(
                filters,
                1,
                padding='same',
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'expand_conv')(
                inputs)
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
            x = tf.keras.layers.Activation(activation, name=name + 'expand_activation')(x)
        else:
            x = inputs

        # Depthwise Convolution
        if strides == 2:
            x = tf.keras.layers.ZeroPadding2D(
                padding=tf.python.keras.applications.imagenet_utils.correct_pad(x, kernel_size),
                name=name + 'dwconv_pad')(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'dwconv')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
        x = tf.keras.layers.Activation(activation, name=name + 'activation')(x)

        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = tf.keras.layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
            se = tf.keras.layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
            se = tf.keras.layers.Conv2D(
                filters_se,
                1,
                padding='same',
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'se_reduce')(
                se)
            se = tf.keras.layers.Conv2D(
                filters,
                1,
                padding='same',
                activation='sigmoid',
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'se_expand')(se)
            x = tf.keras.layers.multiply([x, se], name=name + 'se_excite')

        # Output phase
        x = tf.keras.layers.Conv2D(
            filters_out,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'project_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = tf.keras.layers.Dropout(
                    drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
            x = tf.keras.layers.add([x, inputs], name=name + 'add')
        return x

    @staticmethod
    def resnet_block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
        bn_axis = 3 if tf.python.keras.backend.image_data_format() == 'channels_last' else 1

        if conv_shortcut:
            shortcut = tf.keras.layers.Conv2D(
                4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
            shortcut = tf.keras.layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
        else:
            shortcut = x

        x = tf.keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

        x = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

        x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

        x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
        x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
        return x

    @staticmethod
    def resnet_stack1(x, filters, blocks, stride1=2, name=None):

        x = Model_Structure.resnet_block1(x, filters, stride=stride1, name=name + '_block1')
        for i in range(2, blocks + 1):
            x = Model_Structure.resnet_block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
        return x

    @staticmethod
    def resnet_block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):

        bn_axis = 3 if tf.python.keras.backend.image_data_format() == 'channels_last' else 1

        preact = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
        preact = tf.keras.layers.Activation('relu', name=name + '_preact_relu')(preact)

        if conv_shortcut:
            shortcut = tf.keras.layers.Conv2D(
                4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
        else:
            shortcut = tf.keras.layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

        x = tf.keras.layers.Conv2D(
            filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            name=name + '_2_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

        x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
        x = tf.keras.layers.Add(name=name + '_out')([shortcut, x])
        return x

    @staticmethod
    def resnet_stack2(x, filters, blocks, stride1=2, name=None):

        x = Model_Structure.resnet_block2(x, filters, conv_shortcut=True, name=name + '_block1')
        for i in range(2, blocks):
            x = Model_Structure.resnet_block2(x, filters, name=name + '_block' + str(i))
        x = Model_Structure.resnet_block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
        return x

    @staticmethod
    def resnet_block3(x,
                      filters,
                      kernel_size=3,
                      stride=1,
                      groups=32,
                      conv_shortcut=True,
                      name=None):

        bn_axis = 3 if tf.python.keras.backend.image_data_format() == 'channels_last' else 1

        if conv_shortcut:
            shortcut = tf.keras.layers.Conv2D(
                (64 // groups) * filters,
                1,
                strides=stride,
                use_bias=False,
                name=name + '_0_conv')(x)
            shortcut = tf.keras.layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
        else:
            shortcut = x

        x = tf.keras.layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

        c = filters // groups
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides=stride,
            depth_multiplier=c,
            use_bias=False,
            name=name + '_2_conv')(x)
        x_shape = tf.python.keras.backend.int_shape(x)[1:-1]
        x = tf.keras.layers.Reshape(x_shape + (groups, c, c))(x)
        x = tf.keras.layers.Lambda(
            lambda x: sum(x[:, :, :, :, i] for i in range(c)),
            name=name + '_2_reduce')(x)
        x = tf.keras.layers.Reshape(x_shape + (filters,))(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

        x = tf.keras.layers.Conv2D(
            (64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

        x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
        x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
        return x

    @staticmethod
    def resnet_stack3(x, filters, blocks, stride1=2, groups=32, name=None):

        x = Model_Structure.resnet_block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
        for i in range(2, blocks + 1):
            x = Model_Structure.resnet_block3(
                x,
                filters,
                groups=groups,
                conv_shortcut=False,
                name=name + '_block' + str(i))
        return x

    @staticmethod
    def inception_resnet_conv2d_bn(x,
                                   filters,
                                   kernel_size,
                                   strides=1,
                                   padding='same',
                                   activation='relu',
                                   use_bias=False,
                                   name=None):

        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=name)(
            x)
        if not use_bias:
            bn_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
            bn_name = None if name is None else name + '_bn'
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = None if name is None else name + '_ac'
            x = tf.keras.layers.Activation(activation, name=ac_name)(x)
        return x

    @staticmethod
    def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
        if block_type == 'block35':
            branch_0 = Model_Structure.inception_resnet_conv2d_bn(x, 32, 1)
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(x, 32, 1)
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 32, 3)
            branch_2 = Model_Structure.inception_resnet_conv2d_bn(x, 32, 1)
            branch_2 = Model_Structure.inception_resnet_conv2d_bn(branch_2, 48, 3)
            branch_2 = Model_Structure.inception_resnet_conv2d_bn(branch_2, 64, 3)
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = Model_Structure.inception_resnet_conv2d_bn(x, 192, 1)
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(x, 128, 1)
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 160, [1, 7])
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 192, [7, 1])
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = Model_Structure.inception_resnet_conv2d_bn(x, 192, 1)
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(x, 192, 1)
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 224, [1, 3])
            branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 256, [3, 1])
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))

        block_name = block_type + '_' + str(block_idx)
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
        mixed = tf.keras.layers.Concatenate(
            axis=channel_axis, name=block_name + '_mixed')(
            branches)
        up = Model_Structure.inception_resnet_conv2d_bn(
            mixed,
            tf.keras.backend.int_shape(x)[channel_axis],
            1,
            activation=None,
            use_bias=True,
            name=block_name + '_conv')

        x = tf.keras.layers.Lambda(
            lambda inputs, scale: inputs[0] + inputs[1] * scale,
            output_shape=tf.keras.backend.int_shape(x)[1:],
            arguments={{'scale': scale}},
            name=block_name)([x, up])
        if activation is not None:
            x = tf.keras.layers.Activation(activation, name=block_name + '_ac')(x)
        return x

    @staticmethod
    def inception_conv2d_bn(x,
                            filters,
                            num_row,
                            num_col,
                            padding='same',
                            strides=(1, 1),
                            name=None):

        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if tf.keras.backend.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = tf.keras.layers.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(
            x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = tf.keras.layers.Activation('relu', name=name)(x)
        return x

    @staticmethod
    def mobilenet_conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        filters = int(filters * alpha)
        x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel,
            padding='valid',
            use_bias=False,
            strides=strides,
            name='conv1')(
            x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
        return tf.keras.layers.ReLU(6., name='conv1_relu')(x)

    @staticmethod
    def mobilenet_depthwise_conv_block(inputs,
                                       pointwise_conv_filters,
                                       alpha,
                                       depth_multiplier=1,
                                       strides=(1, 1),
                                       block_id=1):
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        if strides == (1, 1):
            x = inputs
        else:
            x = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(
                inputs)
        x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                            padding='same' if strides == (1, 1) else 'valid',
                                            depth_multiplier=depth_multiplier,
                                            strides=strides,
                                            use_bias=False,
                                            name='conv_dw_%d' % block_id)(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, name='conv_dw_%d_bn' % block_id)(
            x)
        x = tf.keras.layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

        x = tf.keras.layers.Conv2D(
            pointwise_conv_filters, (1, 1),
            padding='same',
            use_bias=False,
            strides=(1, 1),
            name='conv_pw_%d' % block_id)(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, name='conv_pw_%d_bn' % block_id)(
            x)
        return tf.keras.layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    @staticmethod
    def mobilenet_v2_make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @staticmethod
    def mobilenet_v2_inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):

        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        in_channels = tf.keras.backend.int_shape(inputs)[channel_axis]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = Model_Structure.mobilenet_v2_make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{{}}_'.format(block_id)

        if block_id:
            # Expand
            x = tf.keras.layers.Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None,
                name=prefix + 'expand')(
                x)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand_BN')(
                x)
            x = tf.keras.layers.ReLU(6., name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        if stride == 2:
            x = tf.keras.layers.ZeroPadding2D(
                padding=tf.python.keras.applications.imagenet_utils.correct_pad(x, 3),
                name=prefix + 'pad')(x)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            activation=None,
            use_bias=False,
            padding='same' if stride == 1 else 'valid',
            name=prefix + 'depthwise')(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'depthwise_BN')(
            x)

        x = tf.keras.layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

        # Project
        x = tf.keras.layers.Conv2D(
            pointwise_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'project')(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'project_BN')(
            x)

        if in_channels == pointwise_filters and stride == 1:
            return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])
        return x

    @staticmethod
    def nasnetmobile_separable_conv_block(ip,
                                          filters,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          block_id=None):

        channel_dim = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        with tf.keras.backend.name_scope('separable_conv_block_%s' % block_id):
            x = tf.keras.layers.Activation('relu')(ip)
            if strides == (2, 2):
                x = tf.keras.layers.ZeroPadding2D(
                    padding=tf.python.keras.applications.imagenet_utils.correct_pad(x, kernel_size),
                    name='separable_conv_1_pad_%s' % block_id)(x)
                conv_pad = 'valid'
            else:
                conv_pad = 'same'
            x = tf.keras.layers.SeparableConv2D(
                filters,
                kernel_size,
                strides=strides,
                name='separable_conv_1_%s' % block_id,
                padding=conv_pad,
                use_bias=False,
                kernel_initializer='he_normal')(
                x)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_dim,
                momentum=0.9997,
                epsilon=1e-3,
                name='separable_conv_1_bn_%s' % (block_id))(
                x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(
                filters,
                kernel_size,
                name='separable_conv_2_%s' % block_id,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal')(
                x)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_dim,
                momentum=0.9997,
                epsilon=1e-3,
                name='separable_conv_2_bn_%s' % (block_id))(
                x)
        return x

    @staticmethod
    def nasnetmobile_adjust_block(p, ip, filters, block_id=None):

        channel_dim = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        img_dim = 2 if tf.keras.backend.image_data_format() == 'channels_first' else -2

        ip_shape = tf.keras.backend.int_shape(ip)

        if p is not None:
            p_shape = tf.keras.backend.int_shape(p)

        with tf.keras.backend.name_scope('adjust_block'):
            if p is None:
                p = ip

            elif p_shape[img_dim] != ip_shape[img_dim]:
                with tf.keras.backend.name_scope('adjust_reduction_block_%s' % block_id):
                    p = tf.keras.layers.Activation('relu', name='adjust_relu_1_%s' % block_id)(p)
                    p1 = tf.keras.layers.AveragePooling2D((1, 1),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          name='adjust_avg_pool_1_%s' % block_id)(
                        p)
                    p1 = tf.keras.layers.Conv2D(
                        filters // 2, (1, 1),
                        padding='same',
                        use_bias=False,
                        name='adjust_conv_1_%s' % block_id,
                        kernel_initializer='he_normal')(
                        p1)

                    p2 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                    p2 = tf.keras.layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                    p2 = tf.keras.layers.AveragePooling2D((1, 1),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          name='adjust_avg_pool_2_%s' % block_id)(
                        p2)
                    p2 = tf.keras.layers.Conv2D(
                        filters // 2, (1, 1),
                        padding='same',
                        use_bias=False,
                        name='adjust_conv_2_%s' % block_id,
                        kernel_initializer='he_normal')(
                        p2)

                    p = tf.keras.layers.concatenate([p1, p2], axis=channel_dim)
                    p = tf.keras.layers.BatchNormalization(
                        axis=channel_dim,
                        momentum=0.9997,
                        epsilon=1e-3,
                        name='adjust_bn_%s' % block_id)(
                        p)

            elif p_shape[channel_dim] != filters:
                with tf.keras.backend.name_scope('adjust_projection_block_%s' % block_id):
                    p = tf.keras.layers.Activation('relu')(p)
                    p = tf.keras.layers.Conv2D(
                        filters, (1, 1),
                        strides=(1, 1),
                        padding='same',
                        name='adjust_conv_projection_%s' % block_id,
                        use_bias=False,
                        kernel_initializer='he_normal')(
                        p)
                    p = tf.keras.layers.BatchNormalization(
                        axis=channel_dim,
                        momentum=0.9997,
                        epsilon=1e-3,
                        name='adjust_bn_%s' % block_id)(
                        p)
        return p

    @staticmethod
    def nasnetmobile_normal_a_cell(ip, p, filters, block_id=None):

        channel_dim = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        with tf.keras.backend.name_scope('normal_A_block_%s' % block_id):
            p = Model_Structure.nasnetmobile_adjust_block(p, ip, filters, block_id)

            h = tf.keras.layers.Activation('relu')(ip)
            h = tf.keras.layers.Conv2D(
                filters, (1, 1),
                strides=(1, 1),
                padding='same',
                name='normal_conv_1_%s' % block_id,
                use_bias=False,
                kernel_initializer='he_normal')(
                h)
            h = tf.keras.layers.BatchNormalization(
                axis=channel_dim,
                momentum=0.9997,
                epsilon=1e-3,
                name='normal_bn_1_%s' % block_id)(
                h)

            with tf.keras.backend.name_scope('block_1'):
                x1_1 = Model_Structure.nasnetmobile_separable_conv_block(
                    h, filters, kernel_size=(5, 5), block_id='normal_left1_%s' % block_id)
                x1_2 = Model_Structure.nasnetmobile_separable_conv_block(
                    p, filters, block_id='normal_right1_%s' % block_id)
                x1 = tf.keras.layers.add([x1_1, x1_2], name='normal_add_1_%s' % block_id)

            with tf.keras.backend.name_scope('block_2'):
                x2_1 = Model_Structure.nasnetmobile_separable_conv_block(
                    p, filters, (5, 5), block_id='normal_left2_%s' % block_id)
                x2_2 = Model_Structure.nasnetmobile_separable_conv_block(
                    p, filters, (3, 3), block_id='normal_right2_%s' % block_id)
                x2 = tf.keras.layers.add([x2_1, x2_2], name='normal_add_2_%s' % block_id)

            with tf.keras.backend.name_scope('block_3'):
                x3 = tf.keras.layers.AveragePooling2D((3, 3),
                                                      strides=(1, 1),
                                                      padding='same',
                                                      name='normal_left3_%s' % (block_id))(
                    h)
                x3 = tf.keras.layers.add([x3, p], name='normal_add_3_%s' % block_id)

            with tf.keras.backend.name_scope('block_4'):
                x4_1 = tf.keras.layers.AveragePooling2D((3, 3),
                                                        strides=(1, 1),
                                                        padding='same',
                                                        name='normal_left4_%s' % (block_id))(
                    p)
                x4_2 = tf.keras.layers.AveragePooling2D((3, 3),
                                                        strides=(1, 1),
                                                        padding='same',
                                                        name='normal_right4_%s' % (block_id))(
                    p)
                x4 = tf.keras.layers.add([x4_1, x4_2], name='normal_add_4_%s' % block_id)

            with tf.keras.backend.name_scope('block_5'):
                x5 = Model_Structure.nasnetmobile_separable_conv_block(
                    h, filters, block_id='normal_left5_%s' % block_id)
                x5 = tf.keras.layers.add([x5, h], name='normal_add_5_%s' % block_id)

            x = tf.keras.layers.concatenate([p, x1, x2, x3, x4, x5],
                                            axis=channel_dim,
                                            name='normal_concat_%s' % block_id)
        return x, ip

    @staticmethod
    def nasnetmobile_reduction_a_cell(ip, p, filters, block_id=None):

        channel_dim = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        with tf.keras.backend.name_scope('reduction_A_block_%s' % block_id):
            p = Model_Structure.nasnetmobile_adjust_block(p, ip, filters, block_id)

            h = tf.keras.layers.Activation('relu')(ip)
            h = tf.keras.layers.Conv2D(
                filters, (1, 1),
                strides=(1, 1),
                padding='same',
                name='reduction_conv_1_%s' % block_id,
                use_bias=False,
                kernel_initializer='he_normal')(
                h)
            h = tf.keras.layers.BatchNormalization(
                axis=channel_dim,
                momentum=0.9997,
                epsilon=1e-3,
                name='reduction_bn_1_%s' % block_id)(
                h)
            h3 = tf.keras.layers.ZeroPadding2D(
                padding=tf.python.keras.applications.imagenet_utils.correct_pad(h, 3),
                name='reduction_pad_1_%s' % block_id)(
                h)

            with tf.keras.backend.name_scope('block_1'):
                x1_1 = Model_Structure.nasnetmobile_separable_conv_block(
                    h,
                    filters, (5, 5),
                    strides=(2, 2),
                    block_id='reduction_left1_%s' % block_id)
                x1_2 = Model_Structure.nasnetmobile_separable_conv_block(
                    p,
                    filters, (7, 7),
                    strides=(2, 2),
                    block_id='reduction_right1_%s' % block_id)
                x1 = tf.keras.layers.add([x1_1, x1_2], name='reduction_add_1_%s' % block_id)

            with tf.keras.backend.name_scope('block_2'):
                x2_1 = tf.keras.layers.MaxPooling2D((3, 3),
                                                    strides=(2, 2),
                                                    padding='valid',
                                                    name='reduction_left2_%s' % block_id)(
                    h3)
                x2_2 = Model_Structure.nasnetmobile_separable_conv_block(
                    p,
                    filters, (7, 7),
                    strides=(2, 2),
                    block_id='reduction_right2_%s' % block_id)
                x2 = tf.keras.layers.add([x2_1, x2_2], name='reduction_add_2_%s' % block_id)

            with tf.keras.backend.name_scope('block_3'):
                x3_1 = tf.keras.layers.AveragePooling2D((3, 3),
                                                        strides=(2, 2),
                                                        padding='valid',
                                                        name='reduction_left3_%s' % block_id)(
                    h3)
                x3_2 = Model_Structure.nasnetmobile_separable_conv_block(
                    p,
                    filters, (5, 5),
                    strides=(2, 2),
                    block_id='reduction_right3_%s' % block_id)
                x3 = tf.keras.layers.add([x3_1, x3_2], name='reduction_add3_%s' % block_id)

            with tf.keras.backend.name_scope('block_4'):
                x4 = tf.keras.layers.AveragePooling2D((3, 3),
                                                      strides=(1, 1),
                                                      padding='same',
                                                      name='reduction_left4_%s' % block_id)(
                    x1)
                x4 = tf.keras.layers.add([x2, x4])

            with tf.keras.backend.name_scope('block_5'):
                x5_1 = Model_Structure.nasnetmobile_separable_conv_block(
                    x1, filters, (3, 3), block_id='reduction_left4_%s' % block_id)
                x5_2 = tf.keras.layers.MaxPooling2D((3, 3),
                                                    strides=(2, 2),
                                                    padding='valid',
                                                    name='reduction_right5_%s' % block_id)(
                    h3)
                x5 = tf.keras.layers.add([x5_1, x5_2], name='reduction_add4_%s' % block_id)

            x = tf.keras.layers.concatenate([x2, x3, x4, x5],
                                            axis=channel_dim,
                                            name='reduction_concat_%s' % block_id)
            return x, ip

    @staticmethod
    def mobilenet_v3_depth(v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @staticmethod
    def mobilenet_v3_se_block(inputs, filters, se_ratio, prefix):
        x = tf.keras.layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(
            inputs)
        if tf.keras.backend.image_data_format() == 'channels_first':
            x = tf.keras.layers.Reshape((filters, 1, 1))(x)
        else:
            x = tf.keras.layers.Reshape((1, 1, filters))(x)
        x = tf.keras.layers.Conv2D(
            Model_Structure.mobilenet_v3_depth(filters * se_ratio),
            kernel_size=1,
            padding='same',
            name=prefix + 'squeeze_excite/Conv')(
            x)
        x = tf.keras.layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            name=prefix + 'squeeze_excite/Conv_1')(
            x)
        x = tf.keras.layers.ReLU(6.)(x + 3.) * (1. / 6.)
        x = tf.keras.layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
        return x

    @staticmethod
    def mobilenet_v3_inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                                        activation, block_id):
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        shortcut = x
        prefix = 'expanded_conv/'
        infilters = tf.keras.backend.int_shape(x)[channel_axis]
        if block_id:
            # Expand
            prefix = 'expanded_conv_{{}}/'.format(block_id)
            x = tf.keras.layers.Conv2D(
                Model_Structure.mobilenet_v3_depth(infilters * expansion),
                kernel_size=1,
                padding='same',
                use_bias=False,
                name=prefix + 'expand')(
                x)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand/BatchNorm')(
                x)
            x = activation(x)

        if stride == 2:
            x = tf.keras.layers.ZeroPadding2D(
                padding=tf.python.keras.applications.imagenet_utils.correct_pad(x, kernel_size),
                name=prefix + 'depthwise/pad')(
                x)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides=stride,
            padding='same' if stride == 1 else 'valid',
            use_bias=False,
            name=prefix + 'depthwise')(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'depthwise/BatchNorm')(
            x)
        x = activation(x)

        if se_ratio:
            x = Model_Structure.mobilenet_v3_se_block(x, Model_Structure.mobilenet_v3_depth(infilters * expansion),
                                                      se_ratio, prefix)

        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'project')(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'project/BatchNorm')(
            x)

        if stride == 1 and infilters == filters:
            x = tf.keras.layers.Add(name=prefix + 'Add')([shortcut, x])
        return x

    @staticmethod
    def mnasnet_conv_bn(x, filters, kernel_size, strides=1, alpha=1, activation=True):
        filters = Model_Structure.mnasnet_make_divisible(filters * alpha)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                   use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l=0.0003))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        if activation:
            x = tf.keras.layers.ReLU(max_value=6)(x)
        return x

    @staticmethod
    def mnasnet_depthwiseConv_bn(x, depth_multiplier, kernel_size, strides=1):
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                            padding='same', use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.0003))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)
        return x

    @staticmethod
    def mnasnet_sepconv_bn_noskip(x, filters, kernel_size, strides=1):
        x = Model_Structure.mnasnet_depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
        x = Model_Structure.mnasnet_conv_bn(x, filters=filters, kernel_size=1, strides=1)
        return x

    @staticmethod
    def mnasnet_mbconv_idskip(x_input, filters, kernel_size, strides=1, filters_multiplier=1, alpha=1):
        depthwise_conv_filters = Model_Structure.mnasnet_make_divisible(x_input.shape[3])
        pointwise_conv_filters = Model_Structure.mnasnet_make_divisible(filters * alpha)

        x = Model_Structure.mnasnet_conv_bn(x_input, filters=depthwise_conv_filters * filters_multiplier, kernel_size=1,
                                            strides=1)
        x = Model_Structure.mnasnet_depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
        x = Model_Structure.mnasnet_conv_bn(x, filters=pointwise_conv_filters, kernel_size=1, strides=1,
                                            activation=False)
        if strides == 1 and x.shape[3] == x_input.shape[3]:
            return tf.keras.layers.add([x_input, x])
        else:
            return x

    @staticmethod
    def mnasnet_make_divisible(v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @staticmethod
    def squeezenet_firemodule(inputs, s1, e1, e3):
        x = tf.keras.layers.Conv2D(filters=s1,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.nn.relu(x)
        y1 = tf.keras.layers.Conv2D(filters=e1,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")(x)
        y1 = tf.nn.relu(y1)
        y2 = tf.keras.layers.Conv2D(filters=e3,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")(x)
        y2 = tf.nn.relu(y2)
        return tf.concat(values=[y1, y2], axis=-1)

    @staticmethod
    def shufflenet_v2_channel_shuffle(feature, group):
        channel_num = feature.shape[-1]
        if channel_num % group != 0:
            raise ValueError("The group must be divisible by the shape of the last dimension of the feature.")
        x = tf.reshape(feature, shape=(-1, feature.shape[1], feature.shape[2], group, channel_num // group))
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = tf.reshape(x, shape=(-1, feature.shape[1], feature.shape[2], channel_num))
        return x

    @staticmethod
    def shufflenet_v2_blocks1(inputs, in_channels, out_channels, training=None):
        branch, x = tf.split(inputs, num_or_size_splits=2, axis=-1)
        x = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        outputs = tf.concat(values=[branch, x], axis=-1)
        outputs = Model_Structure.shufflenet_v2_channel_shuffle(feature=outputs, group=2)
        return outputs

    @staticmethod
    def shufflenet_v2_blocks2(inputs, in_channels, out_channels, training=None):
        x = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Conv2D(filters=out_channels - in_channels,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        branch = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding="same")(inputs)
        branch = tf.keras.layers.BatchNormalization()(branch, training=training)
        branch = tf.keras.layers.Conv2D(filters=in_channels,
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding="same")(branch)
        branch = tf.keras.layers.BatchNormalization()(branch, training=training)
        branch = tf.nn.swish(branch)
        outputs = tf.concat(values=[x, branch], axis=-1)
        outputs = Model_Structure.shufflenet_v2_channel_shuffle(feature=outputs, group=2)
        return outputs

    @staticmethod
    def shufflenet_v2_make_layer(inputs, repeat_num, in_channels, out_channels):
        x = Model_Structure.shufflenet_v2_blocks2(inputs, in_channels=in_channels, out_channels=out_channels)
        for _ in range(1, repeat_num):
            x = Model_Structure.shufflenet_v2_blocks1(x, in_channels=out_channels, out_channels=out_channels)
        return x

    @staticmethod
    def seresnet_seblock(inputs, input_channels, r=16):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(units=input_channels // r)(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Dense(units=input_channels)(x)
        x = tf.nn.sigmoid(x)
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)
        output = tf.keras.layers.multiply(inputs=[inputs, x])
        return output

    @staticmethod
    def seresnet_bottleneck(inputs, filter_num, stride=1, training=None):
        identity = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                          kernel_size=(1, 1),
                                          strides=stride)(inputs)
        identity = tf.keras.layers.BatchNormalization()(identity)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = Model_Structure.seresnet_seblock(x, input_channels=filter_num * 4)
        output = tf.nn.swish(tf.keras.layers.add([identity, x]))
        return output

    @staticmethod
    def seresnet_make_res_block(inputs, filter_num, blocks, stride=1):
        x = Model_Structure.seresnet_bottleneck(inputs, filter_num, stride=stride)
        for _ in range(1, blocks):
            x = Model_Structure.seresnet_bottleneck(x, filter_num, stride=1)
        return x

    @staticmethod
    def resnext_basicblock(inputs, filter_num, stride=1, training=None):
        if stride != 1:
            residual = tf.keras.layers.Conv2D(filters=filter_num,
                                              kernel_size=(1, 1),
                                              strides=stride)(inputs)
            residual = tf.keras.layers.BatchNormalization()(residual, training=training)
        else:
            residual = inputs

        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        output = tf.keras.layers.concatenate([residual, x])
        return output

    @staticmethod
    def resnext_bottleneck(inputs, filter_num, stride=1, training=None):
        residual = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                          kernel_size=(1, 1),
                                          strides=stride, kernel_initializer=tf.keras.initializers.he_normal())(inputs)
        residual = tf.keras.layers.BatchNormalization()(residual, training=training)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same', kernel_initializer=tf.keras.initializers.he_normal())(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding='same', kernel_initializer=tf.keras.initializers.he_normal())(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding='same', kernel_initializer=tf.keras.initializers.he_normal())(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        return tf.nn.relu(tf.keras.layers.add([residual, x]))

    @staticmethod
    def resnext_make_basic_block_layer(inputs, filter_num, blocks, stride=1):
        res_block = Model_Structure.resnext_basicblock(inputs, filter_num, stride=stride)
        for _ in range(1, blocks):
            res_block = Model_Structure.resnext_basicblock(inputs, filter_num, stride=1)
        return res_block

    @staticmethod
    def resnext_make_bottleneck_layer(inputs, filter_num, blocks, stride=1):
        res_block = Model_Structure.resnext_bottleneck(inputs, filter_num, stride=stride)
        for _ in range(1, blocks):
            res_block = Model_Structure.resnext_bottleneck(inputs, filter_num, stride=1)
        return res_block

    @staticmethod
    def resnext_bottleneck2(inputs, filters, strides, groups, training=None):
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(3, 3),
                                   strides=strides,
                                   padding="same", )(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(filters=2 * filters,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        shortcut = tf.keras.layers.Conv2D(filters=2 * filters,
                                          kernel_size=(1, 1),
                                          strides=strides,
                                          padding="same")(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut, training=training)
        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output

    @staticmethod
    def resnext_build_ResNeXt_block(inputs, filters, strides, groups, repeat_num):
        block = Model_Structure.resnext_bottleneck2(inputs, filters=filters,
                                                    strides=strides,
                                                    groups=groups)
        for _ in range(1, repeat_num):
            block = Model_Structure.resnext_bottleneck2(inputs, filters=filters,
                                                        strides=1,
                                                        groups=groups)
        return block

    @staticmethod
    def regnet_squeeze_excite_block(input_tensor, ratio=16, input_type='2d', channel_axis=-1):
        filters = input_tensor.get_shape().as_list()[channel_axis]
        if input_type == '2d':
            se_shape = (1, 1, filters)
            se = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(input_tensor)
        elif input_type == '1d':
            se_shape = (1, filters)
            se = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(input_tensor)
        else:
            assert 1 > 2, 'squeeze_excite_block unsupport input type {{}}'.format(input_type)
        se = tf.keras.layers.Reshape(se_shape)(se)
        se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(
            se)
        se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        x = tf.keras.layers.Multiply()([input_tensor, se])
        return x

    @staticmethod
    def se_block(x, ratio=16, channel_axis=-1):
        filters = x.get_shape().as_list()[channel_axis]
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        ex = tf.keras.layers.Dense(filters // ratio, use_bias=True)(se)
        ex = tf.keras.layers.Activation('relu')(ex)
        ex = tf.keras.layers.Dense(filters, use_bias=True)(ex)
        ex = tf.keras.layers.Activation('sigmoid')(ex)
        ex = tf.reshape(ex, [-1, 1, 1, filters])
        return x * ex

    @staticmethod
    def regnet_make_attention(input_tensor, input_type='2d', SEstyle_atten='SE'):
        x = input_tensor
        if SEstyle_atten == 'SE':
            x = Model_Structure.regnet_squeeze_excite_block(x, input_type=input_type)
        return x

    @staticmethod
    def regnet_make_stem(input_tensor, filters=32, size=(7, 7), strides=2, channel_axis=-1,
                         active='relu'):
        x = input_tensor
        x = tf.keras.layers.Conv2D(filters, kernel_size=size, strides=strides,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   use_bias=False,
                                   data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
        x = tf.keras.layers.Activation(active)(x)
        return x

    @staticmethod
    def regnet_make_basic_131_block(input_tensor,
                                    filters=96,
                                    group_kernel_size=[3, 3, 3],
                                    filters_per_group=48,
                                    stridess=[1, 2, 1], channel_axis=-1, active='relu'):
        x2 = tf.identity(input_tensor)
        if 2 in stridess:
            x2 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1),
                                        strides=2,
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        use_bias=False,
                                        data_format='channels_last')(x2)
            x2 = tf.keras.layers.BatchNormalization(axis=channel_axis)(x2)
            x2 = tf.keras.layers.Activation(active)(x2)
        x = input_tensor
        x = tf.keras.layers.Conv2D(filters, kernel_size=1,
                                   strides=stridess[0],
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   use_bias=False,
                                   data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
        x = tf.keras.layers.Activation(active)(x)
        x = GroupedConv2D(filters=filters, kernel_size=group_kernel_size, strides=stridess[1],
                          use_keras=True, padding='same', kernel_initializer='he_normal',
                          use_bias=False, data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
        x = tf.keras.layers.Activation(active)(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=1,
                                   strides=stridess[2],
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   use_bias=False,
                                   data_format='channels_last')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
        x = tf.keras.layers.Activation(active)(x)
        x = Model_Structure.regnet_make_attention(x, input_type='2d')
        m2 = tf.keras.layers.Add()([x, x2])
        return m2

    @staticmethod
    def regnet_make_stage(input_tensor,
                          n_block=2,
                          block_width=96,
                          group_G=48):
        x = input_tensor
        x = Model_Structure.regnet_make_basic_131_block(x,
                                                        filters=block_width,
                                                        filters_per_group=group_G,
                                                        stridess=[1, 2, 1])
        for i in range(1, n_block):
            x = Model_Structure.regnet_make_basic_131_block(x,
                                                            filters=block_width,
                                                            filters_per_group=group_G,
                                                            stridess=[1, 1, 1])
        return x

    @staticmethod
    def resnest_make_stem(input_tensor, stem_width=64, deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = tf.keras.layers.Conv2D(stem_width, kernel_size=3, strides=2, padding="same",
                                       kernel_initializer="he_normal",
                                       use_bias=False, data_format="channels_last")(x)

            x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation('Mish_Activation')(x)

            x = tf.keras.layers.Conv2D(stem_width, kernel_size=3, strides=1, padding="same",
                                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

            x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation('Mish_Activation')(x)

            x = tf.keras.layers.Conv2D(stem_width * 2, kernel_size=3, strides=1, padding="same",
                                       kernel_initializer="he_normal",
                                       use_bias=False, data_format="channels_last")(x)

        else:
            x = tf.keras.layers.Conv2D(stem_width, kernel_size=7, strides=2, padding="same",
                                       kernel_initializer="he_normal",
                                       use_bias=False, data_format="channels_last")(x)
        return x

    @staticmethod
    def resnest_rsoftmax(input_tensor, filters, radix, groups):
        x = input_tensor
        if radix > 1:
            x = tf.reshape(x, [-1, groups, radix, filters // groups])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.keras.activations.softmax(x, axis=1)
            x = tf.reshape(x, [-1, 1, 1, radix * filters])
        else:
            x = tf.keras.layers.Activation("sigmoid")(x)
        return x

    @staticmethod
    def resnest_splatconv2d(input_tensor, filters=64, kernel_size=3, stride=1, dilation=1, groups=1, radix=0):
        x = input_tensor
        in_channels = input_tensor.shape[-1]

        x = GroupedConv2D(filters=filters * radix, kernel_size=[kernel_size for i in range(groups * radix)],
                          use_keras=True, padding="same", kernel_initializer="he_normal", use_bias=False,
                          data_format="channels_last", dilation_rate=dilation)(x)

        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation('Mish_Activation')(x)
        if radix > 1:
            splited = tf.split(x, radix, axis=-1)
            gap = sum(splited)
        else:
            gap = x

        # print('sum',gap.shape)
        gap = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(gap)
        gap = tf.reshape(gap, [-1, 1, 1, filters])
        # print('adaptive_avg_pool2d',gap.shape)
        reduction_factor = 4
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        x = tf.keras.layers.Conv2D(inter_channels, kernel_size=1)(gap)

        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation('Mish_Activation')(x)
        x = tf.keras.layers.Conv2D(filters * radix, kernel_size=1)(x)

        atten = Model_Structure.resnest_rsoftmax(x, filters, radix, groups)

        if radix > 1:
            logits = tf.split(atten, radix, axis=-1)
            out = sum([a * b for a, b in zip(splited, logits)])
        else:
            out = atten * x
        return out

    @staticmethod
    def resnest_make_block(input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False,
                           is_first=False, block_expansion=4, avg_down=True, dilation=1, bottleneck_width=64,
                           cardinality=1):
        x = input_tensor
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * block_expansion:
            short_cut = input_tensor
            if avg_down:
                if dilation == 1:
                    short_cut = tf.keras.layers.AveragePooling2D(pool_size=stride, strides=stride, padding="same",
                                                                 data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1, padding="same",
                                                                 data_format="channels_last")(
                        short_cut)
                short_cut = tf.keras.layers.Conv2D(filters * block_expansion, kernel_size=1, strides=1, padding="same",
                                                   kernel_initializer="he_normal", use_bias=False,
                                                   data_format="channels_last")(
                    short_cut)
            else:
                short_cut = tf.keras.layers.Conv2D(filters * block_expansion, kernel_size=1, strides=stride,
                                                   padding="same",
                                                   kernel_initializer="he_normal", use_bias=False,
                                                   data_format="channels_last")(
                    short_cut)

            short_cut = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(short_cut)
        else:
            short_cut = input_tensor

        group_width = int(filters * (bottleneck_width / 64.0)) * cardinality
        x = tf.keras.layers.Conv2D(group_width, kernel_size=1, strides=1, padding="same",
                                   kernel_initializer="he_normal",
                                   use_bias=False,
                                   data_format="channels_last")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation('Mish_Activation')(x)

        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding="same",
                                                         data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = Model_Structure.resnest_splatconv2d(x, filters=group_width, kernel_size=3, stride=stride,
                                                    dilation=dilation,
                                                    groups=cardinality, radix=radix)
        else:
            x = tf.keras.layers.Conv2D(group_width, kernel_size=3, strides=stride, padding="same",
                                       kernel_initializer="he_normal",
                                       dilation_rate=dilation, use_bias=False, data_format="channels_last")(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation('Mish_Activation')(x)

        if avd and not avd_first:
            x = avd_layer(x)
            # print('can')
        x = tf.keras.layers.Conv2D(filters * block_expansion, kernel_size=1, strides=1, padding="same",
                                   kernel_initializer="he_normal",
                                   dilation_rate=dilation, use_bias=False, data_format="channels_last")(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)

        m2 = tf.keras.layers.Add()([x, short_cut])
        m2 = tf.keras.layers.Activation('Mish_Activation')(m2)
        return m2

    @staticmethod
    def resnest_make_block_basic(input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False,
                                 avd_first=False,
                                 is_first=False, block_expansion=4, avg_down=True, dilation=1, bottleneck_width=64,
                                 cardinality=1):
        x = input_tensor
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation('Mish_Activation')(x)

        short_cut = x
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * block_expansion:
            if avg_down:
                if dilation == 1:
                    short_cut = tf.keras.layers.AveragePooling2D(pool_size=stride, strides=stride, padding="same",
                                                                 data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1, padding="same",
                                                                 data_format="channels_last")(
                        short_cut)
                short_cut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding="same",
                                                   kernel_initializer="he_normal",
                                                   use_bias=False, data_format="channels_last")(short_cut)
            else:
                short_cut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding="same",
                                                   kernel_initializer="he_normal",
                                                   use_bias=False, data_format="channels_last")(short_cut)

        group_width = int(filters * (bottleneck_width / 64.0)) * cardinality
        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding="same",
                                                         data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = Model_Structure.resnest_splatconv2d(x, filters=group_width, kernel_size=3, stride=stride,
                                                    dilation=dilation,
                                                    groups=cardinality, radix=radix)
        else:
            x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same",
                                       kernel_initializer="he_normal",
                                       dilation_rate=dilation, use_bias=False, data_format="channels_last")(x)

        if avd and not avd_first:
            x = avd_layer(x)

        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation('Mish_Activation')(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                                   dilation_rate=dilation, use_bias=False, data_format="channels_last")(x)
        m2 = tf.keras.layers.Add()([x, short_cut])
        return m2

    @staticmethod
    def resnest_make_layer(input_tensor, blocks=4, filters=64, stride=2, is_first=True, using_basic_block=False,
                           avd=True, radix=2, avd_first=False):
        x = input_tensor
        if using_basic_block is True:
            x = Model_Structure.resnest_make_block_basic(x, first_block=True, filters=filters, stride=stride,
                                                         radix=radix,
                                                         avd=avd, avd_first=avd_first, is_first=is_first)

            for i in range(1, blocks):
                x = Model_Structure.resnest_make_block_basic(
                    x, first_block=False, filters=filters, stride=1, radix=radix, avd=avd,
                    avd_first=avd_first
                )

        elif using_basic_block is False:
            x = Model_Structure.resnest_make_block(x, first_block=True, filters=filters, stride=stride, radix=radix,
                                                   avd=avd,
                                                   avd_first=avd_first, is_first=is_first)

            for i in range(1, blocks):
                x = Model_Structure.resnest_make_block(
                    x, first_block=False, filters=filters, stride=1, radix=radix, avd=avd,
                    avd_first=avd_first)
        return x

    @staticmethod
    def resnest_make_composite_layer(input_tensor, filters=256, kernel_size=1, stride=1, upsample=True):
        x = input_tensor
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        if upsample:
            x = tf.keras.layers.UpSampling2D(size=2)(x)
        return x

    @staticmethod
    def resnest_get_trainable_parameter(shape=(100, 128)):
        w_init = tf.random_normal_initializer()
        parameter = tf.Variable(
            initial_value=w_init(shape=shape,
                                 dtype='float32'),
            trainable=True)
        return parameter

    @staticmethod
    def resnest_make_transformer_top(x, hidden_dim=512, n_query_pos=100, nheads=8, num_encoder_layers=6,
                                     num_decoder_layers=6):
        h = tf.keras.layers.Conv2D(hidden_dim, kernel_size=1, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=True, data_format='channels_last')(x)
        H, W = h.shape[1], h.shape[2]

        query_pos = Model_Structure.resnest_get_trainable_parameter(shape=(n_query_pos, hidden_dim))
        row_embed = Model_Structure.resnest_get_trainable_parameter(shape=(100, hidden_dim // 2))
        col_embed = Model_Structure.resnest_get_trainable_parameter(shape=(100, hidden_dim // 2))

        cat1_col = tf.expand_dims(col_embed[:W], 0)
        cat1_col = tf.repeat(cat1_col, H, axis=0)

        cat2_row = tf.expand_dims(row_embed[:H], 1)
        cat2_row = tf.repeat(cat2_row, W, axis=1)
        pos = tf.concat([cat1_col, cat2_row], axis=-1)
        pos = tf.expand_dims(tf.reshape(pos, [pos.shape[0] * pos.shape[1], -1]), 0)
        h = tf.reshape(h, [-1, h.shape[1] * h.shape[2], h.shape[3]])
        temp_input = pos + h

        h_tag = tf.transpose(h, perm=[0, 2, 1])

        h_tag = tf.keras.layers.Conv1D(query_pos.shape[0], kernel_size=1, strides=1,
                                       padding='same', kernel_initializer='he_normal',
                                       use_bias=True, data_format='channels_last')(h_tag)

        h_tag = tf.transpose(h_tag, perm=[0, 2, 1])

        query_pos = tf.expand_dims(query_pos, 0)

        query_pos += h_tag
        query_pos -= h_tag

        transformer = Transformer(
            d_model=hidden_dim, nhead=nheads, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers)
        atten_out, attention_weights = transformer(temp_input, query_pos)
        return atten_out

    @staticmethod
    def sedensenet_dense_block(x, blocks, name):
        for i in range(blocks):
            x = Model_Structure.sedensenet_conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    @staticmethod
    def sedensenet_transition_block(x, reduction, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
            x)
        x = tf.keras.layers.Activation('Mish_Activation', name=name + '_relu')(x)
        x = tf.keras.layers.Conv2D(
            int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            1,
            use_bias=False,
            name=name + '_conv')(
            x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    @staticmethod
    def sedensenet_conv_block(x, growth_rate, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x1 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
            x)
        x1 = tf.keras.layers.Activation('Mish_Activation', name=name + '_0_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
            x1)
        x1 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
            x1)
        x1 = tf.keras.layers.Activation('Mish_Activation', name=name + '_1_relu')(x1)
        x1 = Model_Structure.se_block(x1)
        x1 = tf.keras.layers.Conv2D(
            growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    @staticmethod
    def hierarchical_split(x, channel):
        channel_begin = x.get_shape().as_list()[-1]
        ##第一阶段
        layer1 = tf.keras.layers.Conv2D(channel, 1, strides=1, padding='same')(x)
        layer1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(
            layer1)
        layer1 = tf.nn.relu(layer1)
        layer1_0, layer1_1, layer1_2, layer1_3, layer1_4 = tf.split(layer1, 5, -1)
        ##第二阶段
        channel1 = layer1_1.get_shape().as_list()[-1] * 2
        # layer1_1 = tf.nn.relu(
        #     tf.keras.layers.BatchNormalization(tf.keras.layers.Conv2D(layer1_1, channel1, 3, strides=1, padding='same'),
        #                                        axis=-1,
        #                                        momentum=0.99, epsilon=0.001, center=True, scale=True, ))
        layer1_1 = tf.keras.layers.Conv2D(channel1, 1, strides=1, padding='same')(layer1_1)
        layer1_1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(
            layer1_1)
        layer1_1 = tf.nn.relu(layer1_1)
        layer1_1_0, layer1_1_1 = tf.split(layer1_1, 2, -1)
        ##第三阶段
        channel2 = layer1_2.get_shape().as_list()[-1] * 2

        layer1_2 = tf.concat([layer1_2, layer1_1_1], -1)
        # layer1_2 = tf.nn.relu(
        #     tf.keras.layers.BatchNormalization(tf.keras.layers.Conv2D(layer1_2, channel2, 3, strides=1, padding='same'),
        #                                        axis=-1,
        #                                        momentum=0.99, epsilon=0.001, center=True, scale=True, ))
        layer1_2 = tf.keras.layers.Conv2D(channel2, 1, strides=1, padding='same')(layer1_2)
        layer1_2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(
            layer1_2)
        layer1_2 = tf.nn.relu(layer1_2)
        layer1_2_0, layer1_2_1 = tf.split(layer1_2, 2, -1)
        ##第四阶段
        channel3 = layer1_3.get_shape().as_list()[-1] * 2
        layer1_3 = tf.concat([layer1_3, layer1_2_1], -1)
        # layer1_3 = tf.nn.relu(
        #     tf.keras.layers.BatchNormalization(tf.keras.layers.Conv2D(layer1_3, channel3, 3, strides=1, padding='same'),
        #                                        axis=-1,
        #                                        momentum=0.99, epsilon=0.001, center=True, scale=True, ))
        layer1_3 = tf.keras.layers.Conv2D(channel3, 1, strides=1, padding='same')(layer1_3)
        layer1_3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(
            layer1_3)
        layer1_3 = tf.nn.relu(layer1_3)
        layer1_3_0, layer1_3_1 = tf.split(layer1_3, 2, -1)
        ##第五阶段
        channel4 = layer1_4.get_shape().as_list()[-1] * 2
        layer1_4 = tf.concat([layer1_4, layer1_3_1], -1)
        # layer1_4 = tf.nn.relu(
        #     tf.keras.layers.BatchNormalization(tf.keras.layers.Conv2D(layer1_4, channel4, 3, strides=1, padding='same'),
        #                                        axis=-1,
        #                                        momentum=0.99, epsilon=0.001, center=True, scale=True, ))
        layer1_4 = tf.keras.layers.Conv2D(channel4, 1, strides=1, padding='same')(layer1_4)
        layer1_4 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(
            layer1_4)
        layer1_4 = tf.nn.relu(layer1_4)
        ##第六阶段
        layer_all = tf.concat([layer1_0, layer1_1_0, layer1_2_0, layer1_3_0, layer1_4], -1)
        # layer_all = tf.nn.relu(
        #     tf.keras.layers.BatchNormalization(
        #         tf.keras.layers.Conv2D(layer_all, channel_begin, 1, strides=1, padding='same'),
        #         axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, ))
        layer_all = tf.keras.layers.Conv2D(channel_begin, 1, strides=1, padding='same')(layer_all)
        layer_all = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(
            layer_all)
        layer_all = tf.nn.relu(layer_all)
        return x + layer_all

    @staticmethod
    def hs_resnetv2_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):

        bn_axis = 3 if tf.python.keras.backend.image_data_format() == 'channels_last' else 1

        preact = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
        preact = tf.keras.layers.Activation('Mish_Activation', name=name + '_preact_relu')(preact)

        if conv_shortcut:
            shortcut = Model_Structure.hierarchical_split(preact, 4 * filters)
        else:
            shortcut = Model_Structure.hierarchical_split(preact, 4 * filters)
        x = Model_Structure.hierarchical_split(preact, filters)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
        x = tf.keras.layers.Activation('Mish_Activation', name=name + '_1_relu')(x)

        x = Model_Structure.hierarchical_split(x, filters)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
        x = tf.keras.layers.Activation('Mish_Activation', name=name + '_2_relu')(x)
        x = Model_Structure.hierarchical_split(x, 4 * filters)
        x = tf.keras.layers.Add(name=name + '_out')([shortcut, x])
        return x

    @staticmethod
    def hs_resnetv2_stack(x, filters, blocks, stride1=2, name=None):

        x = Model_Structure.hs_resnetv2_block(x, filters, conv_shortcut=True, name=name + '_block1')
        for i in range(2, blocks):
            x = Model_Structure.hs_resnetv2_block(x, filters, name=name + '_block' + str(i))
        x = Model_Structure.hs_resnetv2_block(x, filters, stride=stride1, name=name + '_block' + str(blocks))
        return x

    @staticmethod
    def densenet_dense_block_d(x, blocks, name):
        for i in range(blocks):
            x = Model_Structure.densenet_conv_block_d(x, 32, name=name + '_block' + str(i + 1))
        return x

    @staticmethod
    def densenet_transition_block_d(x, reduction, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x = FRN()(x)
        x = tf.keras.layers.Activation('Mish_Activation', name=name + '_relu')(x)
        x = tf.keras.layers.Conv2D(
            int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            1,
            use_bias=False,
            name=name + '_conv')(
            x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    @staticmethod
    def densenet_conv_block_d(x, growth_rate, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x1 = FRN()(x)
        x1 = tf.keras.layers.Activation('Mish_Activation', name=name + '_0_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
            x1)
        x1 = FRN()(x1)
        x1 = tf.keras.layers.Activation('Mish_Activation', name=name + '_1_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
        x1 = Model_Structure.regnet_squeeze_excite_block(x1)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    @staticmethod
    def densenet_dense_block_sw(x, blocks, name):
        for i in range(blocks):
            x = Model_Structure.densenet_conv_block_sw(x, 32, name=name + '_block' + str(i + 1))
        return x

    @staticmethod
    def densenet_transition_block_sw(x, reduction, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x = SwitchNormalization()(x)
        x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
        x = tf.keras.layers.Conv2D(
            int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            1,
            use_bias=False,
            name=name + '_conv')(
            x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    @staticmethod
    def densenet_conv_block_sw(x, growth_rate, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x1 = SwitchNormalization()(x)
        x1 = tf.keras.layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
            x1)
        x1 = SwitchNormalization()(x1)
        x1 = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    @staticmethod
    def densenet_dense_block_g(x, blocks, name):
        for i in range(blocks):
            x = Model_Structure.densenet_conv_block_g(x, 32, name=name + '_block' + str(i + 1))
        return x

    @staticmethod
    def densenet_transition_block_g(x, reduction, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
            x)
        x = tf.keras.layers.Activation('swish', name=name + '_relu')(x)
        x = tf.keras.layers.Conv2D(
            int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            1,
            use_bias=False,
            name=name + '_conv')(
            x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    @staticmethod
    def densenet_conv_block_g(x, growth_rate, name):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x1 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
            x)
        x1 = tf.keras.layers.Activation('swish', name=name + '_0_relu')(x1)
        x1 = tf.keras.layers.GaussianNoise(0.5)(x1)
        x1 = tf.keras.layers.Conv2D(
            4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
            x1)
        x1 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
            x1)
        x1 = tf.keras.layers.Activation('swish', name=name + '_1_relu')(x1)
        x1 = tf.keras.layers.GaussianNoise(0.5)(x1)
        x1 = tf.keras.layers.Conv2D(
            growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x


class Get_Model(object):
    # DenseNet
    @staticmethod
    def DenseNet(inputs, block, **kwargs):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
            x)
        x = tf.keras.layers.Activation('relu', name='conv1/relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        x = Model_Structure.densenet_dense_block(x, block[0], name='conv2')
        x = Model_Structure.densenet_transition_block(x, 0.5, name='pool2')
        x = Model_Structure.densenet_dense_block(x, block[1], name='conv3')
        x = Model_Structure.densenet_transition_block(x, 0.5, name='pool3')
        x = Model_Structure.densenet_dense_block(x, block[2], name='conv4')
        x = Model_Structure.densenet_transition_block(x, 0.5, name='pool4')
        x = Model_Structure.densenet_dense_block(x, block[3], name='conv5')

        x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = tf.keras.layers.Activation('relu', name='relu')(x)
        return x

    # DIY
    @staticmethod
    def DenseNet_G(inputs, block, **kwargs):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
            x)
        x = tf.keras.layers.Activation('swish', name='conv1/relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        x = Model_Structure.densenet_dense_block_g(x, block[0], name='conv2')
        x = Model_Structure.densenet_transition_block_g(x, 0.5, name='pool2')
        x = Model_Structure.densenet_dense_block_g(x, block[1], name='conv3')
        x = Model_Structure.densenet_transition_block_g(x, 0.5, name='pool3')
        x = Model_Structure.densenet_dense_block_g(x, block[2], name='conv4')
        x = Model_Structure.densenet_transition_block_g(x, 0.5, name='pool4')
        x = Model_Structure.densenet_dense_block_g(x, block[3], name='conv5')

        x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = tf.keras.layers.Activation('swish', name='relu')(x)
        return x

    # DIY
    @staticmethod
    def DenseNet_D(inputs, block, **kwargs):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
            x)
        x = tf.keras.layers.Activation('Mish_Activation', name='conv1/relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        x = Model_Structure.densenet_dense_block_d(x, block[0], name='conv2')
        x = Model_Structure.densenet_transition_block_d(x, 0.5, name='pool2')
        x = Model_Structure.densenet_dense_block_d(x, block[1], name='conv3')
        x = Model_Structure.densenet_transition_block_d(x, 0.5, name='pool3')
        x = Model_Structure.densenet_dense_block_d(x, block[2], name='conv4')
        x = Model_Structure.densenet_transition_block_d(x, 0.5, name='pool4')
        x = Model_Structure.densenet_dense_block_d(x, block[3], name='conv5')

        x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = tf.keras.layers.Activation('Mish_Activation', name='relu')(x)
        return x

    @staticmethod
    def DenseNet_SW(inputs, block, **kwargs):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
            x)
        x = tf.keras.layers.Activation('relu', name='conv1/relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)
        x = Model_Structure.densenet_dense_block_sw(x, block[0], name='conv2')
        x = Model_Structure.densenet_transition_block_sw(x, 0.5, name='pool2')
        x = Model_Structure.densenet_dense_block_sw(x, block[1], name='conv3')
        x = Model_Structure.densenet_transition_block_sw(x, 0.5, name='pool3')
        x = Model_Structure.densenet_dense_block_sw(x, block[2], name='conv4')
        x = Model_Structure.densenet_transition_block_sw(x, 0.5, name='pool4')
        x = Model_Structure.densenet_dense_block_sw(x, block[3], name='conv5')
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = tf.keras.layers.Activation('relu', name='relu')(x)
        return x

    # EfficientNet
    @staticmethod
    def EfficientNet(inputs,
                     width_coefficient,
                     depth_coefficient,
                     default_size=None,
                     dropout_rate=0.2,
                     drop_connect_rate=0.2,
                     depth_divisor=8,
                     activation='swish',
                     blocks_args='default', **kwargs):
        if blocks_args == 'default':
            blocks_args = [{{
                'kernel_size': 3,
                'repeats': 1,
                'filters_in': 32,
                'filters_out': 16,
                'expand_ratio': 1,
                'id_skip': True,
                'strides': 1,
                'se_ratio': 0.25
            }}, {{
                'kernel_size': 3,
                'repeats': 2,
                'filters_in': 16,
                'filters_out': 24,
                'expand_ratio': 6,
                'id_skip': True,
                'strides': 2,
                'se_ratio': 0.25
            }}, {{
                'kernel_size': 5,
                'repeats': 2,
                'filters_in': 24,
                'filters_out': 40,
                'expand_ratio': 6,
                'id_skip': True,
                'strides': 2,
                'se_ratio': 0.25
            }}, {{
                'kernel_size': 3,
                'repeats': 3,
                'filters_in': 40,
                'filters_out': 80,
                'expand_ratio': 6,
                'id_skip': True,
                'strides': 2,
                'se_ratio': 0.25
            }}, {{
                'kernel_size': 5,
                'repeats': 3,
                'filters_in': 80,
                'filters_out': 112,
                'expand_ratio': 6,
                'id_skip': True,
                'strides': 1,
                'se_ratio': 0.25
            }}, {{
                'kernel_size': 5,
                'repeats': 4,
                'filters_in': 112,
                'filters_out': 192,
                'expand_ratio': 6,
                'id_skip': True,
                'strides': 2,
                'se_ratio': 0.25
            }}, {{
                'kernel_size': 3,
                'repeats': 1,
                'filters_in': 192,
                'filters_out': 320,
                'expand_ratio': 6,
                'id_skip': True,
                'strides': 1,
                'se_ratio': 0.25
            }}]

        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        def round_filters(filters, divisor=depth_divisor):
            filters *= width_coefficient
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        # Build stem
        x = inputs
        x = tf.python.keras.layers.VersionAwareLayers().Rescaling(1. / 255.)(x)
        x = tf.python.keras.layers.VersionAwareLayers().Normalization(axis=bn_axis)(x)

        x = tf.keras.layers.ZeroPadding2D(
            padding=tf.python.keras.applications.imagenet_utils.correct_pad(x, 3),
            name='stem_conv_pad')(x)
        x = tf.keras.layers.Conv2D(
            round_filters(32),
            3,
            strides=2,
            padding='valid',
            use_bias=False,
            kernel_initializer={{
                'class_name': 'VarianceScaling',
                'config': {{
                    'scale': 2.0,
                    'mode': 'fan_out',
                    'distribution': 'truncated_normal'
                }}
            }},
            name='stem_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
        x = tf.keras.layers.Activation(activation, name='stem_activation')(x)

        blocks_args = copy.deepcopy(blocks_args)

        b = 0
        blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0

            args['filters_in'] = round_filters(args['filters_in'])
            args['filters_out'] = round_filters(args['filters_out'])

            for j in range(round_repeats(args.pop('repeats'))):

                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = Model_Structure.efficientnet_block(
                    x,
                    activation,
                    drop_connect_rate * b / blocks,
                    name='block{{}}{{}}_'.format(i + 1, chr(j + 97)),
                    **args)
                b += 1

        x = tf.keras.layers.Conv2D(
            round_filters(1280),
            1,
            padding='same',
            use_bias=False,
            kernel_initializer={{
                'class_name': 'VarianceScaling',
                'config': {{
                    'scale': 2.0,
                    'mode': 'fan_out',
                    'distribution': 'truncated_normal'
                }}
            }},
            name='top_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
        x = tf.keras.layers.Activation(activation, name='top_activation')(x)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
        return x

    # ResNet
    @staticmethod
    def ResNet(inputs, block, use_bias=True, preact=False, **kwargs):
        def stack_fn(x):
            x = Model_Structure.resnet_stack1(x, 64, block[0], stride1=1, name='conv2')
            x = Model_Structure.resnet_stack1(x, 128, block[1], name='conv3')
            x = Model_Structure.resnet_stack1(x, 256, block[2], name='conv4')
            return Model_Structure.resnet_stack1(x, 512, block[3], name='conv5')

        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

        if not preact:
            x = tf.keras.layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        x = stack_fn(x)
        return x

    # ResNet_V2
    @staticmethod
    def ResNetV2(inputs, block, use_bias=True, preact=True, **kwargs):
        def stack_fn(x):
            x = Model_Structure.resnet_stack2(x, 64, block[0], name='conv2')
            x = Model_Structure.resnet_stack2(x, 128, block[1], name='conv3')
            x = Model_Structure.resnet_stack2(x, 256, block[2], name='conv4')
            return Model_Structure.resnet_stack2(x, 512, block[3], stride1=1, name='conv5')

        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

        if not preact:
            x = tf.keras.layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        x = stack_fn(x)
        return x

    @staticmethod
    def ResNetV3(inputs, block, use_bias=True, preact=True, **kwargs):
        def stack_fn(x):
            x = Model_Structure.resnet_stack3(x, 64, block[0], name='conv2')
            x = Model_Structure.resnet_stack3(x, 128, block[1], name='conv3')
            x = Model_Structure.resnet_stack3(x, 256, block[2], name='conv4')
            return Model_Structure.resnet_stack3(x, 512, block[3], stride1=1, name='conv5')

        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

        if not preact:
            x = tf.keras.layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        x = stack_fn(x)
        return x

    # InceptionResNetV2
    @staticmethod
    def InceptionResNetV2(inputs, **kwargs):
        x = Model_Structure.inception_resnet_conv2d_bn(inputs, 32, 3, strides=2, padding='valid')
        x = Model_Structure.inception_resnet_conv2d_bn(x, 32, 3, padding='valid')
        x = Model_Structure.inception_resnet_conv2d_bn(x, 64, 3)
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
        x = Model_Structure.inception_resnet_conv2d_bn(x, 80, 1, padding='valid')
        x = Model_Structure.inception_resnet_conv2d_bn(x, 192, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

        # Mixed 5b (Inception-A block): 35 x 35 x 320
        branch_0 = Model_Structure.inception_resnet_conv2d_bn(x, 96, 1)
        branch_1 = Model_Structure.inception_resnet_conv2d_bn(x, 48, 1)
        branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 64, 5)
        branch_2 = Model_Structure.inception_resnet_conv2d_bn(x, 64, 1)
        branch_2 = Model_Structure.inception_resnet_conv2d_bn(branch_2, 96, 3)
        branch_2 = Model_Structure.inception_resnet_conv2d_bn(branch_2, 96, 3)
        branch_pool = tf.keras.layers.AveragePooling2D(3, strides=1, padding='same')(x)
        branch_pool = Model_Structure.inception_resnet_conv2d_bn(branch_pool, 64, 1)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else 3
        x = tf.keras.layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        for block_idx in range(1, 11):
            x = Model_Structure.inception_resnet_block(
                x, scale=0.17, block_type='block35', block_idx=block_idx)

        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        branch_0 = Model_Structure.inception_resnet_conv2d_bn(x, 384, 3, strides=2, padding='valid')
        branch_1 = Model_Structure.inception_resnet_conv2d_bn(x, 256, 1)
        branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 256, 3)
        branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
        branch_pool = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = tf.keras.layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        for block_idx in range(1, 21):
            x = Model_Structure.inception_resnet_block(
                x, scale=0.1, block_type='block17', block_idx=block_idx)

        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        branch_0 = Model_Structure.inception_resnet_conv2d_bn(x, 256, 1)
        branch_0 = Model_Structure.inception_resnet_conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
        branch_1 = Model_Structure.inception_resnet_conv2d_bn(x, 256, 1)
        branch_1 = Model_Structure.inception_resnet_conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
        branch_2 = Model_Structure.inception_resnet_conv2d_bn(x, 256, 1)
        branch_2 = Model_Structure.inception_resnet_conv2d_bn(branch_2, 288, 3)
        branch_2 = Model_Structure.inception_resnet_conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
        branch_pool = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = tf.keras.layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for block_idx in range(1, 10):
            x = Model_Structure.inception_resnet_block(
                x, scale=0.2, block_type='block8', block_idx=block_idx)
        x = Model_Structure.inception_resnet_block(
            x, scale=1., activation=None, block_type='block8', block_idx=10)

        # Final convolution block: 8 x 8 x 1536
        x = Model_Structure.inception_resnet_conv2d_bn(x, 1536, 1, name='conv_7b')
        return x

    # InceptionV3
    @staticmethod
    def InceptionV3(inputs, **kwargs):
        if tf.keras.backend.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        x = Model_Structure.inception_conv2d_bn(inputs, 32, 3, 3, strides=(2, 2), padding='valid')
        x = Model_Structure.inception_conv2d_bn(x, 32, 3, 3, padding='valid')
        x = Model_Structure.inception_conv2d_bn(x, 64, 3, 3)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = Model_Structure.inception_conv2d_bn(x, 80, 1, 1, padding='valid')
        x = Model_Structure.inception_conv2d_bn(x, 192, 3, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0: 35 x 35 x 256
        branch1x1 = Model_Structure.inception_conv2d_bn(x, 64, 1, 1)

        branch5x5 = Model_Structure.inception_conv2d_bn(x, 48, 1, 1)
        branch5x5 = Model_Structure.inception_conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = Model_Structure.inception_conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Model_Structure.inception_conv2d_bn(branch_pool, 32, 1, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                        axis=channel_axis,
                                        name='mixed0')

        # mixed 1: 35 x 35 x 288
        branch1x1 = Model_Structure.inception_conv2d_bn(x, 64, 1, 1)

        branch5x5 = Model_Structure.inception_conv2d_bn(x, 48, 1, 1)
        branch5x5 = Model_Structure.inception_conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = Model_Structure.inception_conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Model_Structure.inception_conv2d_bn(branch_pool, 64, 1, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                        axis=channel_axis,
                                        name='mixed1')

        # mixed 2: 35 x 35 x 288
        branch1x1 = Model_Structure.inception_conv2d_bn(x, 64, 1, 1)

        branch5x5 = Model_Structure.inception_conv2d_bn(x, 48, 1, 1)
        branch5x5 = Model_Structure.inception_conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = Model_Structure.inception_conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Model_Structure.inception_conv2d_bn(branch_pool, 64, 1, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                        axis=channel_axis,
                                        name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = Model_Structure.inception_conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = Model_Structure.inception_conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = Model_Structure.inception_conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                                        axis=channel_axis,
                                        name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = Model_Structure.inception_conv2d_bn(x, 192, 1, 1)

        branch7x7 = Model_Structure.inception_conv2d_bn(x, 128, 1, 1)
        branch7x7 = Model_Structure.inception_conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = Model_Structure.inception_conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = Model_Structure.inception_conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Model_Structure.inception_conv2d_bn(branch_pool, 192, 1, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                        axis=channel_axis,
                                        name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = Model_Structure.inception_conv2d_bn(x, 192, 1, 1)

            branch7x7 = Model_Structure.inception_conv2d_bn(x, 160, 1, 1)
            branch7x7 = Model_Structure.inception_conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = Model_Structure.inception_conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = Model_Structure.inception_conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                                           strides=(1, 1),
                                                           padding='same')(
                x)
            branch_pool = Model_Structure.inception_conv2d_bn(branch_pool, 192, 1, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                            axis=channel_axis,
                                            name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = Model_Structure.inception_conv2d_bn(x, 192, 1, 1)

        branch7x7 = Model_Structure.inception_conv2d_bn(x, 192, 1, 1)
        branch7x7 = Model_Structure.inception_conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = Model_Structure.inception_conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = Model_Structure.inception_conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = Model_Structure.inception_conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Model_Structure.inception_conv2d_bn(branch_pool, 192, 1, 1)
        x = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                        axis=channel_axis,
                                        name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = Model_Structure.inception_conv2d_bn(x, 192, 1, 1)
        branch3x3 = Model_Structure.inception_conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = Model_Structure.inception_conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = Model_Structure.inception_conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = Model_Structure.inception_conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = Model_Structure.inception_conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                                        axis=channel_axis,
                                        name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = Model_Structure.inception_conv2d_bn(x, 320, 1, 1)

            branch3x3 = Model_Structure.inception_conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = Model_Structure.inception_conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = Model_Structure.inception_conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = tf.keras.layers.concatenate([branch3x3_1, branch3x3_2],
                                                    axis=channel_axis,
                                                    name='mixed9_' + str(i))

            branch3x3dbl = Model_Structure.inception_conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = Model_Structure.inception_conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = Model_Structure.inception_conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = Model_Structure.inception_conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = tf.keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                                       axis=channel_axis)

            branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                                           strides=(1, 1),
                                                           padding='same')(
                x)
            branch_pool = Model_Structure.inception_conv2d_bn(branch_pool, 192, 1, 1)
            x = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                                            axis=channel_axis,
                                            name='mixed' + str(9 + i))
            return x

    # MobileNet
    @staticmethod
    def MobileNet(inputs,
                  alpha=1.0,
                  depth_multiplier=1, **kwargs):
        x = Model_Structure.mobilenet_conv_block(inputs, 32, alpha, strides=(2, 2))
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

        x = Model_Structure.mobilenet_depthwise_conv_block(
            x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

        x = Model_Structure.mobilenet_depthwise_conv_block(
            x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

        x = Model_Structure.mobilenet_depthwise_conv_block(
            x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

        x = Model_Structure.mobilenet_depthwise_conv_block(
            x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
        x = Model_Structure.mobilenet_depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
        return x

    # MobileNetV2
    @staticmethod
    def MobileNetV2(inputs,
                    alpha=1.0, **kwargs):
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        first_block_filters = Model_Structure.mobilenet_v2_make_divisible(32 * alpha, 8)
        x = tf.keras.layers.ZeroPadding2D(
            padding=tf.python.keras.applications.imagenet_utils.correct_pad(inputs, 3),
            name='Conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(
            first_block_filters,
            kernel_size=3,
            strides=(2, 2),
            padding='valid',
            use_bias=False,
            name='Conv1')(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(
            x)
        x = tf.keras.layers.ReLU(6., name='Conv1_relu')(x)

        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

        x = Model_Structure.mobilenet_v2_inverted_res_block(
            x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            last_block_filters = Model_Structure.mobilenet_v2_make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280

        x = tf.keras.layers.Conv2D(
            last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(
            x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(
            x)
        x = tf.keras.layers.ReLU(6., name='out_relu')(x)
        return x

    # Xception
    @staticmethod
    def Xception(inputs, **kwargs):
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        x = tf.keras.layers.Conv2D(
            32, (3, 3),
            strides=(2, 2),
            use_bias=False,
            name='block1_conv1')(inputs)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block1_conv1_act')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block1_conv2_act')(x)

        residual = tf.keras.layers.Conv2D(
            128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.SeparableConv2D(
            128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block2_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(
            128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3),
                                         strides=(2, 2),
                                         padding='same',
                                         name='block2_pool')(x)
        x = tf.keras.layers.add([x, residual])

        residual = tf.keras.layers.Conv2D(
            256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = tf.keras.layers.SeparableConv2D(
            256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block3_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(
            256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3),
                                         strides=(2, 2),
                                         padding='same',
                                         name='block3_pool')(x)
        x = tf.keras.layers.add([x, residual])

        residual = tf.keras.layers.Conv2D(
            728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = tf.keras.layers.SeparableConv2D(
            728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block4_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(
            728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3),
                                         strides=(2, 2),
                                         padding='same',
                                         name='block4_pool')(x)
        x = tf.keras.layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = tf.keras.layers.SeparableConv2D(
                728, (3, 3),
                padding='same',
                use_bias=False,
                name=prefix + '_sepconv1')(x)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
            x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = tf.keras.layers.SeparableConv2D(
                728, (3, 3),
                padding='same',
                use_bias=False,
                name=prefix + '_sepconv2')(x)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
            x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = tf.keras.layers.SeparableConv2D(
                728, (3, 3),
                padding='same',
                use_bias=False,
                name=prefix + '_sepconv3')(x)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

            x = tf.keras.layers.add([x, residual])

        residual = tf.keras.layers.Conv2D(
            1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = tf.keras.layers.SeparableConv2D(
            728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(
            1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3),
                                         strides=(2, 2),
                                         padding='same',
                                         name='block13_pool')(x)
        x = tf.keras.layers.add([x, residual])

        x = tf.keras.layers.SeparableConv2D(
            1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = tf.keras.layers.SeparableConv2D(
            2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block14_sepconv2_act')(x)
        return x

    # NASNetMobile
    @staticmethod
    def NASNetMobile(inputs, penultimate_filters=1056,
                     num_blocks=4,
                     stem_block_filters=32,
                     skip_reduction=False,
                     filter_multiplier=2, **kwargs):
        channel_dim = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        filters = penultimate_filters // 24

        x = tf.keras.layers.Conv2D(
            stem_block_filters, (3, 3),
            strides=(2, 2),
            padding='valid',
            use_bias=False,
            name='stem_conv1',
            kernel_initializer='he_normal')(
            inputs)

        x = tf.keras.layers.BatchNormalization(
            axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='stem_bn1')(
            x)

        p = None
        x, p = Model_Structure.nasnetmobile_reduction_a_cell(
            x, p, filters // (filter_multiplier ** 2), block_id='stem_1')
        x, p = Model_Structure.nasnetmobile_reduction_a_cell(
            x, p, filters // filter_multiplier, block_id='stem_2')

        for i in range(num_blocks):
            x, p = Model_Structure.nasnetmobile_normal_a_cell(x, p, filters, block_id='%d' % (i))

        x, p0 = Model_Structure.nasnetmobile_reduction_a_cell(
            x, p, filters * filter_multiplier, block_id='reduce_%d' % (num_blocks))

        p = p0 if not skip_reduction else p

        for i in range(num_blocks):
            x, p = Model_Structure.nasnetmobile_normal_a_cell(
                x, p, filters * filter_multiplier, block_id='%d' % (num_blocks + i + 1))

        x, p0 = Model_Structure.nasnetmobile_reduction_a_cell(
            x,
            p,
            filters * filter_multiplier ** 2,
            block_id='reduce_%d' % (2 * num_blocks))

        p = p0 if not skip_reduction else p

        for i in range(num_blocks):
            x, p = Model_Structure.nasnetmobile_normal_a_cell(
                x,
                p,
                filters * filter_multiplier ** 2,
                block_id='%d' % (2 * num_blocks + i + 1))

        x = tf.keras.layers.Activation('relu')(x)
        return x

    # NASNetLarge
    @staticmethod
    def NASNetLarge(inputs, penultimate_filters=4032,
                    num_blocks=6,
                    stem_block_filters=96,
                    skip_reduction=True,
                    filter_multiplier=2, **kwargs):
        channel_dim = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        filters = penultimate_filters // 24

        x = tf.keras.layers.Conv2D(
            stem_block_filters, (3, 3),
            strides=(2, 2),
            padding='valid',
            use_bias=False,
            name='stem_conv1',
            kernel_initializer='he_normal')(
            inputs)

        x = tf.keras.layers.BatchNormalization(
            axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='stem_bn1')(
            x)

        p = None
        x, p = Model_Structure.nasnetmobile_reduction_a_cell(
            x, p, filters // (filter_multiplier ** 2), block_id='stem_1')
        x, p = Model_Structure.nasnetmobile_reduction_a_cell(
            x, p, filters // filter_multiplier, block_id='stem_2')

        for i in range(num_blocks):
            x, p = Model_Structure.nasnetmobile_normal_a_cell(x, p, filters, block_id='%d' % (i))

        x, p0 = Model_Structure.nasnetmobile_reduction_a_cell(
            x, p, filters * filter_multiplier, block_id='reduce_%d' % (num_blocks))

        p = p0 if not skip_reduction else p

        for i in range(num_blocks):
            x, p = Model_Structure.nasnetmobile_normal_a_cell(
                x, p, filters * filter_multiplier, block_id='%d' % (num_blocks + i + 1))

        x, p0 = Model_Structure.nasnetmobile_reduction_a_cell(
            x,
            p,
            filters * filter_multiplier ** 2,
            block_id='reduce_%d' % (2 * num_blocks))

        p = p0 if not skip_reduction else p

        for i in range(num_blocks):
            x, p = Model_Structure.nasnetmobile_normal_a_cell(
                x,
                p,
                filters * filter_multiplier ** 2,
                block_id='%d' % (2 * num_blocks + i + 1))

        x = tf.keras.layers.Activation('relu')(x)
        return x

    # MobileNetV3Small
    @staticmethod
    def MobileNetV3Small(input_tensor, last_point_ch=1024, alpha=1.0,
                         minimalistic=False, model_type='small', weights=None, **kwargs):

        def hard_sigmoid(x):
            return tf.keras.layers.ReLU(6.)(x + 3.) * (1. / 6.)

        def relu(x):
            return tf.keras.layers.ReLU()(x)

        def stack_fn(x, kernel, activation, se_ratio):
            def depth(d):
                return Model_Structure.mobilenet_v3_depth(d * alpha)

            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)

            return x

        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        if minimalistic:
            kernel = 3
            activation = relu
            se_ratio = None
        else:
            kernel = 5
            activation = hard_sigmoid
            se_ratio = 0.25

        x = input_tensor
        x = tf.python.keras.layers.VersionAwareLayers().Rescaling(1. / 255.)(x)
        x = tf.keras.layers.Conv2D(
            16,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            use_bias=False,
            name='Conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3,
            momentum=0.999, name='Conv/BatchNorm')(x)
        x = activation(x)

        x = stack_fn(x, kernel, activation, se_ratio)

        last_conv_ch = Model_Structure.mobilenet_v3_depth(tf.keras.backend.int_shape(x)[channel_axis] * 6)

        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            last_point_ch = Model_Structure.mobilenet_v3_depth(last_point_ch * alpha)
        x = tf.keras.layers.Conv2D(
            last_conv_ch,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name='Conv_1')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3,
            momentum=0.999, name='Conv_1/BatchNorm')(x)
        x = activation(x)
        x = tf.keras.layers.Conv2D(
            last_point_ch,
            kernel_size=1,
            padding='same',
            use_bias=True,
            name='Conv_2')(x)
        x = activation(x)
        if weights == 'imagenet':
            model = tf.keras.Model(input_tensor, x, name='MobilenetV3' + model_type)
            model_name = '{{}}{{}}_224_{{}}_float'.format(
                model_type, '_minimalistic' if minimalistic else '', str(alpha))
            weights_hashes = {{
                'large_224_0.75_float': ('765b44a33ad4005b3ac83185abf1d0eb',
                                         'e7b4d1071996dd51a2c2ca2424570e20'),
                'large_224_1.0_float': ('59e551e166be033d707958cf9e29a6a7',
                                        '037116398e07f018c0005ffcb0406831'),
                'large_minimalistic_224_1.0_float': ('675e7b876c45c57e9e63e6d90a36599c',
                                                     'a2c33aed672524d1d0b4431808177695'),
                'small_224_0.75_float': ('cb65d4e5be93758266aa0a7f2c6708b7',
                                         '4d2fe46f1c1f38057392514b0df1d673'),
                'small_224_1.0_float': ('8768d4c2e7dee89b9d02b2d03d65d862',
                                        'be7100780f875c06bcab93d76641aa26'),
                'small_minimalistic_224_1.0_float': ('99cd97fb2fcdad2bf028eb838de69e37',
                                                     '20d4e357df3f7a6361f3a288857b1051'),
            }}
            file_name = 'weights_mobilenet_v3_' + model_name + '_no_top.h5'
            file_hash = weights_hashes[model_name][1]
            weights_path = tf.python.keras.utils.data_utils.get_file(
                file_name,
                ('https://storage.googleapis.com/tensorflow/'
                 'keras-applications/mobilenet_v3/') + file_name,
                cache_subdir='models',
                file_hash=file_hash)
            model.load_weights(weights_path)
            return model
        else:
            return x

    # MobileNetV3Large
    @staticmethod
    def MobileNetV3Large(input_tensor, last_point_ch=1280, alpha=1.0,
                         minimalistic=False, model_type='large', weights=None, **kwargs):
        def hard_sigmoid(x):
            return tf.keras.layers.ReLU(6.)(x + 3.) * (1. / 6.)

        def relu(x):
            return tf.keras.layers.ReLU()(x)

        def stack_fn(x, kernel, activation, se_ratio):
            def depth(d):
                return Model_Structure.mobilenet_v3_depth(d * alpha)

            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation,
                                                                12)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation,
                                                                13)
            x = Model_Structure.mobilenet_v3_inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation,
                                                                14)
            return x

        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        if minimalistic:
            kernel = 3
            activation = relu
            se_ratio = None
        else:
            kernel = 5
            activation = hard_sigmoid
            se_ratio = 0.25

        x = input_tensor
        x = tf.python.keras.layers.VersionAwareLayers().Rescaling(1. / 255.)(x)
        x = tf.keras.layers.Conv2D(
            16,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            use_bias=False,
            name='Conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3,
            momentum=0.999, name='Conv/BatchNorm')(x)
        x = activation(x)

        x = stack_fn(x, kernel, activation, se_ratio)

        last_conv_ch = Model_Structure.mobilenet_v3_depth(tf.keras.backend.int_shape(x)[channel_axis] * 6)

        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            last_point_ch = Model_Structure.mobilenet_v3_depth(last_point_ch * alpha)
        x = tf.keras.layers.Conv2D(
            last_conv_ch,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name='Conv_1')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3,
            momentum=0.999, name='Conv_1/BatchNorm')(x)
        x = activation(x)
        x = tf.keras.layers.Conv2D(
            last_point_ch,
            kernel_size=1,
            padding='same',
            use_bias=True,
            name='Conv_2')(x)
        x = activation(x)

        model = tf.keras.models.Model(input_tensor, x, name='MobilenetV3' + model_type)

        # Load weights.
        if weights == 'imagenet':
            model_name = '{{}}{{}}_224_{{}}_float'.format(
                model_type, '_minimalistic' if minimalistic else '', str(alpha))

            weights_hashes = {{
                'large_224_0.75_float': ('765b44a33ad4005b3ac83185abf1d0eb',
                                         'e7b4d1071996dd51a2c2ca2424570e20'),
                'large_224_1.0_float': ('59e551e166be033d707958cf9e29a6a7',
                                        '037116398e07f018c0005ffcb0406831'),
                'large_minimalistic_224_1.0_float': ('675e7b876c45c57e9e63e6d90a36599c',
                                                     'a2c33aed672524d1d0b4431808177695'),
                'small_224_0.75_float': ('cb65d4e5be93758266aa0a7f2c6708b7',
                                         '4d2fe46f1c1f38057392514b0df1d673'),
                'small_224_1.0_float': ('8768d4c2e7dee89b9d02b2d03d65d862',
                                        'be7100780f875c06bcab93d76641aa26'),
                'small_minimalistic_224_1.0_float': ('99cd97fb2fcdad2bf028eb838de69e37',
                                                     '20d4e357df3f7a6361f3a288857b1051'),
            }}
            file_name = 'weights_mobilenet_v3_' + model_name + '_no_top.h5'
            file_hash = weights_hashes[model_name][1]
            weights_path = tf.python.keras.utils.data_utils.get_file(
                file_name,
                ('https://storage.googleapis.com/tensorflow/'
                 'keras-applications/mobilenet_v3/') + file_name,
                cache_subdir='models',
                file_hash=file_hash)
            model.load_weights(weights_path)
            return model
        else:
            return x

    # MnasNet
    @staticmethod
    def MnasNet(inputs, alpha=1, **kwargs):
        x = Model_Structure.mnasnet_conv_bn(inputs, 32 * alpha, 3, strides=2)
        x = Model_Structure.mnasnet_depthwiseConv_bn(x, 16 * alpha, 3, strides=1)
        # MBConv3 3x3
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=24, kernel_size=3, strides=2, filters_multiplier=3,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=24, kernel_size=3, strides=1, filters_multiplier=3,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=24, kernel_size=3, strides=1, filters_multiplier=3,
                                                  alpha=alpha)
        # MBConv3 5x5
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=40, kernel_size=5, strides=2, filters_multiplier=3,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=40, kernel_size=5, strides=1, filters_multiplier=3,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=40, kernel_size=5, strides=1, filters_multiplier=3,
                                                  alpha=alpha)
        # MBConv6 5x5
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=80, kernel_size=5, strides=2, filters_multiplier=6,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=80, kernel_size=5, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=80, kernel_size=5, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        # MBConv6 3x3
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=96, kernel_size=3, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=96, kernel_size=3, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        # MBConv6 5x5
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=192, kernel_size=5, strides=2, filters_multiplier=6,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        # MBConv6 3x3
        x = Model_Structure.mnasnet_mbconv_idskip(x, filters=320, kernel_size=3, strides=1, filters_multiplier=6,
                                                  alpha=alpha)
        # FC + POOL
        x = Model_Structure.mnasnet_conv_bn(x, filters=1152 * alpha, kernel_size=1, strides=1)
        return x

    # SqueezeNet
    @staticmethod
    def SqueezeNet(inputs, **kwargs):
        x = tf.keras.layers.Conv2D(filters=96,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = Model_Structure.squeezenet_firemodule(x, s1=16, e1=64, e3=64)
        x = Model_Structure.squeezenet_firemodule(x, s1=16, e1=64, e3=64)
        x = Model_Structure.squeezenet_firemodule(x, s1=32, e1=128, e3=128)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = Model_Structure.squeezenet_firemodule(x, s1=32, e1=128, e3=128)
        x = Model_Structure.squeezenet_firemodule(x, s1=48, e1=192, e3=192)
        x = Model_Structure.squeezenet_firemodule(x, s1=48, e1=192, e3=192)
        x = Model_Structure.squeezenet_firemodule(x, s1=64, e1=256, e3=256)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = Model_Structure.squeezenet_firemodule(x, s1=64, e1=256, e3=256)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Conv2D(filters=Settings.settings(),
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")(x)
        return x

    # ShuffleNetV2
    @staticmethod
    def ShuffleNetV2(inputs, channel_scale, training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=2, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.nn.swish(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)
        x = Model_Structure.shufflenet_v2_make_layer(x, repeat_num=4, in_channels=24, out_channels=channel_scale[0])
        x = Model_Structure.shufflenet_v2_make_layer(x, repeat_num=8, in_channels=channel_scale[0],
                                                     out_channels=channel_scale[1])
        x = Model_Structure.shufflenet_v2_make_layer(x, repeat_num=4, in_channels=channel_scale[1],
                                                     out_channels=channel_scale[2])
        x = tf.keras.layers.Conv2D(filters=channel_scale[3], kernel_size=(1, 1), strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x, training)
        return x

    # SEResNet
    @staticmethod
    def SEResNet(inputs, block, training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)(x)
        x = Model_Structure.seresnet_make_res_block(x, filter_num=64,
                                                    blocks=block[0])
        x = Model_Structure.seresnet_make_res_block(x, filter_num=128,
                                                    blocks=block[1],
                                                    stride=2)
        x = Model_Structure.seresnet_make_res_block(x, filter_num=256,
                                                    blocks=block[2],
                                                    stride=2)
        x = Model_Structure.seresnet_make_res_block(x, filter_num=512,
                                                    blocks=block[3],
                                                    stride=2)
        return x

    # ResNeXt
    @staticmethod
    def ResNeXt(inputs, block, cardinality=32, training=None, **kwargs):
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding="same")(x)
        x = Model_Structure.resnext_build_ResNeXt_block(x, filters=128,
                                                        strides=1,
                                                        groups=cardinality,
                                                        repeat_num=block[0])
        x = Model_Structure.resnext_build_ResNeXt_block(x, filters=256,
                                                        strides=2,
                                                        groups=cardinality,
                                                        repeat_num=block[1])
        x = Model_Structure.resnext_build_ResNeXt_block(x, filters=512,
                                                        strides=2,
                                                        groups=cardinality,
                                                        repeat_num=block[2])
        x = Model_Structure.resnext_build_ResNeXt_block(x, filters=1024,
                                                        strides=2,
                                                        groups=cardinality,
                                                        repeat_num=block[3])
        return x

    # RegNet
    @staticmethod
    def RegNet(inputs, active='relu', dropout_rate=0.2, fc_activation=None, stem_set=48, stage_depth=[2, 6, 17, 2],
               stage_width=[48, 120, 336, 888], stage_G=24, SEstyle_atten="SE", using_cb=False, **kwargs):
        x = Model_Structure.regnet_make_stem(inputs, filters=stem_set, size=(3, 3), strides=2, active=active)
        for i in range(len(stage_depth)):
            depth = stage_depth[i]
            width = stage_width[i]
            group_G = stage_G
            x = Model_Structure.regnet_make_stage(x, n_block=depth,
                                                  block_width=width,
                                                  group_G=group_G)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, noise_shape=None)(x)
        if fc_activation:
            x = tf.keras.layers.Activation(fc_activation)(x)
        return x

    # ResNest
    @staticmethod
    def ResNest(inputs, dropout_rate=0.2, fc_activation=None, blocks_set=[3, 4, 6, 3], radix=2, groups=1,
                bottleneck_width=64, deep_stem=True, stem_width=32, block_expansion=4, avg_down=True, avd=True,
                avd_first=False, preact=False, using_basic_block=False, using_cb=False, using_transformer=True,
                hidden_dim=512, nheads=8, num_encoder_layers=6, num_decoder_layers=6, n_query_pos=100, **kwargs):
        x = Model_Structure.resnest_make_stem(inputs, stem_width=stem_width, deep_stem=deep_stem)

        if preact is False:
            x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation('Mish_Activation')(x)

        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", data_format="channels_last")(x)

        if preact is True:
            x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation('Mish_Activation')(x)

        if using_cb:
            second_x = x
            second_x = Model_Structure.resnest_make_layer(x, blocks=blocks_set[0], filters=64, stride=1, is_first=False,
                                                          using_basic_block=using_basic_block, avd=avd, radix=radix,
                                                          avd_first=avd_first)
            second_x_tmp = Model_Structure.resnest_make_composite_layer(second_x, filters=x.shape[-1], upsample=False)

            x = tf.keras.layers.Add()([second_x_tmp, x])
        x = Model_Structure.resnest_make_layer(x, blocks=blocks_set[0], filters=64, stride=1, is_first=False,
                                               using_basic_block=using_basic_block, avd=avd, radix=radix,
                                               avd_first=avd_first)

        b1_b3_filters = [64, 128, 256, 512]
        for i in range(3):
            idx = i + 1
            if using_cb:
                second_x = Model_Structure.resnest_make_layer(x, blocks=blocks_set[idx], filters=b1_b3_filters[idx],
                                                              stride=2,
                                                              using_basic_block=using_basic_block, avd=avd, radix=radix,
                                                              avd_first=avd_first)
                second_x_tmp = Model_Structure.resnest_make_composite_layer(second_x, filters=x.shape[-1])

                x = tf.keras.layers.Add()([second_x_tmp, x])
            x = Model_Structure.resnest_make_layer(x, blocks=blocks_set[idx], filters=b1_b3_filters[idx], stride=2,
                                                   is_first=False,
                                                   using_basic_block=using_basic_block, avd=avd, radix=radix,
                                                   avd_first=avd_first)

        if using_transformer:
            x = Model_Structure.resnest_make_transformer_top(x, hidden_dim=hidden_dim, n_query_pos=n_query_pos,
                                                             nheads=nheads,
                                                             num_encoder_layers=num_encoder_layers,
                                                             num_decoder_layers=num_decoder_layers)

        else:
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, noise_shape=None)(x)

        if fc_activation:
            x = tf.keras.layers.Activation('Mish_Activation')(x)
        if using_transformer:
            x = tf.expand_dims(x, axis=1)

        return x

    # GhostNet
    @staticmethod
    def GhostNet(inputs, **kwargs):
        x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation=None, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = GBNeck(dwkernel=3, strides=1, exp=16, out=16, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=3, strides=2, exp=48, out=24, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=3, strides=1, exp=72, out=24, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=5, strides=2, exp=72, out=40, ratio=2, use_se=True)(x)
        x = GBNeck(dwkernel=5, strides=1, exp=120, out=40, ratio=2, use_se=True)(x)
        x = GBNeck(dwkernel=3, strides=2, exp=240, out=80, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=3, strides=1, exp=200, out=80, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=3, strides=1, exp=184, out=80, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=3, strides=1, exp=184, out=80, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=3, strides=1, exp=480, out=112, ratio=2, use_se=True)(x)
        x = GBNeck(dwkernel=3, strides=1, exp=672, out=112, ratio=2, use_se=True)(x)
        x = GBNeck(dwkernel=5, strides=2, exp=672, out=160, ratio=2, use_se=True)(x)
        x = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=True)(x)
        x = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=False)(x)
        x = GBNeck(dwkernel=5, strides=1, exp=960, out=160, ratio=2, use_se=True)(x)
        x = tf.keras.layers.Conv2D(960, (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                                   activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(1280, (1, 1), strides=(1, 1), padding='same', activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    # SE_DenseNet
    @staticmethod
    def SE_DenseNet(inputs, block, **kwargs):
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
            x)
        x = tf.keras.layers.Activation('Mish_Activation', name='conv1/relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        x = Model_Structure.sedensenet_dense_block(x, block[0], name='conv2')
        x = Model_Structure.sedensenet_transition_block(x, 0.5, name='pool2')
        x = Model_Structure.sedensenet_dense_block(x, block[1], name='conv3')
        x = Model_Structure.sedensenet_transition_block(x, 0.5, name='pool3')
        x = Model_Structure.sedensenet_dense_block(x, block[2], name='conv4')
        x = Model_Structure.sedensenet_transition_block(x, 0.5, name='pool4')
        x = Model_Structure.sedensenet_dense_block(x, block[3], name='conv5')

        x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = tf.keras.layers.Activation('Mish_Activation', name='relu')(x)
        return x

    # HS_ResNet_V2
    @staticmethod
    def HS_ResNetV2(inputs, block, use_bias=True, preact=True, **kwargs):
        def stack_fn(x):
            x = Model_Structure.hs_resnetv2_stack(x, 60, block[0], name='conv2')
            x = Model_Structure.hs_resnetv2_stack(x, 120, block[1], name='conv3')
            x = Model_Structure.hs_resnetv2_stack(x, 250, block[2], name='conv4')
            return Model_Structure.hs_resnetv2_stack(x, 500, block[3], stride1=1, name='conv5')

        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        x = tf.keras.layers.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

        if not preact:
            x = tf.keras.layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        x = stack_fn(x)
        return x


class Models(object):
    @staticmethod
    def captcha_model_yolo_tiny():
        anchors = YOLO_anchors.get_anchors()
        model_body = Yolo_tiny_model.yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), len(anchors) // 2,
                                               Settings.settings_num_classes())
        weights_path = os.path.join(WEIGHT, 'yolov4_tiny_weights_voc.h5')
        if os.path.exists(weights_path):
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            for i in range(int(len(list(model_body.layers)) * 0.9)): model_body.layers[i].trainable = False
            logger.success('有预训练权重')
        else:
            logger.error('没有权重')
        y_true = [tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT // {{0: 32, 1: 16}}[l], IMAGE_WIDTH // {{0: 32, 1: 16}}[l], len(anchors) // 2,
                   Settings.settings_num_classes() + 5)) for
            l in
            range(2)]
        loss_input = [*model_body.output, *y_true]

        model_loss = tf.keras.layers.Lambda(Yolo_Loss.yolo_loss, output_shape=(1,), name='yolo_loss',
                                            arguments={{'anchors': anchors,
                                                       'num_classes': Settings.settings_num_classes(),
                                                       'ignore_thresh': 0.5,
                                                       'label_smoothing': LABEL_SMOOTHING}})(loss_input)
        model = tf.keras.Model([model_body.input, *y_true], model_loss)
        model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, epsilon=1e-14, rectify=False),
                      loss={{'yolo_loss': lambda y_true, y_pred: y_pred}})
        return model

    @staticmethod
    def captcha_model_yolo():
        anchors = YOLO_anchors.get_anchors()
        model_body = Yolo_model.yolo_body(tf.keras.layers.Input(shape=(None, None, 3)), len(anchors) // 3,
                                          Settings.settings_num_classes())

        weights_path = os.path.join(WEIGHT, 'yolo4_weight.h5')
        if os.path.exists(weights_path):
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            for i in range(int(len(list(model_body.layers)) * 0.9)): model_body.layers[i].trainable = False
            logger.success('有预训练权重')
        else:
            logger.error('没有权重')
        y_true = [tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT // {{0: 32, 1: 16, 2: 8}}[l], IMAGE_WIDTH // {{0: 32, 1: 16, 2: 8}}[l], len(anchors) // 3,
                   Settings.settings_num_classes() + 5)) for
            l in
            range(3)]
        loss_input = [*model_body.output, *y_true]

        model_loss = tf.keras.layers.Lambda(Yolo_Loss.yolo_loss, output_shape=(1,), name='yolo_loss',
                                            arguments={{'anchors': anchors,
                                                       'num_classes': Settings.settings_num_classes(),
                                                       'ignore_thresh': 0.5,
                                                       'label_smoothing': LABEL_SMOOTHING}})(loss_input)
        model = tf.keras.Model([model_body.input, *y_true], model_loss)
        model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss={{'yolo_loss': lambda y_true, y_pred: y_pred}})
        return model

    @staticmethod
    def captcha_model_efficientdet():
        input_size = IMAGE_SIZES[PHI]
        input_shape = (input_size, input_size, 3)
        inputs = tf.keras.layers.Input(input_shape)

        fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        backbones = [(1.0, 1.0, 224, 0.2, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs),
                     (1.0, 1.0, 240, 0.2, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs),
                     (1.1, 1.2, 260, 0.3, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs),
                     (1.2, 1.4, 300, 0.3, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs),
                     (1.4, 1.8, 380, 0.4, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs),
                     (1.6, 2.2, 456, 0.4, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs),
                     (1.8, 2.6, 528, 0.5, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs),
                     (2.0, 3.1, 600, 0.5, 0.2, 8, DEFAULT_BLOCKS_ARGS, inputs)]

        x = Efficientdet(*backbones[PHI])
        # x = GhostNet_det.ghostnet(inputs)

        if PHI < 6:
            for i in range(fpn_cell_repeats[PHI]):
                x = Efficientdet_anchors.build_wBiFPN(x, fpn_num_filters[PHI], i)
        else:

            for i in range(fpn_cell_repeats[PHI]):
                x = Efficientdet_anchors.build_BiFPN(x, fpn_num_filters[PHI], i)

        # x = GhostNet_det.ghostdet(x)

        box_net = BoxNet(fpn_num_filters[PHI], box_class_repeats[PHI],
                         num_anchors=9, name='box_net')
        class_net = ClassNet(fpn_num_filters[PHI], box_class_repeats[PHI], num_classes=Settings.settings_num_classes(),
                             num_anchors=9, name='class_net')
        classification = [class_net.call([feature, i]) for i, feature in enumerate(x)]
        classification = tf.keras.layers.Concatenate(axis=1, name='classification')(classification)
        regression = [box_net.call([feature, i]) for i, feature in enumerate(x)]
        regression = tf.keras.layers.Concatenate(axis=1, name='regression')(regression)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[regression, classification], name='efficientdet')

        weights_path = os.path.join(WEIGHT, 'efficientdet-d0-voc.h5')
        if os.path.exists(weights_path):
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            for i in range(int(len(list(model.layers)) * 0.9)): model.layers[i].trainable = False
            logger.success('有预训练权重')
        else:
            logger.error('没有权重')
        model.compile(loss={{'regression': Efficientdet_Loss.smooth_l1(), 'classification': Efficientdet_Loss.focal()}},
                      optimizer=AdaBeliefOptimizer(learning_rate=LR, epsilon=1e-14, rectify=False))
        return model

    @staticmethod
    def captcha_model():
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = Get_Model.SE_DenseNet(inputs, block=[6, 12, 32, 32])
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=CAPTCHA_LENGTH * Settings.settings(),
                                        activation=tf.keras.activations.softmax)(x)
        outputs = tf.keras.layers.Reshape((CAPTCHA_LENGTH, Settings.settings()))(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                                   weight_decay=1e-2, rectify=False),
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                      metrics=['acc'])
        return model

    @staticmethod
    def captcha_model_num_classes():
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = tf.keras.applications.DenseNet169(input_tensor=inputs, include_top=False, weights='imagenet')
        x = x.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(units=Settings.settings_num_classes(),
                                        activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        for i in range(int(len(list(model.layers)) * 0.9)): model.layers[i].trainable = False
        model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                                   weight_decay=1e-2, rectify=False),
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                      metrics=['acc'])
        return model

    @staticmethod
    def captcha_model_ctc():
        inputs = tf.keras.layers.Input(shape=inputs_shape)
        x = Get_Model.MobileNetV3Small(inputs)
        filters = x.get_shape().as_list()[-1]
        x = tf.keras.layers.Reshape((-1, filters))(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(
            x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(
            x)
        outputs = tf.keras.layers.Dense(units=Settings.settings())(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                                   weight_decay=1e-2, rectify=False),
                      loss=CTCLoss(), metrics=[WordAccuracy()])
        return model

    @staticmethod
    def captcha_model_ctc_tiny(training=False):
        inputs = tf.keras.layers.Input(shape=inputs_shape, name='inputs')
        x = Get_Model.DenseNet(inputs, block=[6, 12, 32, 32])
        x = tf.keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]), name='reshape_len')(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Dense(Settings.settings(), activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs, x)
        labels = tf.keras.layers.Input(shape=(CAPTCHA_LENGTH), name='label')
        input_len = tf.keras.layers.Input(shape=(1), name='input_len')
        label_len = tf.keras.layers.Input(shape=(1), name='label_len')
        ctc_out = tf.keras.layers.Lambda(ctc_lambda_func, name='ctc')([x, labels, input_len, label_len])
        ctc_model = tf.keras.Model(inputs=[inputs, labels, input_len, label_len], outputs=ctc_out)
        if training:
            ctc_model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                                           weight_decay=1e-2, rectify=False),
                              loss={{'ctc': lambda y_true, y_pred: y_pred}})
            return ctc_model
        else:
            model.compile(optimizer=AdaBeliefOptimizer(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                                       weight_decay=1e-2, rectify=False),
                          loss={{'ctc': lambda y_true, y_pred: y_pred}})
            return model

    @staticmethod
    def captcha_yolov3(training=False):
        anchors = YOLO_anchors.get_anchors() / 416
        masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        x = inputs = tf.keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x_36, x_61, x = Yolov3_block.darknet(x)
        x = Yolov3_block.yoloconv(512)(x)
        output_0 = Yolov3_block.yolo_output(512, len(masks[0]), Settings.settings_num_classes())(x)
        x = Yolov3_block.yoloconv(256)((x, x_61))
        output_1 = Yolov3_block.yolo_output(256, len(masks[1]), Settings.settings_num_classes())(x)
        x = Yolov3_block.yoloconv(128)((x, x_36))
        output_2 = Yolov3_block.yolo_output(128, len(masks[2]), Settings.settings_num_classes())(x)

        if training:
            return tf.keras.Model(inputs, (output_0, output_1, output_2))

        boxes_0 = tf.keras.layers.Lambda(
            lambda x: Yolov3_losses.yolo_boxes(x, anchors[masks[0]], Settings.settings_num_classes()))(output_0)
        boxes_1 = tf.keras.layers.Lambda(
            lambda x: Yolov3_losses.yolo_boxes(x, anchors[masks[1]], Settings.settings_num_classes()))(output_1)
        boxes_2 = tf.keras.layers.Lambda(
            lambda x: Yolov3_losses.yolo_boxes(x, anchors[masks[2]], Settings.settings_num_classes()))(output_2)
        outputs = tf.keras.layers.Lambda(
            lambda x: Yolov3_losses.yolo_nms(x, anchors, masks, Settings.settings_num_classes()))(
            (boxes_0[:3], boxes_1[:3], boxes_2[:3]))
        return tf.keras.Model(inputs, outputs)

    @staticmethod
    def captcha_yolov3tiny(anchors, training=False):
        anchors = YOLO_anchors.get_anchors() / 416
        masks = np.array([[3, 4, 5], [0, 1, 2]])
        x = inputs = tf.keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x_8, x = Yolov3_block.darknettiny(x)
        x = Yolov3_block.yoloconv(256)(x)
        output_0 = Yolov3_block.yolo_output(256, len(masks[0]), Settings.settings_num_classes())(x)
        x = Yolov3_block.yoloconv(128)((x, x_8))
        output_1 = Yolov3_block.yolo_output(128, len(masks[1]), Settings.settings_num_classes())(x)

        if training:
            return tf.keras.Model(inputs, (output_0, output_1))

        boxes_0 = tf.keras.layers.Lambda(
            lambda x: Yolov3_losses.yolo_boxes(x, anchors[masks[0]], Settings.settings_num_classes()))(output_0)
        boxes_1 = tf.keras.layers.Lambda(
            lambda x: Yolov3_losses.yolo_boxes(x, anchors[masks[1]], Settings.settings_num_classes()))(output_1)
        outputs = tf.keras.layers.Lambda(
            lambda x: Yolov3_losses.yolo_nms(x, anchors, masks, Settings.settings_num_classes()))(
            (boxes_0[:3], boxes_1[:3]))
        return tf.keras.Model(inputs, outputs)


## big(适合使用GPU训练)
# InceptionResNetV2
# x = Get_Model.InceptionResNetV2(inputs)

# Densenet_121
# x = Get_Model.DenseNet(inputs, block=[6, 12, 24, 16])
# Densenet_169
# x = Get_Model.DenseNet(inputs, block=[6, 12, 32, 32])
# Densenet_201
# x = Get_Model.DenseNet(inputs, block=[6, 12, 48, 32])
# Densenet_264
# x = Get_Model.DenseNet(inputs, block=[6, 12, 64, 48])

# Efficient_net_b0
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
# Efficient_net_b1
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2)
# Efficient_net_b2
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.3)
# Efficient_net_b3
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.3)
# Efficient_net_b4
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.4)
# Efficient_net_b5
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.4)
# Efficient_net_b6
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=0.5)
# Efficient_net_b7
# x = Get_Model.EfficientNet(inputs, width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.5)

# ResNest
# x = Get_Model.ResNest(inputs)

# RegNet
# x = Get_Model.RegNet(inputs, active='Mish_Activation')

# Resnet_18
# x = Get_Model.ResNet(inputs, block=[2, 2, 2, 2])
# Resnet_50
# x = Get_Model.ResNet(inputs, block=[3, 4, 6, 3])
# Resnet_101
# x = Get_Model.ResNet(inputs, block=[3, 4, 23, 3])
# Resnet_152
# x = Get_Model.ResNet(inputs, block=[3, 8, 36, 3])

# ResNeXt50
# x = Get_Model.ResNeXt(inputs, block=[3, 4, 6, 3])
# ResNeXt101
# x = Get_Model.ResNeXt(inputs, block=[3, 4, 23, 3])

# SEResNet50
# x = Get_Model.SEResNet(inputs, block=[3, 4, 6, 3])
# SEResNet152
# x = Get_Model.SEResNet(inputs, block=[3, 8, 36, 3])


## small(适合使用CPU训练)

# MobileNetV2
# x = Get_Model.MobileNetV2(inputs)
# MobileNetV3Large
# x = Get_Model.MobileNetV3Large(inputs)
# MobileNetV3Small
# x = Get_Model.MobileNetV3Small(inputs)

# ShuffleNet_0_5x
# x = Get_Model.ShuffleNetV2(inputs, channel_scale=[48, 96, 192, 1024])
# ShuffleNet_1_0x
# x = Get_Model.ShuffleNetV2(inputs, channel_scale=[116, 232, 464, 1024])
# ShuffleNet_1_5x
# x = Get_Model.ShuffleNetV2(inputs, channel_scale=[176, 352, 704, 1024])
# ShuffleNet_2_0x
# x = Get_Model.ShuffleNetV2(inputs, channel_scale=[244, 488, 976, 2048])

# Efficient_net_b0
# x = Get_Model.EfficientNet(inputs, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, lite=False)

# SqueezeNet
# x = Get_Model.SqueezeNet(inputs)

# MnasNet
# x = Get_Model.MnasNet(inputs)

# GhostNet
# x = Get_Model.GhostNet(inputs)


## tf.keras(加载imagenet权重，冻结部分层的权重可以加速训练并获得不错的效果)
## tf2.3版本可用的模型
# x = tf.keras.applications.MobileNet(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.NASNetLarge(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.NASNetMobile(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.ResNet50(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.ResNet50V2(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.ResNet101(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.ResNet101V2(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.ResNet152(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.ResNet152V2(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.DenseNet121(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.DenseNet169(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.DenseNet201(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB0(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB1(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB2(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB3(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB4(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB5(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB6(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.EfficientNetB7(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.Xception(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.InceptionResNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
# x = tf.keras.applications.InceptionV3(input_tensor=inputs, include_top=False, weights='imagenet')
# x = Get_Model.MobileNetV3Small(input_tensor=inputs, weights='imagenet')
# x = Get_Model.MobileNetV3Large(input_tensor=inputs, weights='imagenet')

## 冻结部分层的权重代码(模型编译前冻结即可)
# for i in range(int(len(list(model.layers)) * 0.9)): model.layers[i].trainable = False


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        model = Models.captcha_model()
        model.summary()
        for i, n in enumerate(model.layers):
            logger.debug(f'{{i}} {{n.name}}')
        # model._layers = [layer for layer in model.layers if not isinstance(layer, dict)]
        # tf.keras.utils.plot_model(model, show_shapes=True, dpi=48, to_file='model.png')

"""


def split_dataset(work_path, project_name):
    return f"""import random
from {work_path}.{project_name}.settings import TRAIN_PATH
from {work_path}.{project_name}.utils import Image_Processing

train_image = Image_Processing.extraction_image(TRAIN_PATH)
random.shuffle(train_image)
Image_Processing.split_dataset(train_image)
"""


def pack_dataset(work_path, project_name):
    return f"""import random
from loguru import logger
from {work_path}.{project_name}.settings import TRAIN_PATH
from {work_path}.{project_name}.settings import VALIDATION_PATH
from {work_path}.{project_name}.settings import TEST_PATH
from {work_path}.{project_name}.settings import DATA_ENHANCEMENT
from {work_path}.{project_name}.settings import TFRECORD_TRAIN_PATH
from {work_path}.{project_name}.settings import TFRECORD_VALIDATION_PATH
from {work_path}.{project_name}.utils import Image_Processing
from {work_path}.{project_name}.utils import WriteTFRecord
from concurrent.futures import ThreadPoolExecutor

if DATA_ENHANCEMENT:
    image_path = Image_Processing.extraction_image(TRAIN_PATH)
    number = len(image_path)
    with ThreadPoolExecutor(max_workers=100) as t:
        for i in image_path:
            number = number - 1
            task = t.submit(Image_Processing.preprosess_save_images, i, number)

train_image = Image_Processing.extraction_image(TRAIN_PATH)
random.shuffle(train_image)
validation_image = Image_Processing.extraction_image(VALIDATION_PATH)
test_image = Image_Processing.extraction_image(TEST_PATH)

Image_Processing.extraction_label(train_image + validation_image + test_image)

train_lable = Image_Processing.extraction_label(train_image)
validation_lable = Image_Processing.extraction_label(validation_image)
test_lable = Image_Processing.extraction_label(test_image)
# logger.debug(train_image)
# logger.debug(train_lable)
#
with ThreadPoolExecutor(max_workers=3) as t:
    t.submit(WriteTFRecord.WriteTFRecord, TFRECORD_TRAIN_PATH, train_image, train_lable, 'train', 10000)
    t.submit(WriteTFRecord.WriteTFRecord, TFRECORD_VALIDATION_PATH, validation_image, validation_lable, 'validation',
             10000)

"""


def save_model(work_path, project_name):
    return f"""import os
import operator
import tensorflow as tf
from loguru import logger
from {work_path}.{project_name}.models import Models
from {work_path}.{project_name}.callback import CallBack
from {work_path}.{project_name}.settings import MODEL
from {work_path}.{project_name}.settings import PRUNING
from {work_path}.{project_name}.settings import MODEL_NAME
from {work_path}.{project_name}.settings import MODEL_PATH
from {work_path}.{project_name}.settings import CHECKPOINT_PATH
from {work_path}.{project_name}.settings import PRUNING_MODEL_NAME

model = operator.methodcaller(MODEL)(Models)
try:
    weight = CallBack.calculate_the_best_weight()
    logger.info(f'读取的权重为{{weight}}')
    model.load_weights(os.path.join(CHECKPOINT_PATH, weight))
except:
    raise OSError(f'没有任何的权重和模型在{{MODEL_PATH}}')
weight_path = os.path.join(MODEL_PATH, MODEL_NAME)
model.save(weight_path)
logger.success(f'{{weight_path}}模型保存成功')

if PRUNING:
    logger.debug('开始进行模型压缩')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()
    quantized_and_pruned_tflite_file = os.path.join(MODEL_PATH, PRUNING_MODEL_NAME)
    with open(quantized_and_pruned_tflite_file, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)
    logger.success(f'{{quantized_and_pruned_tflite_file}}模型保存成功')

"""


def settings(work_path, project_name):
    return f"""import os
import datetime

## 硬件配置
# 是否使用GPU
USE_GPU = True

# 模式选择 ORDINARY默认模式，需要设置验证码的长度 | NUM_CLASSES图片分类 | CTC识别文字，不需要文本设置长度 | CTC_TINY识别文字，需要设置长度
# | EFFICIENTDET目标检测 YOLO目标检测 YOLO_TINY目标检测
MODE = 'ORDINARY'

## 超参数设置(通用设置)
# 学习率
LR = 1e-4

# 标签平滑
LABEL_SMOOTHING = 0

# 是否使用余弦退火衰减(默认关闭,False使用的是常数衰减)
COSINE_SCHEDULER = False

# 训练次数
EPOCHS = 900

# batsh批次
BATCH_SIZE = 4

# 训练多少轮验证损失下不去，学习率/10
LR_PATIENCE = 2

# 训练多少轮验证损失下不去，停止训练
EARLY_PATIENCE = 16

## 图片设置请先运行check_file.py查看图片的宽和高，设置的高和宽最好大于等于你的数据集的高和宽
# 图片高度
IMAGE_HEIGHT = 120

# 图片宽度
IMAGE_WIDTH = 120

# 图片通道
IMAGE_CHANNALS = 3

# 二值化阈值(图片通道为3时生效)
THRESHOLD = 180

# 验证码的长度
CAPTCHA_LENGTH = 8

# 是否使用数据增强(数据集多的时候不需要用，接收一个整数，代表增强多少张图片)
DATA_ENHANCEMENT = False

## 数据集设置
# 是否划分出验证集
DIVIDE = True

# 划分的比例
DIVIDE_RATO = 0.1

## YOLO | YOLO_TINY
# mosaic数据增强
MOSAIC = False

# 最大目标数
MAX_BOXES = 50

# EFFICIENTDET
# 选择模型
PHI = 0

# 图片尺寸
IMAGE_SIZES = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

## 模型设置
# 定义模型的方法,模型在models.py定义

# 是否压缩模型
PRUNING = False

if MODE == 'ORDINARY':
    MODEL = 'captcha_model'
elif MODE == 'NUM_CLASSES':
    MODEL = 'captcha_model_num_classes'
elif MODE == 'CTC':
    MODEL = 'captcha_model_ctc'
elif MODE == 'CTC_TINY':
    MODEL = 'captcha_model_ctc_tiny'
elif MODE == 'YOLO':
    MODEL = 'captcha_model_yolo'
elif MODE == 'YOLO_TINY':
    MODEL = 'captcha_model_yolo_tiny'
elif MODE == 'EFFICIENTDET':
    MODEL = 'captcha_model_efficientdet'

# 保存的模型名称
MODEL_NAME = 'captcha.h5'

# 压缩后的模型名称
PRUNING_MODEL_NAME = 'captcha.tflite'

## 路径设置，一般无需改动
# 可视化配置batch或epoch
UPDATE_FREQ = 'epoch'

# 项目工作路径
BASIC_PATH = os.path.dirname(os.path.abspath(__file__))

# 训练集路径
TRAIN_PATH = os.path.join(BASIC_PATH, 'train_dataset')

# 验证集路径
VALIDATION_PATH = os.path.join(BASIC_PATH, 'validation_dataset')

# 测试集路径
TEST_PATH = os.path.join(BASIC_PATH, 'test_dataset')

# 标签路径
LABEL_PATH = os.path.join(BASIC_PATH, 'label')

# 打包训练集路径
TFRECORD_TRAIN_PATH = os.path.join(BASIC_PATH, 'train_pack_dataset')

# 打包验证集
TFRECORD_VALIDATION_PATH = os.path.join(BASIC_PATH, 'validation_pack_dataset')

# 模型保存路径
MODEL_PATH = os.path.join(BASIC_PATH, 'model')

# 可视化日志路径
LOG_DIR = os.path.join(os.path.join(BASIC_PATH, 'logs'), f'{{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}}')

# csv_logger日志路径
CSV_PATH = os.path.join(os.path.join(BASIC_PATH, 'CSVLogger'), 'traing.csv')

# 断点续训路径
CHECKPOINT_PATH = os.path.join(BASIC_PATH, 'checkpoint')  # 检查点路径

if MODE == 'CTC':
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH,
                                        'Model_weights.-{{epoch:02d}}-{{val_loss:.4f}}-{{val_word_acc:.4f}}.hdf5')
elif MODE == 'YOLO' or MODE == 'CTC_TINY' or MODE == 'YOLO_TINY':
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'Model_weights.-{{epoch:02d}}-{{loss:.4f}}.hdf5')
elif MODE == 'EFFICIENTDET':
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH,
                                        'Model_weights.-{{epoch:02d}}-{{loss:.4f}}-{{regression_loss:.4f}}-{{classification_loss:.4f}}.hdf5')
else:
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH,
                                        'Model_weights.-{{epoch:02d}}-{{val_loss:.4f}}-{{val_acc:.4f}}.hdf5')
# TF训练集(打包后)
TRAIN_PACK_PATH = os.path.join(BASIC_PATH, 'train_pack_dataset')

# TF验证集(打包后)
VALIDATION_PACK_PATH = os.path.join(BASIC_PATH, 'validation_pack_dataset')

# 提供后端放置的模型路径
APP_MODEL_PATH = os.path.join(BASIC_PATH, 'App_model')

# 映射表
NUMBER_CLASSES_FILE = os.path.join(BASIC_PATH, 'num_classes.json')

# 权重
WEIGHT = os.path.join(BASIC_PATH, 'weight')

# 先验框
ANCHORS_PATH = os.path.join(BASIC_PATH, 'anchors.json')

"""


def spider_example(work_path, project_name):
    return f"""import time
import json
import base64
import random
import requests
from loguru import logger


def get_captcha():
    r = int(random.random() * 100000000)
    params = {{
        'r': str(r),
        's': '0',
    }}
    response = requests.get('https://login.sina.com.cn/cgi/pin.php', params=params)
    if response.status_code == 200:
        return response.content


if __name__ == '__main__':
    content = get_captcha()
    if content:
        logger.info(f'获取验证码成功')
        with open(f'{{int(time.time())}}.jpg', 'wb') as f:
            f.write(content)
        data = {{'data': [f'data:image/jpeg;base64,{{base64.b64encode(content).decode()}}']}}
        response = requests.post('http://127.0.0.1:7860/api/predict/', json=data)
        data = json.loads(response.text)
        label = data.get('data')[0].get('label')
        label = json.loads(label)
        label = label.get('result')
        logger.debug(f'验证码为{{label}}')
    else:
        logger.error(f'获取验证码失败')

"""


def test(work_path, project_name):
    return f"""import os
import time
import random
import numpy as np
import tensorflow as tf
from loguru import logger
from {work_path}.{project_name}.settings import USE_GPU
from {work_path}.{project_name}.settings import PRUNING
from {work_path}.{project_name}.settings import TEST_PATH
from {work_path}.{project_name}.settings import MODEL_PATH
from {work_path}.{project_name}.settings import MODEL_NAME
from {work_path}.{project_name}.settings import PRUNING_MODEL_NAME
from {work_path}.{project_name}.utils import running_time
from {work_path}.{project_name}.utils import Predict_Image
from {work_path}.{project_name}.utils import Image_Processing

start = time.time()
time_list = []

logger.add('result.log')


if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    if gpus:
        logger.info("use gpu device")
        logger.info(f'可用GPU数量: {{len(gpus)}}')
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

test_image_list = Image_Processing.extraction_image(TEST_PATH)
number = len(test_image_list)
random.shuffle(test_image_list)
if PRUNING:
    model_path = os.path.join(MODEL_PATH, PRUNING_MODEL_NAME)
else:
    model_path = os.path.join(MODEL_PATH, MODEL_NAME)

logger.debug(f'加载模型{{model_path}}')
if not os.path.exists(model_path):
    raise OSError(f'{{model_path}}没有模型')
Predict = Predict_Image(model_path=model_path)
for image in test_image_list:
    start_time = time.time()
    number -= 1
    Predict.predict_image(image)
    end_time = time.time()
    now_time = end_time - start_time
    time_list.append(now_time)
    logger.debug(f'已耗时: {{running_time(end_time - start)}}')
    logger.debug(f'预计耗时: {{running_time(np.mean(time_list) * number)}}')
end = time.time()
logger.info(f'总共运行时间{{running_time(end - start)}}')

"""


def train(work_path, project_name):
    return f"""import os
import random
import operator
import pandas as pd
import tensorflow as tf
from loguru import logger
from {work_path}.{project_name}.utils import cheak_path
from {work_path}.{project_name}.utils import BBoxUtility
from {work_path}.{project_name}.utils import parse_function
from {work_path}.{project_name}.utils import Image_Processing
from {work_path}.{project_name}.utils import YOLO_Generator
from {work_path}.{project_name}.utils import Efficientdet_Generator
from {work_path}.{project_name}.models import Models
from {work_path}.{project_name}.models import Settings
from {work_path}.{project_name}.models import YOLO_anchors
from {work_path}.{project_name}.models import Efficientdet_anchors
from {work_path}.{project_name}.callback import CallBack
from {work_path}.{project_name}.settings import PHI
from {work_path}.{project_name}.settings import MODE
from {work_path}.{project_name}.settings import MODEL
from {work_path}.{project_name}.settings import MOSAIC
from {work_path}.{project_name}.settings import EPOCHS
from {work_path}.{project_name}.settings import USE_GPU
from {work_path}.{project_name}.settings import BATCH_SIZE
from {work_path}.{project_name}.settings import MODEL_PATH
from {work_path}.{project_name}.settings import MODEL_NAME
from {work_path}.{project_name}.settings import IMAGE_SIZES
from {work_path}.{project_name}.settings import IMAGE_HEIGHT
from {work_path}.{project_name}.settings import IMAGE_WIDTH
from {work_path}.{project_name}.settings import CSV_PATH
from {work_path}.{project_name}.settings import TRAIN_PATH
from {work_path}.{project_name}.settings import VALIDATION_PATH
from {work_path}.{project_name}.settings import TEST_PATH
from {work_path}.{project_name}.settings import TRAIN_PACK_PATH
from {work_path}.{project_name}.settings import VALIDATION_PACK_PATH

if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    if gpus:
        logger.info("use gpu device")
        logger.info(f'可用GPU数量: {{len(gpus)}}')
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

if MODE == 'YOLO' or MODE == 'YOLO_TINY':
    with tf.device('/cpu:0'):
        train_image = Image_Processing.extraction_image(TRAIN_PATH)
        random.shuffle(train_image)
        validation_image = Image_Processing.extraction_image(VALIDATION_PATH)
        test_image = Image_Processing.extraction_image(TEST_PATH)
        Image_Processing.extraction_label(train_image + validation_image + test_image)
        train_label = Image_Processing.extraction_label(train_image)
        validation_label = Image_Processing.extraction_label(validation_image)

    logger.info(f'一共有{{int(len(Image_Processing.extraction_image(TRAIN_PATH)) / BATCH_SIZE)}}个batch')
    try:
        logs = pd.read_csv(CSV_PATH)
        data = logs.iloc[-1]
        initial_epoch = int(data.get('epoch')) + 1
    except:
        initial_epoch = 0

    anchors = YOLO_anchors.get_anchors()

    model, c_callback = CallBack.callback(operator.methodcaller(MODEL)(Models))
    model.summary()
    if validation_image:
        model.fit(
            YOLO_Generator().data_generator(train_image, train_label, BATCH_SIZE, (IMAGE_HEIGHT, IMAGE_WIDTH), anchors,
                                            Settings.settings_num_classes(), mosaic=MOSAIC),
            steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
            validation_data=YOLO_Generator().data_generator(validation_image, validation_label, BATCH_SIZE,
                                                            (IMAGE_HEIGHT, IMAGE_WIDTH), anchors,
                                                            Settings.settings_num_classes(), mosaic=MOSAIC),
            validation_steps=max(1, len(validation_image) // BATCH_SIZE),
            initial_epoch=initial_epoch,
            epochs=EPOCHS,
            max_queue_size=1,
            verbose=2,
            callbacks=c_callback)
    else:
        logger.debug('没有验证集')
        model.fit(
            YOLO_Generator().data_generator(train_image, train_label, BATCH_SIZE, (IMAGE_HEIGHT, IMAGE_WIDTH), anchors,
                                            Settings.settings_num_classes(), mosaic=MOSAIC),
            steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
            initial_epoch=initial_epoch,
            epochs=EPOCHS,
            max_queue_size=1,
            verbose=2,
            callbacks=c_callback)


elif MODE == 'EFFICIENTDET':
    with tf.device('/cpu:0'):
        train_image = Image_Processing.extraction_image(TRAIN_PATH)
        random.shuffle(train_image)
        validation_image = Image_Processing.extraction_image(VALIDATION_PATH)
        test_image = Image_Processing.extraction_image(TEST_PATH)
        Image_Processing.extraction_label(train_image + validation_image + test_image)
        train_label = Image_Processing.extraction_label(train_image)
        validation_label = Image_Processing.extraction_label(validation_image)

    logger.info(f'一共有{{int(len(Image_Processing.extraction_image(TRAIN_PATH)) / BATCH_SIZE)}}个batch')
    try:
        logs = pd.read_csv(CSV_PATH)
        data = logs.iloc[-1]
        initial_epoch = int(data.get('epoch')) + 1
    except:
        initial_epoch = 0
    model, c_callback = CallBack.callback(operator.methodcaller(MODEL)(Models))
    model.summary()
    priors = Efficientdet_anchors.get_anchors(IMAGE_SIZES[PHI])
    bbox_util = BBoxUtility(Settings.settings_num_classes(), priors)
    if validation_image:
        model.fit(
            Efficientdet_Generator(bbox_util, BATCH_SIZE, train_image, train_label,
                                   (IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]), Settings.settings_num_classes()).generate(),
            validation_data=Efficientdet_Generator(bbox_util, BATCH_SIZE, validation_image, validation_label,
                                                   (IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]),
                                                   Settings.settings_num_classes()).generate(),
            validation_steps=max(1, len(validation_image) // BATCH_SIZE),
            steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
            initial_epoch=initial_epoch,
            epochs=EPOCHS,
            verbose=2,
            callbacks=c_callback)
    else:
        logger.debug('没有验证集')
        model.fit(
            Efficientdet_Generator(bbox_util, BATCH_SIZE, train_image, train_label,
                                   (IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]), Settings.settings_num_classes()).generate(),
            steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
            initial_epoch=initial_epoch,
            epochs=EPOCHS,
            verbose=2,
            callbacks=c_callback)

else:
    with tf.device('/cpu:0'):
        train_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(TRAIN_PACK_PATH)).map(
            map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            batch_size=BATCH_SIZE).prefetch(
            buffer_size=BATCH_SIZE)
        logger.debug(train_dataset)
        validation_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(VALIDATION_PACK_PATH)).map(
            map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            batch_size=BATCH_SIZE).prefetch(
            buffer_size=BATCH_SIZE)
    if MODE == 'CTC_TINY':
        model, c_callback = CallBack.callback(operator.methodcaller(MODEL, training=True)(Models))
    else:
        model, c_callback = CallBack.callback(operator.methodcaller(MODEL)(Models))

    model.summary()
    logger.info(f'一共有{{int(len(Image_Processing.extraction_image(TRAIN_PATH)) / BATCH_SIZE)}}个batch')

    try:
        logs = pd.read_csv(CSV_PATH)
        data = logs.iloc[-1]
        initial_epoch = int(data.get('epoch')) + 1
    except:
        initial_epoch = 0
    if VALIDATION_PACK_PATH:
        model.fit(train_dataset, initial_epoch=initial_epoch, epochs=EPOCHS, callbacks=c_callback,
                  validation_data=validation_dataset, verbose=2)
    else:
        model.fit(train_dataset, initial_epoch=initial_epoch, epochs=EPOCHS, callbacks=c_callback, verbose=2)

save_model_path = cheak_path(os.path.join(MODEL_PATH, MODEL_NAME))

model.save(save_model_path, save_format='tf')

"""


class New_Work(object):
    def __init__(self, work_path='works', project_name='project'):
        self.work_parh = work_path
        self.project_name = project_name
        self.work = os.path.join(os.path.dirname(os.path.abspath(__file__)), work_path)
        if not os.path.exists(self.work):
            os.mkdir(self.work)
        self.path = os.path.join(self.work, self.project_name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        else:
            raise OSError('已有的项目')

    def file_name(self, name):
        return os.path.join(self.path, name)

    def callback(self):
        with open(self.file_name('callback.py'), 'w', encoding='utf-8') as f:
            f.write(callback(self.work_parh, self.project_name))

    def app(self):
        with open(self.file_name('app.py'), 'w', encoding='utf-8') as f:
            f.write(app(self.work_parh, self.project_name))

    def captcha_config(self):
        with open(self.file_name('captcha_config.json'), 'w') as f:
            f.write(captcha_config())

    def check_file(self):
        with open(self.file_name('check_file.py'), 'w', encoding='utf-8') as f:
            f.write(check_file(self.work_parh, self.project_name))

    def delete_file(self):
        with open(self.file_name('delete_file.py'), 'w', encoding='utf-8') as f:
            f.write(delete_file(self.work_parh, self.project_name))

    def utils(self):
        with open(self.file_name('utils.py'), 'w', encoding='utf-8') as f:
            f.write(utils(self.work_parh, self.project_name))

    def gen_sample_by_captcha(self):
        with open(self.file_name('gen_sample_by_captcha.py'), 'w', encoding='utf-8') as f:
            f.write(gen_sample_by_captcha(self.work_parh, self.project_name))

    def init_working_space(self):
        with open(self.file_name('init_working_space.py'), 'w', encoding='utf-8') as f:
            f.write(init_working_space(self.work_parh, self.project_name))

    def models(self):
        with open(self.file_name('models.py'), 'w', encoding='utf-8') as f:
            f.write(models(self.work_parh, self.project_name))

    def split_dataset(self):
        with open(self.file_name('split_dataset.py'), 'w', encoding='utf-8') as f:
            f.write(split_dataset(self.work_parh, self.project_name))

    def pack_dataset(self):
        with open(self.file_name('pack_dataset.py'), 'w', encoding='utf-8') as f:
            f.write(pack_dataset(self.work_parh, self.project_name))

    def save_model(self):
        with open(self.file_name('save_model.py'), 'w', encoding='utf-8') as f:
            f.write(save_model(self.work_parh, self.project_name))

    def settings(self):
        with open(self.file_name('settings.py'), 'w', encoding='utf-8') as f:
            f.write(settings(self.work_parh, self.project_name))

    def spider_example(self):
        with open(self.file_name('spider_example.py'), 'w', encoding='utf-8') as f:
            f.write(spider_example(self.work_parh, self.project_name))

    def test(self):
        with open(self.file_name('test.py'), 'w', encoding='utf-8') as f:
            f.write(test(self.work_parh, self.project_name))

    def train(self):
        with open(self.file_name('train.py'), 'w', encoding='utf-8') as f:
            f.write(train(self.work_parh, self.project_name))

    def main(self):
        self.callback()
        self.app()
        self.captcha_config()
        self.check_file()
        self.delete_file()
        self.utils()
        self.gen_sample_by_captcha()
        self.init_working_space()
        self.models()
        self.split_dataset()
        self.pack_dataset()
        self.save_model()
        self.settings()
        self.spider_example()
        self.test()
        self.train()


if __name__ == '__main__':
    New_Work(work_path='works', project_name='simple').main()
