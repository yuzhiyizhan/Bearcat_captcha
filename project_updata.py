import re


def project(string: str, work_path: str, project_name: str):
    string = re.sub('{', '{{', string)
    string = re.sub('}', '}}', string)
    string = re.sub(work_path, '{work_path}', string)
    string = re.sub(project_name, '{project_name}', string)
    print(string)


if __name__ == '__main__':
    string = """import os
import random
import operator
import pandas as pd
import tensorflow as tf
from loguru import logger
from works.simple_coco.models import Models
from works.simple_coco.models import Settings
from works.simple_coco.models import SSD_anchors
from works.simple_coco.models import YOLO_anchors
from works.simple_coco.models import Efficientdet_anchors
from works.simple_coco.utils import cheak_path
from works.simple_coco.utils import SSD_Generator
from works.simple_coco.utils import parse_function
from works.simple_coco.utils import YOLO_Generator
from works.simple_coco.utils import SSD_BBoxUtility
from works.simple_coco.utils import Image_Processing
from works.simple_coco.utils import Efficientdet_Generator
from works.simple_coco.utils import EfficientDet_BBoxUtility
from works.simple_coco.callback import CallBack
from works.simple_coco.settings import PHI
from works.simple_coco.settings import MODE
from works.simple_coco.settings import MODEL
from works.simple_coco.settings import MOSAIC
from works.simple_coco.settings import EPOCHS
from works.simple_coco.settings import USE_GPU
from works.simple_coco.settings import BATCH_SIZE
from works.simple_coco.settings import MODEL_PATH
from works.simple_coco.settings import MODEL_NAME
from works.simple_coco.settings import IMAGE_SIZES
from works.simple_coco.settings import IMAGE_HEIGHT
from works.simple_coco.settings import IMAGE_WIDTH
from works.simple_coco.settings import CSV_PATH
from works.simple_coco.settings import TRAIN_PATH
from works.simple_coco.settings import VALIDATION_PATH
from works.simple_coco.settings import TEST_PATH
from works.simple_coco.settings import TRAIN_PACK_PATH
from works.simple_coco.settings import VALIDATION_PACK_PATH

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

if MODE == 'YOLO' or MODE == 'YOLO_TINY':
    with tf.device('/cpu:0'):
        train_image = Image_Processing.extraction_image(TRAIN_PATH)
        random.shuffle(train_image)
        validation_image = Image_Processing.extraction_image(VALIDATION_PATH)
        test_image = Image_Processing.extraction_image(TEST_PATH)
        Image_Processing.extraction_label(train_image + validation_image + test_image)
        train_label = Image_Processing.extraction_label(train_image)
        validation_label = Image_Processing.extraction_label(validation_image)

    logger.info(f'一共有{int(len(Image_Processing.extraction_image(TRAIN_PATH)) / BATCH_SIZE)}个batch')
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
    save_model_path = cheak_path(os.path.join(MODEL_PATH, MODEL_NAME))

    model.save(save_model_path, save_format='tf')


elif MODE == 'EFFICIENTDET':
    with tf.device('/cpu:0'):
        train_image = Image_Processing.extraction_image(TRAIN_PATH)
        random.shuffle(train_image)
        validation_image = Image_Processing.extraction_image(VALIDATION_PATH)
        test_image = Image_Processing.extraction_image(TEST_PATH)
        Image_Processing.extraction_label(train_image + validation_image + test_image)
        train_label = Image_Processing.extraction_label(train_image)
        validation_label = Image_Processing.extraction_label(validation_image)

    logger.info(f'一共有{int(len(Image_Processing.extraction_image(TRAIN_PATH)) / BATCH_SIZE)}个batch')

    model, c_callback = CallBack.callback(operator.methodcaller(MODEL)(Models))
    model.summary()
    priors = Efficientdet_anchors.get_anchors(IMAGE_SIZES[PHI])
    bbox_util = EfficientDet_BBoxUtility(Settings.settings_num_classes(), priors)
    for _ in range(EPOCHS):
        try:
            logs = pd.read_csv(CSV_PATH)
            data = logs.iloc[-1]
            initial_epoch = int(data.get('epoch')) + 1
        except:
            initial_epoch = 0
        if validation_image:
            model.fit(
                Efficientdet_Generator(bbox_util, BATCH_SIZE, train_image, train_label,
                                       (IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]),
                                       Settings.settings_num_classes()).generate(),
                validation_data=Efficientdet_Generator(bbox_util, BATCH_SIZE, validation_image, validation_label,
                                                       (IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]),
                                                       Settings.settings_num_classes()).generate(),
                validation_steps=max(1, len(validation_image) // BATCH_SIZE),
                steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
                initial_epoch=initial_epoch,
                epochs=initial_epoch + 1,
                verbose=2,
                callbacks=c_callback)
        else:
            logger.debug('没有验证集')
            model.fit(
                Efficientdet_Generator(bbox_util, BATCH_SIZE, train_image, train_label,
                                       (IMAGE_SIZES[PHI], IMAGE_SIZES[PHI]),
                                       Settings.settings_num_classes()).generate(),
                steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
                initial_epoch=initial_epoch,
                epochs=initial_epoch + 1,
                verbose=2,
                callbacks=c_callback)

elif MODE == 'SSD':
    with tf.device('/cpu:0'):
        train_image = Image_Processing.extraction_image(TRAIN_PATH)
        random.shuffle(train_image)
        validation_image = Image_Processing.extraction_image(VALIDATION_PATH)
        test_image = Image_Processing.extraction_image(TEST_PATH)
        Image_Processing.extraction_label(train_image + validation_image + test_image)
        train_label = Image_Processing.extraction_label(train_image)
        validation_label = Image_Processing.extraction_label(validation_image)

    logger.info(f'一共有{int(len(Image_Processing.extraction_image(TRAIN_PATH)) / BATCH_SIZE)}个batch')

    model, c_callback = CallBack.callback(operator.methodcaller(MODEL)(Models))
    model.summary()
    priors = SSD_anchors.get_anchors((IMAGE_HEIGHT, IMAGE_WIDTH))
    bbox_util = SSD_BBoxUtility(Settings.settings(), priors)
    for _ in range(EPOCHS):
        try:
            logs = pd.read_csv(CSV_PATH)
            data = logs.iloc[-1]
            initial_epoch = int(data.get('epoch')) + 1
        except:
            initial_epoch = 0

        if validation_image:
            model.fit(
                SSD_Generator(bbox_util, BATCH_SIZE, train_image, train_label,
                              (IMAGE_HEIGHT, IMAGE_WIDTH), Settings.settings()).generate(),
                validation_data=SSD_Generator(bbox_util, BATCH_SIZE, validation_image, validation_label,
                                              (IMAGE_HEIGHT, IMAGE_WIDTH),
                                              Settings.settings()).generate(),
                validation_steps=max(1, len(validation_image) // BATCH_SIZE),
                steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
                initial_epoch=initial_epoch,
                epochs=initial_epoch + 1,
                verbose=2,
                callbacks=c_callback)
        else:
            logger.debug('没有验证集')
            model.fit(
                SSD_Generator(bbox_util, BATCH_SIZE, train_image, train_label,
                              (IMAGE_HEIGHT, IMAGE_WIDTH), Settings.settings()).generate(),
                steps_per_epoch=max(1, len(train_image) // BATCH_SIZE),
                initial_epoch=initial_epoch,
                epochs=initial_epoch + 1,
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
    logger.info(f'一共有{int(len(Image_Processing.extraction_image(TRAIN_PATH)) / BATCH_SIZE)}个batch')

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

    project(string, work_path='works', project_name='simple_coco')
