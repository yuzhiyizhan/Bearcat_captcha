import re


def project(string: str, work_path: str, project_name: str):
    string = re.sub('{', '{{', string)
    string = re.sub('}', '}}', string)
    string = re.sub(work_path, '{work_path}', string)
    string = re.sub(project_name, '{project_name}', string)
    print(string)


if __name__ == '__main__':
    string = """import random
from loguru import logger
from works.simple.settings import TRAIN_PATH
from works.simple.settings import VALIDATION_PATH
from works.simple.settings import TEST_PATH
from works.simple.settings import DATA_ENHANCEMENT
from works.simple.settings import TFRECORD_TRAIN_PATH
from works.simple.settings import TFRECORD_VALIDATION_PATH
from works.simple.utils import Image_Processing
from works.simple.utils import WriteTFRecord
from concurrent.futures import ThreadPoolExecutor

if DATA_ENHANCEMENT:
    image_path = Image_Processing.extraction_image(TRAIN_PATH)
    number = len(image_path)
    for i in image_path:
        number = number - 1
        Image_Processing.preprosess_save_images(i, number)

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

    project(string, work_path='works', project_name='simple')
