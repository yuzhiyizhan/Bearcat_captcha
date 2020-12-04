import tensorflow as tf
from loguru import logger

logger.debug(tf.test.gpu_device_name())
logger.debug(tf.test.is_built_with_cuda())
logger.debug(tf.test.is_built_with_gpu_support())
