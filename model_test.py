import os
import json
import logging
from collections import OrderedDict
import codecs
import tensorflow as tf

def get_logger(log_file):
    """
    定义日志方法
    :param log_file:
    :return:
    """
    # 创建一个logging的实例 logger
    logger = logging.getLogger(log_file)
    # 设置logger的全局日志级别为DEBUG
    logger.setLevel(logging.DEBUG)
    # 创建一个日志文件的handler，并且设置日志级别为DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # 创建一个控制台的handler，并设置日志级别为DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def load_config(config_file):
    """
    加载配置文件
    :param config_file:
    :return:
    """
    with open(config_file, encoding='utf-8') as f:
        return json.load(f)


def create(sess, Model, ckpt_path, config):
    """
    :param sess:
    :param Model:
    :param ckpt_path:
    :param config:
    :return:
    """
    model = Model(config)

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception("没有已存在模型可以获取")

    return model

def result_to_json(strings, tags):
    """
    :param strings:
    :param tags:
    :return:
    """
    item = {"string":strings, "entities":[]}

    entity_name = ""
    entity_start = 0
    idx = 0

    for word, tag in zip(strings, tags):
        if tag[0] == "S":
            item['entities'].append({"word":word, "start":idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name = entity_name + word
            entity_start = idx
        elif tag[0] == "I":
            entity_name = entity_name + word
        elif tag[0] == "E":
            entity_name = entity_name + word
            item['entities'].append({"word":entity_name, "start":entity_start, "end":idx+1, "type":tag[2:]})
        else:
            entity_name=""
            entity_start = idx
        idx = idx + 1
    return item
