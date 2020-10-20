#!/user/bin/env python
# coding=utf-8
"""
@file: 模型建立.py
@author: zwt
@time: 2020/10/20 18:39
@desc: 
"""
import tensorflow as tf
"""
模型的构建： tf.keras.Model 和 tf.keras.layers
模型的损失函数： tf.keras.losses
模型的优化器： tf.keras.optimizer
模型的评估： tf.keras.metrics
"""


class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        layer1 = tf.keras.layers
        

    def call(self, input):

        return output