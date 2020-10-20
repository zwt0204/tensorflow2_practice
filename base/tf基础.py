#!/user/bin/env python
# coding=utf-8
"""
@file: tf基础.py
@author: zwt
@time: 2020/10/20 16:44
@desc: 
"""
import tensorflow as tf

# 定义一个随机数标量
random_float = tf.random.uniform(shape=())
print(random_float)

# 定义一个有两个元素的零向量
zero_vector = tf.zeros(shape=(2))
print(zero_vector)

# 定已两个2*2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[1., 2.], [3., 4.]])

C = tf.add(A, B)
print(C)
print(C.shape)
print(C.dtype)
print(C.numpy())

D = tf.matmul(A, B)
print(D)

# 自动求导机制

x = tf.Variable(initial_value=3.)
# 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
with tf.GradientTape() as tape:
    y = tf.square(x)
# 计算y关于x的导数
y_grad = tape.gradient(y, x)
print(y, y_grad)

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print(L, w_grad, b_grad)