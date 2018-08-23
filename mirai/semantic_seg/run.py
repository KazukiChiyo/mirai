#!/usr/bin/env python
import tensorflow as tf
import helper

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./model/model.ckpt.meta")
    saver.restore(sess, "./model/model.ckpt")
    runs_dir = './runs'
    data_dir = './data'
    image_shape = (160, 576)
    output_shape = (375, 1242)
    logits = tf.get_default_graph().get_tensor_by_name("logits:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
    input = tf.get_default_graph().get_tensor_by_name("image_input:0")
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, output_shape, logits, keep_prob, input)
