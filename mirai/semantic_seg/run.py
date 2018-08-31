#!/usr/bin/env python
import tensorflow as tf
import helper
import numpy as np
import scipy.misc
from moviepy.editor import VideoFileClip

def detect_road(img):
    image = scipy.misc.imresize(img, image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > confidence).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[255, 255, 0, 127]]))
    mask[:roi_max_height, :] = 0
    mask = scipy.misc.imresize(mask, output_shape)
    # scipy.misc.imsave('./tests/mask.png', mask)
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(img)
    street_im.paste(mask, box=None, mask=mask)
    return np.array(street_im)

clip = VideoFileClip("./driving.mp4").subclip(0, 5)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./model/model.meta")
    saver.restore(sess, "./model/model")
    image_shape = (160, 576)
    output_shape = (720, 1280)
    roi_height = 0.6
    confidence = 0.7
    roi_max_height = int(image_shape[0]*roi_height)
    logits = tf.get_default_graph().get_tensor_by_name("logits:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
    input = tf.get_default_graph().get_tensor_by_name("image_input:0")
    # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, output_shape, logits, keep_prob, input)
    new_clip = clip.fl_image(detect_road)
    new_clip.write_videofile("./result.mp4")
    # image = scipy.misc.imread("./tests/extracted-0.0.jpg")
    # ret = detect_road(image)
    # scipy.misc.imsave('./tests/final.jpg', ret)
