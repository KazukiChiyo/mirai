#!/usr/bin/env python
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from tqdm import tqdm


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    Parameters:
    -----------
    sess: tf.Session object
        TensorFlow Session.
    vgg_path: string
        Path to vgg folder, containing "variables/" and "saved_model.pb".
    """
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    l3_output = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    l4_output = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    l7_output = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, l3_output, l4_output, l7_output


def decoder(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the decoder for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    stddev = 0.01
    l2_scale = 1e-3
    padding = "same"
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_scale)

    conv_3 = tf.layers.conv2d(
        inputs=vgg_layer3_out,
        filters=num_classes,
        kernel_size=1,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    conv_4 = tf.layers.conv2d(
        inputs=vgg_layer4_out,
        filters=num_classes,
        kernel_size=1,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    conv_7 = tf.layers.conv2d(
        inputs=vgg_layer7_out,
        filters=num_classes,
        kernel_size=1,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    deconv_1 = tf.layers.conv2d_transpose(
        inputs=conv_7,
        filters=num_classes,
        kernel_size=4,
        strides=(2, 2),
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    skip_1 = tf.add(conv_4, deconv_1)

    deconv_2 = tf.layers.conv2d_transpose(
        inputs=skip_1,
        filters=num_classes,
        kernel_size=4,
        strides=(2, 2),
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    skip_2 = tf.add(conv_3, deconv_2)

    deconv_3 = tf.layers.conv2d_transpose(
        inputs=skip_2,
        filters=num_classes,
        kernel_size=16,
        strides=(8, 8),
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    return deconv_3

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    Parameters:
    -----------
    sess: tf.Session object
        TensorFlow Session.
    epochs: int
        Number of epochs.
    batch_size: int
        Batch size.
    get_batches_fn: function object
        Batch generator.
    train_op: tf.train.optimizer object
        Optimizer.
    cross_entropy_loss: tf.tensor object
        Loss function.
    input_image: tf.Placeholder object
        Placeholder for input images.
    correct_label: tf.Placeholder object
        Placeholder for image labels.
    keep_prob: tf.Placeholder object
        Placeholder
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    _keep_prob = 0.5
    _learning_rate = 1e-4
    sess.run(tf.global_variables_initializer())
    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for X, y in tqdm(get_batches_fn(batch_size)):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: X, correct_label: y, keep_prob: _keep_prob, learning_rate: _learning_rate})
        print("Loss: = {:.3f}".format(loss))
        print()


def run():
    num_classes = 2
    image_shape = (160, 576)
    batch_size = 1
    n_epochs = 10

    data_dir = './data'

    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        print("Importing VGG model as encoder...")
        input, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        print("Generating decoder...")
        output = decoder(l3_out, l4_out, l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        train_nn(sess, n_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input, correct_label, keep_prob, learning_rate)
        print("Saving model...")
        tf.train.Saver().save(sess, "./model/model")

if __name__ == '__main__':
    run()
