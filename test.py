import tensorflow as tf
import numpy as np

import data
import argparse
import utils
import os

from utils import Option

opt = Option('./config.json')

utils.init()

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

args, flags = utils.parse_args(opt, parser)
mnist = data.mnist(args)

save_path = args['log_dir']

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(save_path + '/cnn_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint(save_path))
    CNN = tf.get_default_graph()
    preds, feats = np.zeros([mnist.n_images, 22]), np.zeros([mnist.n_images, 128])

    for m, batch_idx, images1, images2, labels, mask in mnist.next_batch(is_training=True):
        feed_dict = {CNN.get_tensor_by_name('inputs:0'): images1,
                     CNN.get_tensor_by_name('is_first:0'): False,
                     CNN.get_tensor_by_name('is_training:0'): False}

        feat, pred = sess.run([CNN.get_tensor_by_name('test_features:0'),
                               CNN.get_tensor_by_name('test_preds:0')],
                              feed_dict=feed_dict)
        preds[batch_idx, :] = pred
        feats[batch_idx, :] = feat

    arg_target = np.argmax(mnist.y_train, axis=1)
    arg_pred = np.argmax(preds, axis=1)

    acc = np.sum(arg_target == arg_pred, dtype=np.float32) / mnist.n_images
    print('Top-1 Accuracy of CNN (Train): {0:0.4f}'.format(acc))

    np.savetxt('results/feats.tsv', feats, fmt='%.4f', delimiter='\t')
    np.savetxt('results/labels.tsv', preds, fmt='%.4f', delimiter='\t')

    preds_test, feats_test = [], []
    for m, images1, target in mnist.next_batch(is_training=False):
        feed_dict = {CNN.get_tensor_by_name('inputs:0'): images1,
                     CNN.get_tensor_by_name('is_first:0'): False,
                     CNN.get_tensor_by_name('is_training:0'): False}

        feat, pred = sess.run([CNN.get_tensor_by_name('test_features:0'),
                               CNN.get_tensor_by_name('test_preds:0')],
                              feed_dict=feed_dict)
        preds_test.extend(pred)
        feats_test.extend(feat)

    arg_pred = np.argmax(preds_test, axis=1)
    arg_target = np.argmax(mnist.y_test[:len(arg_pred)], axis=1)
    
    np.savetxt('results/test_labels.tsv', preds_test, fmt='%.4f', delimiter='\t') # 테스트레이블

    acc = np.mean(arg_target == arg_pred, dtype=np.float32)
    print('Top-1 Accuracy of CNN (Test): {0:0.4f}'.format(acc))
