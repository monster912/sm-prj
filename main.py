import tensorflow as tf
import numpy as np
import time
import math
import argparse
import model
import utils
import data
import os

from utils import Option
opt = Option('./config.json')

utils.init()

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

args, flags = utils.parse_args(opt, parser)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.set_random_seed(args['random_seed'])
print('a')
def rampup(epoch):
    if epoch < args['rampup_length']:
        p = max(0.0, float(epoch)) / float(args['rampup_length'])
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    else:
        return 1.0

def rampdown(epoch):
    if epoch >= (args['n_epochs'] - args['rampdown_length']):
        ep = (epoch - (args['n_epochs'] - args['rampdown_length'])) * 0.5
        return math.exp(-(ep * ep) / args['rampdown_length'])
    else:
        return 1.0

print('a1')

mnist = data.mnist(args)

args['shape']     = mnist.img_size
args['n_classes'] = mnist.n_classes
SNTG  = model.SNTG(args)

batch_size       = args['batch_size']
scaled_ratio_max = args['ratio_max']
scaled_ratio_max *= 1.0 * args['n_labeled'] / mnist.n_images

save_path = args['log_dir']
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(np.mean(np.random.standard_normal([100000, 1])))

print('\n\n[*] Start optimization\n')
with tf.Session() as sess:
    SNTG.train()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    tic = time.time()
    is_first = True
    for epoch in range(int(args['n_epochs'])):
        print("\n\nEpoch {0:03d} / {1:03d}\n".format(epoch, int(args['n_epochs'])))
        rampup_value    = rampup(epoch)
        rampdown_value  = rampdown(epoch)
        learning_rate   = rampup_value * rampdown_value * args['lr_max']
        ratio           = rampup_value * scaled_ratio_max
        adam_beta1      = rampdown_value * args['adam_beta1'] + (1.0 - rampdown_value) * args['rampdown_beta1']

        if epoch < 1:
            for m, batch_idx, images1, images2, target, mask in mnist.next_batch(is_training=True):
                print(np.shape(images1))
                ratio = 0.0

                feed_dict = {SNTG.data        : images1,
                             SNTG.data_2      : images2,
                             SNTG.targets     : np.zeros_like(target),
                             SNTG.mask        : np.zeros_like(mask),
                             SNTG.ratio       : ratio,
                             SNTG.lr          : learning_rate,
                             SNTG.adam_beta1  : adam_beta1,
                             SNTG.is_first    : True,
                             SNTG.is_training : True}

                _, _ = sess.run([SNTG.init_feats, SNTG.init_op], feed_dict=feed_dict)
                is_first = False

                break

        total_losses, train_n, sup_losses, graph_losses = 0., 0., 0., 0.
        pos_err, neg_err, n_hardlabel = 0., 0., 0.
        top1_accs = []
        for m, batch_idx, images1, images2, target, mask in mnist.next_batch(is_training=True):
            feed_dict = {SNTG.data        : images1,
                         SNTG.data_2      : images2,
                         SNTG.targets     : target,
                         SNTG.mask        : mask,
                         SNTG.ratio       : ratio,
                         SNTG.lr          : learning_rate,
                         SNTG.adam_beta1  : adam_beta1,
                         SNTG.is_first    : False,
                         SNTG.is_training : True}

            _, loss_CNN, graph_loss, perturbation, sup_loss, pred_cnn, check, check2, check3 = sess.run([SNTG.train_op, SNTG.loss_CNN, SNTG.semi_loss, SNTG.entropy_kl, SNTG.loss_cnn, SNTG.preds_labels_2, SNTG.pos, SNTG.neg, SNTG.check], feed_dict=feed_dict)

            arg_target = np.argmax(mnist.y_train[batch_idx, :], axis=1)
            arg_pred   = np.argmax(pred_cnn, axis=1)
            top1_accs.append(np.mean(arg_target == arg_pred, dtype=np.float32))

            total_losses += loss_CNN * m
            sup_losses   += sup_loss * m
            graph_losses += graph_loss * m
            train_n      += m
            pos_err      += np.mean(check) * m
            neg_err      += np.mean(check2) * m
            n_hardlabel  += check3

        toc = time.time()

        print("[*] total loss:%.4e | sup loss: %.4e | perturbation-loss:%.4e | graph-loss:%.4e" % (total_losses / train_n, sup_losses / train_n, perturbation, graph_losses / train_n))
        print("[*] lr: %.7f | beta1: %f | ratio: %f | time: %f" % (learning_rate, adam_beta1, ratio, toc-tic))
        print("[*] TRAIN Top-1 CNN Acc: %.4f" % (np.mean(top1_accs)))
        print("[*] Pos: %f, NEG: %f, N_hard: %f"%(pos_err, neg_err, n_hardlabel))

        saver.save(sess, save_path + '/cnn_model')
        preds = []
        preds_mc = []

        for m, images1, target in mnist.next_batch(is_training=False):
            feed_dict = {SNTG.data          : images1,
                         SNTG.is_training   : False,
                         SNTG.is_first      : False}

            pred, feats = sess.run([SNTG.test_preds, SNTG.test_feats], feed_dict=feed_dict)
            preds.extend(pred)

        preds = np.asarray(preds)
        arg_pred = np.argmax(preds, axis=1)

        arg_target  = np.argmax(mnist.y_test[:len(arg_pred)], axis=1)
        top1_acc    = np.mean(arg_target == arg_pred, dtype=np.float32)

        print("[*] TEST  Top-1 CNN Acc: {0:0.4f}".format(top1_acc))