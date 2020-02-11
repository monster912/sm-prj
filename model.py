import tensorflow as tf
import numpy as np
from functools import partial
from tensorflow.contrib.framework.python.ops import add_arg_scope

class SNTG:
    def __init__(self, args):
        # self.lr         = args['lr_init']
        self.l2_lambda   = args['l2_lambda']
        self.ema_decay   = args['ema_decay']
        self.shape       = args['shape']
        self.n_labeled   = args['n_labeled']
        self.random_seed = args['random_seed']
        self.n_classes   = args['n_classes']

        self.data       = tf.placeholder(dtype=tf.float32, shape=(None, self.shape[0], self.shape[1], self.shape[2]), name='inputs')
        self.data_2     = tf.placeholder(dtype=tf.float32, shape=(None, self.shape[0], self.shape[1], self.shape[2]), name='inputs_2')
        self.targets    = tf.placeholder(dtype=tf.float32, shape=(None, self.n_classes), name='target_cnn')
        self.mask       = tf.placeholder(dtype=tf.float32, shape=(None, self.n_classes), name='mask')

        self.features       = tf.placeholder(dtype=tf.float32, shape=(None, 128), name='features')
        self.is_training    = tf.placeholder(dtype=tf.bool, name='is_training')
        self.ratio          = tf.placeholder(dtype=tf.float32)
        self.is_first       = tf.placeholder(dtype=tf.bool, name='is_first')
        self.lr             = tf.placeholder(dtype=tf.float32)
        self.adam_beta1     = tf.placeholder(dtype=tf.float32)

    def build_graph(self):
        self.data_1 = tf.cond(self.is_training,
                              lambda: self.data + tf.random.normal(tf.shape(self.data), mean=0.0, stddev=0.15),
                              lambda: self.data)
        self.data_2 = self.data_2 + tf.random.normal(tf.shape(self.data_2), mean=0.0, stddev=0.15)

        self.feats, self.outputs = self.vggnet(self.data_1)
        self.feats = tf.identity(self.feats, name='train_features')
        _, self.outputs_corrupted = self.vggnet(self.data_2, reuse=True)

        self.preds_labels   = tf.nn.softmax(self.outputs)
        self.preds_labels_2 = tf.nn.softmax(self.outputs_corrupted)

        self.ema = tf.train.ExponentialMovingAverage(self.ema_decay, zero_debias=True)


    def optimization(self):
        self.margin = 1.0

        total_params = 0
        vars_c = [var for var in tf.trainable_variables() if var.name.startswith("CNN")]
        for var in vars_c:
            shape = var.get_shape()
            total_params += np.prod(shape)
        print("Total # of params: {}".format(total_params))

        merged = self.mask * self.targets + (1. - self.mask) * self.preds_labels_2
        pseudo = tf.argmax(merged, axis=1)

        batch_size = tf.shape(self.data)[0]
        self.hardlabel = tf.cast(tf.equal(pseudo[:batch_size//2], pseudo[batch_size//2:]), dtype=tf.float32)

        self.D = tf.reduce_mean(tf.square(self.feats[batch_size//2:] - self.feats[:batch_size//2]), axis=1)#self._pairwise_distances(self.feats[:50], self.feats[50:], squared=True)
        self.D_sq = tf.sqrt(self.D)
        self.check = tf.reduce_sum(self.hardlabel)

        self.pos = self.D * self.hardlabel
        self.neg = (1. - self.hardlabel) * tf.square(tf.maximum(self.margin - self.D_sq, 0))
        self.semi_loss  = self.ratio * 0.0 * tf.reduce_mean(self.pos+self.neg)
        self.entropy_kl = self.ratio * 1.0 * tf.reduce_mean(tf.pow(self.preds_labels - self.preds_labels_2, 2))

        self.loss_cnn = tf.losses.softmax_cross_entropy(onehot_labels=self.targets,
                                                        logits=self.outputs * self.mask)

        self.loss_CNN = self.loss_cnn + self.entropy_kl + self.semi_loss

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        CNN_op = self.adam_updates(params=vars_c, cost_or_grads=self.loss_CNN, lr=self.lr, mom1=self.adam_beta1)
        ema_op = self.ema.apply(vars_c)
        self.train_op = tf.group([CNN_op, ema_op] + self.bn_updates)
        # with tf.control_dependencies([opt_op]):
        #     self.train_op = ema.apply(vars_c)


    def train(self):
        self.bn_updates = []
        self.init_op = []
        self.build_graph()
        self.optimization()

        self.test_feats, self.test_outputs = self.vggnet(self.data_1, reuse=True, ema=self.ema)
        self.init_feats, _ = self.vggnet(self.data_1, reuse=True, init=True)
        self.test_feats = tf.identity(self.test_feats, name='test_features')
        self.test_outputs = tf.identity(self.test_outputs, name='test_outputs')
        self.test_preds = tf.nn.softmax(self.test_outputs, name='test_preds')

    def vggnet(self, images, ema=None, reuse=False, init=False):
        with tf.variable_scope("CNN", reuse=reuse):
            vggnet_conv = partial(self.conv2d,
                                  nonlinearity=tf.nn.leaky_relu,
                                  pad='SAME',
                                  init=init,
                                  ema=ema)

            x = vggnet_conv(images, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], indices=0, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], indices=1, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], indices=2, reuse=reuse)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
            x = tf.layers.dropout(x, training=self.is_training)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], indices=3, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], indices=4, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], indices=5, reuse=reuse)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
            x = tf.layers.dropout(x, training=self.is_training)
            x = vggnet_conv(x, filters=512, kernel_size=[3, 3],
                            strides=[1, 1], indices=6, pad='VALID', init=init, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[1, 1],
                            strides=[1, 1], indices=7, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[1, 1],
                            strides=[1, 1], indices=8, reuse=reuse)

            # Global Average Pooling
            feats = tf.reduce_mean(x, axis=[1, 2])

            outputs = self.dense(feats, num_units=self.n_classes, nonlinearity=None, init=init,
                                 ema=ema, indices=0, reuse=reuse)

        return feats, outputs

    def destroy_graph(self):
        tf.reset_default_graph()

    @add_arg_scope
    def conv2d(self, x_, filters, kernel_size=[3, 3], strides=[1, 1], pad='SAME',
               nonlinearity=None, init_scale=1., counters={}, init=False, ema=None,
               reuse=False, indices=1, **kwargs):
        ''' convolutional layer '''
        name = 'conv2d_{}'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('V', ema, shape=kernel_size + [int(x_.get_shape()[-1]), filters], dtype=tf.float32,
                                       initializer=tf.initializers.truncated_normal(stddev=0.05, seed=self.random_seed), trainable=True)
            g = self.get_var_maybe_avg('g', ema, shape=[filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.), trainable=True)
            b = self.get_var_maybe_avg('b', ema, shape=[filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.), trainable=True)
            self.batch_mean = tf.get_variable('batch_mean', shape=[filters], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.0),
                                              trainable=False)
            # use weight normalization (Salimans & Kingma, 2016)
            W = V * tf.reshape(g, [1, 1, 1, filters]) / tf.sqrt(tf.reduce_sum(tf.square(V), axis=(0, 1, 2), keepdims=True))

            # calculate convolutional layer output
            x = tf.nn.conv2d(x_, W, [1] + strides + [1], pad)

            # mean-only batch-norm.
            if ema is not None:
                x = x - self.batch_mean
            else:
                m = tf.reduce_mean(x, axis=[0, 1, 2])
                x = x - tf.reshape(m, [1, 1, 1, filters])
                bn_update = self.batch_mean.assign(self.ema_decay*self.batch_mean + (1.0-self.ema_decay)*m)

                # x, bn_update, m_init = mean_only_batch_norm()
                if not reuse:
                    self.bn_updates.append(bn_update)

                if init:
                    stdv = tf.sqrt(tf.reduce_mean(tf.square(x), axis=(0, 1, 2)))
                    scale_init = init_scale / stdv
                    x *= tf.reshape(scale_init, [1, 1, 1, filters])
                    self.init_op.append(g.assign(g * scale_init))

            x = tf.nn.bias_add(x, b)

             # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x, alpha=0.1)

            return x


    @add_arg_scope
    def dense(self, x_, num_units, nonlinearity=None, init_scale=1., counters={},
              init=False, ema=None, reuse=False, indices=0, **kwargs):
        ''' fully connected layer '''
        # name = self.get_name('dense', counters)
        name = 'dense_'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('V', ema, shape=[int(x_.get_shape()[1]), num_units], dtype=tf.float32,
                                       initializer=tf.initializers.truncated_normal(stddev=0.05, seed=self.random_seed), trainable=True)
            g = self.get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.), trainable=True)
            b = self.get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                                       initializer=tf.initializers.constant(0., dtype=tf.float32), trainable=True)
            self.batch_mean = tf.get_variable('batch_mean', shape=[num_units], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.0),
                                              trainable=False)
            # use weight normalization (Salimans & Kingma, 2016)
            W = V * g / tf.sqrt(tf.reduce_sum(tf.square(V), axis=[0]))

            x = tf.matmul(x_, W)

            if ema is not None:
                x = x - self.batch_mean
            else:
                m = tf.reduce_mean(x, axis=[0])
                x = x - m
                bn_update = self.batch_mean.assign(self.ema_decay * self.batch_mean + (1.0 - self.ema_decay) * m)
                # x, bn_update = mean_only_batch_norm()
                if not reuse:
                    self.bn_updates.append(bn_update)

                if init:
                    stdv = tf.sqrt(tf.reduce_mean(tf.square(x), axis=[0]))
                    scale_init = init_scale / stdv
                    x /= stdv
                    self.init_op.append([g.assign(g * scale_init)])

            # apply nonlinearity
            # if nonlinearity is not None:
            #     x = nonlinearity(x)
            return x + tf.reshape(b, [1, num_units])

    def get_name(self,layer_name, counters):
        ''' utlity for keeping track of layer names '''
        if not layer_name in counters:
            counters[layer_name] = 0
        name = layer_name + '_' + str(counters[layer_name])
        counters[layer_name] += 1
        return name

    def get_var_maybe_avg(self, var_name, ema, **kwargs):
        ''' utility for retrieving polyak averaged params '''
        v = tf.get_variable(var_name, **kwargs)

        if ema is not None:
            v = ema.average(v)
        return v

    def adam_updates(self, params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
        ''' Adam optimizer '''
        updates = []
        if type(cost_or_grads) is not list:
            grads = tf.gradients(cost_or_grads, params)
        else:
            grads = cost_or_grads

        t = tf.Variable(1., 'adam_t', dtype=tf.float32)
        coef = lr * tf.sqrt(1 - mom2 ** t) / (1 - mom1 ** t)
        for p, g in zip(params, grads):
            mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
            v  = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')

            mg_t = mom1 * mg + (1. - mom1) * g
            v_t  = mom2 * v + (1. - mom2) * tf.square(g)
            g_t  = mg_t / (tf.sqrt(v_t) + 1e-8)
            p_t  = p - coef * g_t

            updates.append(mg.assign(mg_t))
            updates.append(v.assign(v_t))
            updates.append(p.assign(p_t))
        updates.append(t.assign_add(1))
        return tf.group(*updates)

