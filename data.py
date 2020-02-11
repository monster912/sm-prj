import numpy as np
import os
import utils

class mnist():
    def __init__(self, args):
        self.rand_seed = np.random.RandomState(args['random_seed'])
        print("Random seed: {0}".format(args['random_seed']))

        self.batch_size              = args['batch_size']
        self.n_labeled               = args['n_labeled']
        self.dataset                 = args['dataset']
        self.aug_trans               = args['augment_translation']
        self.max_unlabeled_per_epoch = args['max_unlabeled_per_epoch']
        self.augment_mirror          = args['augment_mirror']

        # self.max_unlabeled_per_epoch = args['max_unlabeled_per_epoch']

        x_train, y_train, x_test, y_test \
            = load_hynix(args['data_dir'], args)

        self.n_classes = len(np.unique(y_train))
        self.img_size  = np.shape(x_train)[1:]
        print(self.img_size)

        if args['whiten_norm'] == 'norm':
            x_train = whiten_norm(x_train)
            x_test  = whiten_norm(x_test)
        elif args['whiten_norm'] == 'zca':
            whitener = utils.ZCA(x=x_train)
            x_train = whitener.apply(x_train)
            x_test  = whitener.apply(x_test)

        else:
            print("Unkonwon input whitening mode {}".format(args['whiten_norm']))
            exit()

        num_classes = len(set(y_train))
        y_train = np.concatenate([y_train, np.zeros(len(x_train)-self.n_labeled, dtype=np.int32)], axis=0)
        y_test  = y_test

        mask_train = np.concatenate([np.ones([self.n_labeled, num_classes]),
                                     np.zeros([len(x_train) - self.n_labeled, num_classes])], axis=0)

        p = args['augment_translation']
        if p > 0:
            x_train = np.pad(x_train, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')
            x_test  = np.pad(x_test, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')


        # Random Shuffle.
        indices = np.arange(len(x_train))
        self.rand_seed.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]
        mask_train = mask_train[indices]

        # Corrupt some of labels if needed.

        ####################################

        # Reshuffle
        indices = np.arange(len(x_train))
        self.rand_seed.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]
        mask_train = mask_train[indices]

        # Construct mask_train. It has a zero when label is unknown, otherwise one.

        self.train_mask = mask_train

        self.x_train = x_train
        print(y_train)
        self.y_train = one_hot(y_train) * mask_train
        self.x_test  = x_test
        self.y_test  = one_hot(y_test)


        self.n_images   = np.shape(self.x_train)[0]
        self.n_t_images = np.shape(self.x_test)[0]

        self.labeled_idx   = np.where(self.train_mask[:, 0] == 1)[0]#np.asarray(self.labeled_idx)
        self.unlabeled_idx = np.where(self.train_mask[:, 0] == 0)[0]#np.setdiff1d(np.arange(self.n_images), self.labeled_idx)#np.random.choice(self.n_images, args['n_unlabel'], replace=False)

        self.test_mask = np.ones_like(self.y_test)

        self.sparse_label = self.train_mask * self.y_train
        self.sparse_label = np.asarray(self.sparse_label, dtype=np.float32)

    def next_batch(self, is_training):
        if is_training:
            crop = self.aug_trans

            if self.max_unlabeled_per_epoch == "None":
                indices = np.arange(self.n_images)

            self.rand_seed.shuffle(indices)

            n_xl = self.img_size[0]
            for start_idx in range(0, self.n_images, self.batch_size):
                if start_idx + self.batch_size <= self.n_images:
                    excerpt = indices[start_idx : start_idx + self.batch_size]
                    noisy_a, noisy_b = [], []
                    for img in self.x_train[excerpt]:
                        if self.augment_mirror == "True" and self.rand_seed.uniform() > 0.5:
                            img = img[:, ::-1, :]

                        # if self.augment_mirror == "True" and self.rand_seed.uniform() > 0.5:
                        #     img = img[::-1, :, :]

                        t = self.aug_trans
                        ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                        ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                        img_a = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                        ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                        ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                        img_b = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                        noisy_a.append(img_a)
                        noisy_b.append(img_b)

                    yield len(excerpt), excerpt, np.asarray(noisy_a), np.asarray(noisy_b), self.sparse_label[excerpt], self.train_mask[excerpt]

        else:
            indices = np.arange(self.n_t_images)
            crop = self.aug_trans
            n_xl = self.img_size[0]
            batch_size = 50

            for start_idx in range(0, self.n_t_images, batch_size):
                if start_idx + batch_size <= self.n_t_images:
                    excerpt = indices[start_idx: start_idx + batch_size]

                    yield len(excerpt), self.x_test[excerpt, crop: crop+n_xl, crop: crop+n_xl, :], self.y_test[excerpt]



def load_hynix(data_path, args):
    import h5py
    path = os.path.join(data_path, '{}x{}_seed{}.h5py'.format(args['shape'],args['shape'],args['random_seed']))

    hynix_data = h5py.File(path, 'r')

    x_train, y_train, x_test, y_test = hynix_data['x_train'][:], hynix_data['y_train'][:], hynix_data['x_test'][:], hynix_data['y_test'][:]
    img_size = np.shape(x_train)[1:]

    x_train = x_train.reshape([-1, img_size[0], img_size[1], img_size[2]]).astype('float32') / 255.0
    x_test  = x_test.reshape([-1, img_size[0], img_size[1], img_size[2]]).astype('float32') / 255.0
    y_train = y_train.astype('int32')
    y_test  = y_test.astype('int32')

    return x_train, y_train, x_test, y_test

def whiten_norm(x):
    x = x - np.mean(x, axis=(1, 2, 3), keepdims=True)
    x = x / (np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)
    return x

def one_hot(x):
    n_values = np.max(x) + 1

    return np.eye(n_values)[x]

