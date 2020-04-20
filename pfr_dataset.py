'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''

import os
import os.path
import numpy as np
import sys
from glob import glob
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider


AA_INDEX = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PFRDataset:
    def __init__(
        self,
        root,
        batch_size=32,
        npoints=1024,
        split='train',
        normalize=True,
        normal_channel=True,
        cache_size=15000,
        shuffle=None,
        add_n_c_info=True,
        to_categorical_indexes=[],
        to_categorical_sizes=[]
    ):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.add_n_c_info = add_n_c_info
        if add_n_c_info:
            self.n_c = np.expand_dims(np.array(np.arange(npoints) / npoints), axis=1)
        self.classes_names = [i.strip() for i in os.listdir(self.root)]
        self.classes = dict(zip(self.classes_names, range(len(self.classes_names))))
        self.normal_channel = normal_channel
        self.to_categorical_indexes = to_categorical_indexes
        self.to_categorical_sizes = to_categorical_sizes

        assert split == 'train' or split == 'test'

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(i.split('/')[2], i) for i in glob(self.root + f'/*/{split}/*.npy')]

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train':
                self.shuffle = True
            else:
                self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data):
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

        jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        return provider.shuffle_points(rotated_data)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            if self.normal_channel:
                point_set = np.load(fn[1])[:, :]
            else:
                point_set = np.load(fn[1])[:, :3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        # Take exactly n npoints
        ind = np.arange(point_set.shape[0])
        if len(ind) > self.npoints:
            ind = np.sort(np.random.choice(ind, self.npoints, replace=False))
        else:
            ind = np.sort(np.random.choice(ind, self.npoints, replace=True))

        point_set = point_set[ind, :]

        for cat_ind, cat_size in zip(self.to_categorical_indexes, self.to_categorical_sizes):
            cat = tf.keras.utils.to_categorical(point_set[:, cat_ind], num_classes=cat_size)
            point_set = np.concatenate([point_set[:, :cat_ind], cat, point_set[:, cat_ind+1:]], axis=1)

        if self.add_n_c_info:
            point_set = np.concatenate([point_set, self.n_c], axis=1)

        if self.normalize:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        return self._get_item(0)[0].shape[1]

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath) + self.batch_size - 1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps, cls = self._get_item(self.idxs[i + start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        if augment:
            batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label
