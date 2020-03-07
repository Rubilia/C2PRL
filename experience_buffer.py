import random
import os
from typing import Tuple
import numpy as np

from Executor.utils import PART_TO_REMOVE, R_DIM, S_DIM, Path, A_DIM

A_percentage = 0.2
Statistics = 50000
SAVE_FREQ = 32000


class ReplayBuffer(object):

    def __init__(self, max_negative=10 ** 6, max_positive=10 ** 6, batch_size=1024, s_dim=0, a_dim=0):
        self.max_negative = max_negative
        self.max_positive_size = max_positive
        self.batch_size = batch_size
        self.s_dim = s_dim
        self.r_dim = R_DIM
        self.a_dim = a_dim
        self.positive_data = DataStorage(Path, max_positive, R_DIM, a_dim, R_DIM, scope='positive_', r=0.)
        self.positive_data.restore_data(Path)
        self.negative_data = DataStorage(Path, max_negative, R_DIM, a_dim, R_DIM, scope='negative_', r=-1.)
        self.negative_data.restore_data(Path)
        self.meta_data = []

    def add(self, experience):
        if (self.negative_data.Cursor + self.positive_data.Cursor + 1) % SAVE_FREQ == 0:
            self.positive_data.save_data(Path)
            self.negative_data.save_data(Path)
            print('Learning data successfully saved')
        if experience[2] == 0.:
            self.positive_data.add_item(experience)
            self.meta_data.append(1.)
        else:
            self.negative_data.add_item(experience)
            self.meta_data.append(0.)

        if len(self.meta_data) >= Statistics:
            self.meta_data = self.meta_data[Statistics // PART_TO_REMOVE:]
        del experience

    def size(self):
        return self.positive_data.size()

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # sample negative_data
        perc = min(.9, max(A_percentage, np.mean(self.meta_data)))

        s1, a1, r1, s_1, g1, d1 = self.negative_data.sample(int(self.batch_size * (1 - perc)))
        s2, a2, r2, s_2, g2, d2 = self.positive_data.sample(self.batch_size - int(self.batch_size * (1 - perc)))

        S = np.append(s1, s2)
        A = np.append(a1, a2)
        R = np.append(r1, r2)
        S_ = np.append(s_1, s_2)
        G = np.append(g1, g2)
        D = np.append(d1, d2)

        return S.reshape((-1, R_DIM)), A.reshape((-1, A_DIM)), R, S_.reshape((-1, R_DIM)), G.reshape((-1, R_DIM)), D


class DataStorage:
    def __init__(self, path, size, s_dim, a_dim, r_dim, scope='', r=0.):
        self.scope = scope
        self.path = path
        self.max_size = size
        self.a_dim = a_dim
        self.r_dim = r_dim
        self.s_dim = s_dim
        self.is_full = False
        self.r = r
        self.S = np.memmap(path + scope + 'States.dat', dtype=np.float, mode='r+' if os.path.exists(path + scope +
                                                                        'States.dat') else 'w+', shape=(size, s_dim))
        self.A = np.memmap(path + scope + 'Actions.dat', dtype=np.float, mode='r+' if os.path.exists(path + scope + 'Actions.dat') else 'w+', shape=(size, a_dim))
        self.S_ = np.memmap(path + scope + 'Next_states.dat', dtype=np.float, mode='r+' if os.path.exists(path + scope + 'Next_states.dat') else 'w+', shape=(size, s_dim))
        self.G = np.memmap(path + scope + 'Goals.dat', dtype=np.float, mode='r+' if os.path.exists(path + scope + 'Goals.dat') else 'w+', shape=(size, r_dim))
        self.D = np.memmap(path + scope + 'Dones.dat', dtype=np.float, mode='r+' if os.path.exists(path + scope + 'Dones.dat') else 'w+', shape=(size,))
        self.Cursor = 0

    def size(self):
        return self.max_size if self.is_full else self.Cursor

    def add_item(self, experience):
        self.S[self.Cursor] = list(experience[0].flatten())
        self.A[self.Cursor] = list(experience[1].flatten())
        self.S_[self.Cursor] = list(experience[3].flatten())
        self.G[self.Cursor] = list(experience[4].flatten())
        self.D[self.Cursor] = experience[5]
        self.Cursor += 1

        if self.Cursor >= self.max_size:
            self.Cursor = 0
            self.is_full = True

    def sample(self, batch):
        max_id = self.max_size if self.is_full else self.Cursor
        ids = [np.random.randint(0, max_id) for _ in range(batch)]
        S = [np.copy(self.S[i]) for i in ids]
        A = [np.copy(self.A[i]) for i in ids]
        R = [self.r for _ in ids]
        S_= [np.copy(self.S_[i]) for i in ids]
        G = [np.copy(self.G[i]) for i in ids]
        D = [np.copy(self.D[i]) for i in ids]
        return S, A, R, S_, G, D

    def save_data(self, path):
        data = np.array([self.Cursor, self.is_full])
        np.save(path + self.scope + 'metadata', data)

    def restore_data(self, path):
        try:
            data = np.load(path + self.scope + 'metadata.npy')
            self.Cursor = data.item(0)
            self.is_full = data.item(1)
            print('Learning data has been restored')
        except Exception:
            print('Learning data has not been restored')
