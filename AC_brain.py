import queue
import threading
import time
import numpy as np
import tensorflow as tf
from GoalEnvironment import GoalEnvironment

EP_MAX = 800000
EP_LEN = 800
N_WORKER = 1
GAMMA = 0.99
A_LR = 0.0001
C_LR = 0.0002
MIN_BATCH_SIZE = 64
UPDATE_STEP = 10
EPSILON = 0.02
GAME = 'Pong-v0'
S_DIM, A_DIM = 32, 32
IS_AC = True
RENDER = True
# Agent will be tested if set to false
Learning = False


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 160, tf.nn.relu)
        l2 = tf.layers.dense(l1, 160, tf.nn.relu)
        self.v = tf.layers.dense(l2, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def set_bounds(self, high, low):
        self.HIGH = high
        self.LOW = low

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()
                self.sess.run(self.update_oldpi_op)
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()
                GLOBAL_UPDATE_COUNTER = 0
                ROLLING_EVENT.set()

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu, trainable=trainable)
            l3 = tf.layers.dense(l2, 64, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l3, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l3, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        if not s.shape.__len__() == 2:
            s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, self.LOW, self.HIGH)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)


class Worker(object):
    def __init__(self, wid, env):
        self.wid = wid
        self.env = env
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, Learning
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                    if not ROLLING_EVENT.is_set():
                        ROLLING_EVENT.wait()
                        buffer_s, buffer_a, buffer_r = [], [], []
                    a = self.ppo.choose_action(s)
                    s_, r, done, _ = self.env.step(a)
                    if Learning:
                        buffer_s.append(s)
                        buffer_a.append(a)
                        buffer_r.append(r)
                    s = s_
                    ep_r += r

                    GLOBAL_UPDATE_COUNTER += 1
                    if (t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done) and Learning:
                        v_s_ = self.ppo.get_v(s_)
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + GAMMA * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                        buffer_s, buffer_a, buffer_r = [], [], []
                        QUEUE.put(np.hstack((bs, ba, br)))
                        if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                            ROLLING_EVENT.clear()
                            UPDATE_EVENT.set()

                        if GLOBAL_EP >= EP_MAX:
                            COORD.request_stop()
                            break
                    if done: break

            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )
        if Learning:
            GLOBAL_PPO.save('data/models/pretrain/brain')


if __name__ == '__main__':
    env = GoalEnvironment(GAME, render=RENDER, is_ac=IS_AC)
    GLOBAL_PPO = PPO()
    GLOBAL_PPO.set_bounds(env.HIGH, env.LOW)
    try:
        GLOBAL_PPO.restore('data/models/pretrain/brain')
    except Exception: pass
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    workers = [Worker(wid=i, env=env) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()
    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)
    threads.append(threading.Thread(target=GLOBAL_PPO.update))
    threads[-1].start()
    time.sleep(10)
    COORD.join(threads)

    GLOBAL_PPO.save('data/models/pretrain/brain')
    print('Saved')
    env.set_render(True)
    while True:
        s = env.reset()
        for t in range(500):
            s, _, done, _ = env.step(GLOBAL_PPO.choose_action(s))
            if done: break
