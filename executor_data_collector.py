import os
import numpy as np
import tensorflow as tf

from Executor.environment import ExecutorEnvironment
from Executor.run_executor_training import ENV_NAME
from Executor.state_representer import VariantionalAutoEncoder

path = os.getcwd()
path = path[:-(len(os.path.basename(path)) + 1)]
path += '/data/models/compressor/compressor0/'

env = ExecutorEnvironment(ENV_NAME, render=False)


def compressor_data_collector(compressor, episodes, env):
    for i in range(episodes):
        done = False
        s = env.reset()
        compressor.add_data(s)
        t = False
        while not done:
            a = np.random.uniform(env.low, env.high)
            s, _, done = env.step(a)
            if t:
                compressor.add_data(s)
            else: t = True



sess_compressor = tf.Session()
compressor = VariantionalAutoEncoder(sess_compressor)
sess_compressor.run(tf.global_variables_initializer())

compressor_data_collector(compressor, 1000, env)

compressor.save_data()

for t in range(8000):
    loss = compressor.learn(16)
    print(loss)
    if (t+1) % 200 == 0:
        print('Step #%d' % (t+1))

compressor.save(path + 'compressor')

compressor.create_images(100, 100)
compressor.create_correct_images(100, 100)
