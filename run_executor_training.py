import os
from threading import Thread

import numpy as np
import tensorflow as tf

from Executor.agent import Agent
from Executor.environment import ExecutorEnvironment
from Executor.state_representer import VariantionalAutoEncoder
from Executor.utils import STATE_REPRESENTATION, get_available_gpus

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAX_STEPS = 50000
TEST_FREQ = 40
SAVE_FREQ = 100
TEST_EPISODES = 5
ENV_NAME = 'AirRaid-v0'
GLOBAL_STEP = 0
NUM_THREADS = 1
AC_AGENT = True
render = True
LEARNING = False


def get_thread(agent, compressor):
    def thread():
        return Thread(target=agent.train, args=(compressor,))

    return thread


def test(agent, compressor=None, wait: float = 0.):
    agent.test = True
    goal_counter = 0
    for episode in range(TEST_EPISODES):
        success, _ = agent.train(compressor=compressor, wait=wait)
        goal_counter += success
    goal_counter /= TEST_EPISODES
    print(
        '########################\nTesting agent: avg goals reached in episode: %f\n########################' % goal_counter)
    if LEARNING:
        agent.test = False


def collect_data(threads: list):
    T = []
    for t in threads:
        T.append(t())
        T[-1].start()

    for t in T:
        t.join()


def run_training(agent: Agent, Agents, threads, compressor=None):
    global GLOBAL_STEP
    if not LEARNING:
        agent.test = False

    while GLOBAL_STEP < MAX_STEPS:
        GLOBAL_STEP += 1
        step = GLOBAL_STEP
        # Save agent with given frequency
        if GLOBAL_STEP % SAVE_FREQ == 0 and GLOBAL_STEP > 0:
            # Save agent
            path = os.getcwd()
            path = path[:-(len(os.path.basename(path)) + 1)]
            path += '/data/models/executor/executor_' + str(step) + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            agent.save(path + 'executor')
            agent.restore(path + 'executor')
            print('Executor network saved')

        # Test performance every TEST_FREQ steps
        if GLOBAL_STEP % (TEST_FREQ // NUM_THREADS) == 0 and GLOBAL_STEP > 0:
            test(agent, compressor=compressor)

        collect_data(threads)

        success, data = agent.train(compressor)
        # Update target weights
        for A in Agents:
            A.set_params(agent)

        print('\n%f%% - reached %d goals during episode' % (100 * step / MAX_STEPS, success))
        if not data == '':
            print(data)


def compressor_data_collector(compressor, episodes, env):
    for i in range(episodes):
        if compressor.is_full:
            print('Data collected')
            return

        if (i + 1) % 10 == 0:
            print('Iteration #%d, %d samples has been collected' % (i + 1, compressor.Cursor))
            compressor.save_data()
        done = False
        s = env.reset()
        while not done:
            a = np.random.uniform(env.low, env.high)
            s, _, done = env.step(a)
            compressor.add_data(s)
            # compressor.create_correct_images(100, 100)


def main():
    if STATE_REPRESENTATION:
        sess_compressor = tf.Session()
        compressor = VariantionalAutoEncoder(sess_compressor)
        sess_compressor.run(tf.global_variables_initializer())
    else:
        compressor = None

    env = ExecutorEnvironment(ENV_NAME, render=render, compressor=compressor)

    # Create global agent
    sess = tf.Session()
    GLOBAL_AGENT = Agent(sess=sess, env=env, compressor=compressor, scope='main', batch_size=1024,
                         GLOBAL=True, device='gpu:0', buffer_size=10 ** 4, is_AC=AC_AGENT)
    GLOBAL_AGENT.test = LEARNING

    sess.run(tf.global_variables_initializer())

    try:
        path = os.getcwd()
        path = path[:-(len(os.path.basename(path)) + 1)]
        path += '\\data\\models\\pretrain\\'
        GLOBAL_AGENT.restore(path)
        print('Executor model restored')

    except Exception as e:
        print('Executor restoration failed')

    # restore compressor
    if STATE_REPRESENTATION:
        try:
            path = os.getcwd()
            path = path[:-(len(os.path.basename(path)) + 1)]
            path += '\\data\\models\\pretrain\\'
            compressor.restore(path)
            compressor.clear_data()

            # compressor.restore_data()
            # for _ in range(1000):
            #     compressor.create_correct_images(100, 100)
            #     compressor.create_images(100, 100)

            print('\nCompressor model restored\n')
        except Exception:
            print('\nCompressor restoration failed\n')
            if not compressor.restore_data():
                # pretrain compressor on random data
                compressor.start_learning()
                print('Collecting data for compressor\n')
                compressor_data_collector(compressor, 16000, env)
                compressor.save_data()

            # for _ in range(1000):
            #     compressor.create_images(100, 100)

            print('Training compressor\n')
            for t in range(400):
                loss = compressor.learn(8)
                print(loss)
                if (t+1) % 10 == 0:
                    print('Iteration #%d' % (t+1))
                if (t+1) % 100 == 0:
                    path = os.getcwd()
                    path = path[:-(len(os.path.basename(path)) + 1)]
                    path += '/data/models/compressor/compressor0/'
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    compressor.save(path + 'compressor')
                    print('Compressor saved')
            compressor.stop_learning()
            print('Compressor is successfully pretrained\n')
            path = os.getcwd()
            path = path[:-(len(os.path.basename(path)) + 1)]
            path += '/data/models/compressor/compressor0/'
            if not os.path.isdir(path):
                os.mkdir(path)
            compressor.save(path + 'compressor')
            print('\nRepresenter saved')
            for _ in range(10):
                compressor.create_images(100, 100)
    if LEARNING:
        # Create parallel agents
        Agents = []
        Threads = []

        devices = get_available_gpus() + ['gpu:0']

        for i in range(NUM_THREADS):
            # Instantiate agent with its own environment
            sess = tf.Session()
            agent = Agent(sess=sess, env=ExecutorEnvironment(ENV_NAME, render=False, compressor=compressor), buffer=GLOBAL_AGENT.buffer,
                          scope='thread_%d_' % i, device=devices[i % max(1, len(devices))], is_AC=AC_AGENT,
                          compressor=compressor)
            agent.test = LEARNING
            sess.run(tf.global_variables_initializer())
            agent.set_params(GLOBAL_AGENT)
            Agents.append(agent)
            # Create Thread for agent
            Threads.append(get_thread(agent, compressor))

        run_training(agent=GLOBAL_AGENT, Agents=Agents, threads=Threads, compressor=compressor)
    else:
        # Test executor`s accuracy
        while True:
            test(GLOBAL_AGENT, compressor, wait=20/1000)


if __name__ == '__main__':
    main()
