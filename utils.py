import types

import cv2
import numpy as np
from typing import Tuple
from tensorflow.python.client import device_lib

Atari = ['Pong-v0', 'Breakout-v0', 'SpaceInvaders-v0', 'AirRaid-v0']


def configure_training(env: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, types.LambdaType, types.LambdaType, int]:
    """
    :param env: name of the environment :rtype: tuple consists of goal_space, highest_action, lowest_action, noise,
    max_steps_per_episode, state_function_preprocessor, state_array_post_processor, action_repeat
    """
    global Atari
    if env in Atari:
        goal_space = np.array([[0, 1] for _ in range(R_DIM)])
        highest_action = np.array([1., 1., 1., 1., 1., 1.]) * 1e10
        lowest_action = np.array([-1., -1., -1., -1., -1., -1.]) * 1e10
        noise = np.array([.0])
        max_steps_per_episode = 10000
        return goal_space, highest_action, lowest_action, noise, max_steps_per_episode, preprocessing, \
               atari_post_processing, 1
    else: raise Exception('Environment {} has no parameters to run'.format(env))


def preprocessing(item: list) -> np.ndarray:
    img = cv2.cvtColor(item, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (210, 160))
    img = cv2.resize(img, (84, 110))
    img = img[18:102, :]
    img = cv2.resize(img, tuple(TARGET_R_SHAPE))
    return np.array(img, dtype=np.float32)/255.


def atari_post_processing(data, env):
    x = np.array(data).transpose([1, 2, 0])
    return x if not STATE_REPRESENTATION else env.compressor.compress_state(x)


# Contains networks configurations
def get_executor_config(subj: str):
    if subj == 'actor_network':
        return [32, 16]
    elif subj == 'critic_network':
        return [32, 16]
    elif subj == 'state_compressor':
        # filters, kernel, stride, padding
        encoder_config = [(32, 8, 4, 'VALID'), (64, 4, 2, 'VALID'), (32, 3, 1, 'VALID')]
        decoder_config = encoder_config[::-1]

        dense_encoder = [512, 32]
        dense_decoder = [32, 512, 1568]
        return encoder_config, decoder_config, dense_encoder, dense_decoder
    elif subj == 'q_network':
        return [64, 64, 32, 8]


# def is_goal_reached(s, g):
#     return np.linalg.norm(s - g) <= THRESHOLD

def is_goal_reached(s, g):
    for S, G, Thr in zip(s.flatten(), g, THRESHOLD):
        if abs(S - G) > Thr: return False
    return True


def process_done(done: bool, env) -> bool:
    global Atari
    if env.env_name == 'Pendulum-v0':
        return env.step_counter >= env.max_steps
    elif env.env_name == 'MountainCarContinuous-v0':
        return done or env.step_counter < env.max_steps
    elif env.env_name == 'Acrobot-v1':
        return env.step_counter >= env.max_steps or done
    elif 'CartPole' in env.env_name:
        return env.step_counter >= env.max_steps or done
    elif env.env_name in Atari:
        return env.step_counter >= env.max_steps or done


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


PART_TO_REMOVE = 8

DEFAULT_DEVICE = 'gpu:0'

# Path to save data from experience replay
Path = 'D:\\Project_data\\training_datasets\\'

TIME_BTWN_FRAMES = .02

# Agent config (depends on environment)

RANDOM_ACTIONS_PERCENT = .1

GOALS_TO_REPLAY = 4

THRESHOLD = [.2, .2, .2, .2]

# THRESHOLD = .32

SHOW_GOALS = False

N_FRAMES = 4

# DHER_BATCH = 64

# If STATE_REPRESENTATION = False, then R_DIM must be equal S_DIM
STATE_REPRESENTATION = True

R_DIM = 32

TARGET_R_SHAPE = [84, 84]

# Depends on the current environment

S_DIM = TARGET_R_SHAPE[0] * TARGET_R_SHAPE[1]

A_DIM = 6

