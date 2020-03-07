import time
from AC_brain import PPO
from GoalEnvironment import GoalEnvironment

env = GoalEnvironment('Acrobot-v1', render=True)

policy = PPO()
policy.set_bounds(env.HIGH, env.LOW)
policy.restore('data/models/pretrain/brain')
step = 0

while True:
    if step > 0:
        print('Environment is done in %d steps' % step)
    done = False
    s = env.reset()
    step = 0
    while not done and step <= 500:
        s, _, done,  k = env.step(policy.choose_action(s))
        step += k
        time.sleep(.08)
