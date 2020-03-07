import numpy as np

from Executor.utils import is_goal_reached


class DHERReplay(object):
    def __init__(self, size, evaluation_size=4096):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of episodes to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self.max_ep_len_dher = 1000
        self.episode_data = []
        self.curr_ep = -1
        self._maxsize = size
        self.evaluation_size = evaluation_size
        self._next_idx = 0
        self.episode_out_ratio = 3

    def __len__(self):
        return len(self._storage)

    def start_episode(self):
        self.curr_ep += 1

    def add(self, s, a, r, s_, g, done):
        data = (s, a, r, s_, g, done)

        if len(self.episode_data) >= self._maxsize:
            self.episode_data.__delitem__(0)
            self._storage.__delitem__(0)
        self._storage.append(data)
        self.episode_data.append(self.curr_ep)

    def sample_episode(self, ep_id):
        data = []
        id = self.episode_data.index(ep_id)
        while self.episode_data[id] == ep_id and len(data) <= self.max_ep_len_dher:
            data.append(self._storage[id])
            id += 1
            if id == len(self.episode_data): break
        return data, [ep_id for _ in range(len(data))]

    def _encode_sample(self, batch_size):
        assert batch_size > 0 and batch_size % 2 == 0
        if batch_size > len(set(self.episode_data)): return []
        samples = 0
        sampled_batch = []
        while samples < batch_size:
            data = []
            ep_data = []
            while len(data) < self.evaluation_size:
                id = np.random.randint(self.episode_data[0], self.episode_data[-1] + 1)
                if ep_data.__contains__(id): continue
                sub_data, sub_ep_data = self.sample_episode(id)
                data.extend(sub_data)
                ep_data.extend(sub_ep_data)
            batch, smpls = self._sample_dher(data, ep_data, batch_size)
            sampled_batch.extend(batch)
            samples += smpls
        return sampled_batch

    def sample(self, batch_size):
        return self._encode_sample(batch_size)

    @staticmethod
    def _sample_dher(data, ep_data, batch_size):
        """Creates a sample function that can be used for HER experience replay.
        """

        def build_hash_table(data):
            achieved_hash = {}
            desired_hash = {}
            # Contains id in dataset, episode number and id in episode
            for idx in range(len(data) - 1, -1, -1):
                achieved = data[idx][3]
                ep = ep_data[idx]
                ep_id = idx - ep_data.index(ep)
                achieved_hash[tuple(achieved)] = idx, ep, ep_id
            reverse = ep_data[::-1]
            episodes = set(ep_data)
            for ep in episodes:
                last_id = len(ep_data) - 1 - reverse.index(ep)
                goals_data = find_different_goals(ep, last_id)
                for idx in goals_data:
                    desired = data[idx][4]
                    ep_id = idx - ep_data.index(ep)
                    desired_hash[tuple(desired)] = idx, ep, ep_id

            return achieved_hash, desired_hash

        def find_different_goals(ep, last_id):
            goal_ids = [last_id]
            t = last_id
            g = data[t][1]
            while t >= 0 and ep_data[t - 1] == ep:
                t -= 1
                if np.linalg.norm(data[t][4] - g) == 0: continue
                g = data[t][4]
                goal_ids.append(t)
            return goal_ids

        def search_memory(data):
            intersection_set = []
            achieved_hash, desired_hash = build_hash_table(data)
            for key, val in achieved_hash.items():
                temp_ach = key
                ep = val[1]
                ep_id = val[-1]
                if ep_id == 0: continue
                B = find(desired_hash, np.array(temp_ach), ep)
                if B is not None:
                    de_idx = B
                    intersection_set.append((val, de_idx))
            return intersection_set

        def find(d, item, ep):
            values = list(d.values())
            for id, t in enumerate(d):
                if ep_data[id] == ep or values[id][-1] == 0: continue
                if is_goal_reached(t, item):
                    return id, values[id][1], values[id][2]
            return None

        def get_goal_trajectory(p, q, id0):
            goals = []
            m = min(p, q)
            for t in range(m + 1):
                goals.append(data[id0 + q - m + t][4])
            return goals

        intersection = search_memory(data)
        if len(intersection) == 0:
            return [], 0

        samples = 0
        sample_batch = []
        for k in range(min(batch_size, len(intersection))):
            episode = []
            ac, de = intersection[np.random.randint(0, len(intersection))]
            ac_id, i, p = ac
            de_id, j, q = de
            id0_i = ac_id - p
            id0_j = de_id - q
            m = min(p, q)
            if m == 0:
                k -= 1
                continue
            goals = get_goal_trajectory(p, q, id0_j)
            for t_ in range(0, m + 1):
                s, a, r, s_, g, done = data[p - m + t_ + id0_i]
                g = goals[t_]
                done = is_goal_reached(s_, g)
                r = done - 1.
                episode.append((s, a, r, s_, g, done))
            samples += len(episode)
            sample_batch.append(episode)
        return sample_batch, samples

    @property
    def storage(self):
        return self._storage

