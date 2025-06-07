import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarObstacleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.size = 10
        self.observation_space = spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.start = np.array([0.0, 0.0])
        self.goal_min = np.array([9.0, 9.0])
        self.goal_max = np.array([10.0, 10.0])
        self.obstacles = [
            (np.array([1.0, 1.0]), np.array([4.0, 2.0])),
            (np.array([0.0, 4.0]), np.array([3.0, 5.0])),
            (np.array([6.0, 4.0]), np.array([7.0, 8.0])),
            (np.array([8.0, 0.0]), np.array([9.0, 2.5]))
        ]

        self.state = self.start.copy()

    def reset(self, *, seed=None, options=None):
        self.state = self.start.copy()
        return self.state.copy(), {}

    def step(self, action):
        move = {
            0: np.array([0, 1]),    # ↑
            1: np.array([0, -1]),   # ↓
            2: np.array([-1, 0]),   # ←
            3: np.array([1, 0]),    # →
            4: np.array([-1, 1]),   # ↖
            5: np.array([1, 1]),    # ↗
            6: np.array([-1, -1]),  # ↙
            7: np.array([1, -1])    # ↘
        }[action]

        self.state += move * 0.5  # 이동 거리를 float로 유지
        self.state = np.clip(self.state, 0.0, float(self.size))

        reward = -1
        done = False

        for obs_min, obs_max in self.obstacles:
            if self._in_box(self.state, obs_min, obs_max):
                reward = -100
                done = True

        if self._in_box(self.state, self.goal_min, self.goal_max):
            reward = 100
            done = True

        return self.state.copy(), reward, done, False, {}

    def _in_box(self, pos, box_min, box_max):
        return np.all(pos >= box_min) and np.all(pos <= box_max)

    def render(self):
        pass
