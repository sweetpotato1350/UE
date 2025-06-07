import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarObstacleEnv(gym.Env):
    def __init__(self):
        super(CarObstacleEnv, self).__init__()
        self.size = 100
        self.observation_space = spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.start = np.array([1.0, 1.0])
        self.goal_min = np.array([95, 95])
        self.goal_max = np.array([98, 98])

        self.obstacles = self._generate_obstacles()
        self.state = self.start.copy()

    def _generate_obstacles(self):
        obstacles = []

        # 대각선 선형 장애물 더 조밀하게
        for i in range(10, 90, 10):  # 기존 20 → 10으로 간격 좁힘
            x_vals = np.linspace(i, i + 10, 15)  # 점 개수 늘림
            y_vals = np.linspace(10, 90, 15)
            for x, y in zip(x_vals, y_vals):
                obstacles.append(((x, y), (x + 2, y + 2)))

        # 박스형 장애물 더 추가
        obstacles.append(((30, 70), (45, 85)))
        obstacles.append(((60, 20), (75, 35)))
        obstacles.append(((50, 50), (60, 60)))  # 추가 예시

        return obstacles

    def reset(self, seed=None, options=None):
        self.state = self.start.copy()
        return self.state, {}

    def step(self, action):
        self.state = self.state + action * 2.0  # 스케일 조절
        self.state = np.clip(self.state, 0, self.size)

        reward = 0.0
        done = False
        dist_to_goal_center = np.linalg.norm(self.state - np.array([92.5, 92.5]))

        # 장애물 근접 패널티
        for obs_min, obs_max in self.obstacles:
            if self._in_box(self.state, np.array(obs_min) - 1, np.array(obs_max) + 1):
                reward -= 5.0

        # 충돌 처리
        for obs_min, obs_max in self.obstacles:
            if self._in_box(self.state, np.array(obs_min), np.array(obs_max)):
                reward -= 100.0
                done = True

        # 목표 도달
        if self._in_box(self.state, self.goal_min, self.goal_max):
            reward += 200.0
            done = True
        else:
            reward -= dist_to_goal_center * 0.05

        return self.state, reward, done, False, {}

    def _in_box(self, pos, box_min, box_max):
        return np.all(pos >= box_min) and np.all(pos <= box_max)

    def render(self):
        pass

    @property
    def obstacle_min(self):
        return [o[0] for o in self.obstacles]

    @property
    def obstacle_max(self):
        return [o[1] for o in self.obstacles]
