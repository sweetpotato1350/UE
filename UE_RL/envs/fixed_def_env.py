import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarObstacleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.size = 10
        self.observation_space = spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)
        self.actions = [
            np.array([1, 0]),   # 오른쪽
            np.array([0, 1]),   # 위
            np.array([-1, 0]),  # 왼쪽
            np.array([0, -1]),  # 아래
            np.array([1, 1]),   # 오른쪽 위 대각
            np.array([1, -1]),  # 오른쪽 아래 대각
            np.array([-1, 1]),  # 왼쪽 위 대각
            np.array([-1, -1])  # 왼쪽 아래 대각
        ]

        self.start = np.array([0.0, 0.0])
        self.goal_min = np.array([9.0, 9.0])
        self.goal_max = np.array([10.0, 10.0])
        self.goal_center = (self.goal_min + self.goal_max) / 2

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
        move = self.actions[action]
        next_state = self.state + move

        # 경계 안으로 제한
        next_state = np.clip(next_state, 0, self.size - 1)

        float_state = next_state.astype(float)
        reward = -0.1  # 기본 이동 패널티
        done = False

        # 장애물 충돌 체크
        for obs_min, obs_max in self.obstacles:
            if self._in_box(float_state, obs_min, obs_max):
                reward = -30
                done = True
                break

        if not done:
            # 목표 도달 체크
            if self._in_box(float_state, self.goal_min, self.goal_max):
                reward = 100
                done = True
            else:
                # 거리 기반 보상
                old_dist = np.linalg.norm(self.state - self.goal_center)
                new_dist = np.linalg.norm(float_state - self.goal_center)

                if new_dist < old_dist:
                    reward += 1
                elif new_dist > old_dist:
                    reward -= 1

        self.state = float_state.copy()
        return self.state.copy(), reward, done, False, {}

    def _in_box(self, pos, box_min, box_max):
        return np.all(pos >= box_min) and np.all(pos <= box_max)

    def render(self):
        pass
