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
        self._generate_random_obstacles()
        self.state = self.start.copy()
        return self.state.copy(), {}

    def _generate_random_obstacles(self):
        self.obstacles = []
        for _ in range(4):
            for _ in range(100):
                x1, y1 = np.random.uniform(0, 8, size=2)
                w, h = np.random.uniform(1.0, 2.0, size=2)
                x2, y2 = x1 + w, y1 + h
                if x2 > 10 or y2 > 10:
                    continue
                obs_min = np.array([x1, y1])
                obs_max = np.array([x2, y2])
                if not (np.all(self.goal_min < obs_max) and np.all(self.goal_max > obs_min)):
                    self.obstacles.append((obs_min, obs_max))
                    break

    def step(self, action):
        move = self.actions[action]
        next_state = self.state + move
        next_state = np.clip(next_state, 0, self.size - 1)
        float_state = next_state.astype(float)
        reward = -0.01  # 기본 이동 패널티
        done = False

        # ✅ 1. 경계 근처 패널티
        if float_state[0] <= 0.3 or float_state[1] <= 0.3:
            reward -= 0.4

        # ✅ 2. 장애물 근처에서 방향 전환 보상
        is_diagonal = action >= 4  # 대각선은 4~7
        is_horizontal = action in [0, 2]  # 좌우

        for obs_min, obs_max in self.obstacles:
            # 장애물에 닿으면 -4, 종료
            if self._in_box(float_state, obs_min, obs_max):
                reward = -4
                done = True
                break

            # 장애물에서 0.5 이내일 때
            center = (obs_min + obs_max) / 2
            distance = np.linalg.norm(float_state - center)
            if distance <= 0.5 and (is_diagonal or is_horizontal):
                reward += 2

        if not done:
            # ✅ 목표 도달
            if self._in_box(float_state, self.goal_min, self.goal_max):
                reward = 8
                done = True
            else:
                # ✅ 거리 기반 보상
                old_dist = np.linalg.norm(self.state - self.goal_center)
                new_dist = np.linalg.norm(float_state - self.goal_center)
                if new_dist < old_dist:
                    reward += 0.2
                elif new_dist > old_dist:
                    reward -= 0.2

        self.state = float_state.copy()
        return self.state.copy(), reward, done, False, {}

    def _in_box(self, pos, box_min, box_max):
        return np.all(pos >= box_min) and np.all(pos <= box_max)

    def render(self):
        pass
