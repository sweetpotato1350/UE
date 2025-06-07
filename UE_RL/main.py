import gym
from gym import spaces
import numpy as np


class GridDrivingEnv(gym.Env):
    def __init__(self, grid_size=(5, 5)):
        super(GridDrivingEnv, self).__init__()

        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 상하좌우
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self):
        # 맵 초기화 (0=빈칸, 1=벽, 3=목표)
        self.map = np.zeros(self.grid_size, dtype=int)
        self.map[1, 1] = 1  # 장애물
        self.map[1, 2] = 1  # 장애물
        self.map[3, 3] = 3  # 목표지점

        self.agent_pos = np.array([0, 0])  # 시작 위치
        return self.agent_pos.copy()

    def step(self, action):
        # 액션 정의: 0=상, 1=하, 2=좌, 3=우
        move_map = {
            0: [-1, 0],
            1: [1, 0],
            2: [0, -1],
            3: [0, 1]
        }
        next_pos = self.agent_pos + move_map[action]

        reward = -0.01  # 기본 보상
        done = False

        if not self._is_valid(next_pos):
            reward = -1.0  # 벽 부딪힘
        else:
            self.agent_pos = next_pos
            if self.map[tuple(self.agent_pos)] == 3:
                reward = 1.0
                done = True

        return self.agent_pos.copy(), reward, done, {}

    def _is_valid(self, pos):
        x, y = pos
        if (0 <= x < self.grid_size[0]) and (0 <= y < self.grid_size[1]):
            return self.map[x, y] != 1
        return False

    def render(self, mode="human"):
        grid = np.array(self.map, dtype=str)
        grid[grid == '0'] = '.'
        grid[grid == '1'] = '#'
        grid[grid == '3'] = 'G'
        x, y = self.agent_pos
        grid[x, y] = 'A'
        print("\n".join(" ".join(row) for row in grid))
        print()

if __name__ == "__main__":
    env = GridDrivingEnv()
    obs = env.reset()
    env.render()

    for _ in range(10):
        action = env.action_space.sample()  # 랜덤 이동
        obs, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            break
