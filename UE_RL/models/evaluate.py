import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.car_obstacle_env import CarObstacleEnv

env = CarObstacleEnv()
model = PPO.load("models/ppo_car_obstacle_live")

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("🚗 Car Evaluation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 11, 1))
# 장애물
obstacle_rect = plt.Rectangle(env.obstacle_min,
                              env.obstacle_max[0] - env.obstacle_min[0],
                              env.obstacle_max[1] - env.obstacle_min[1],
                              color='red', alpha=0.4, label='Obstacle')
ax.add_patch(obstacle_rect)

# 목표
goal_rect = plt.Rectangle(env.goal_min,
                          env.goal_max[0] - env.goal_min[0],
                          env.goal_max[1] - env.goal_min[1],
                          color='green', alpha=0.3, label='Goal')
ax.add_patch(goal_rect)

# 자동차
car_plot, = ax.plot([], [], 'bo', label='Car')
car_path, = ax.plot([], [], 'b-', alpha=0.6)
ax.legend()

obs, _ = env.reset()
path_x, path_y = [obs[0]], [obs[1]]

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)

    path_x.append(obs[0])
    path_y.append(obs[1])
    car_plot.set_data([obs[0]], [obs[1]])
    car_path.set_data(path_x, path_y)

    plt.draw()
    plt.pause(0.05)

    if done or truncated:
        print("🏁 에피소드 종료")
        break

plt.ioff()
plt.show()
print("✅ 평가 시각화 완료!")
