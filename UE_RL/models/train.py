import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.car_obstacle_env import CarObstacleEnv

# 환경 및 모델
env = CarObstacleEnv()
model = PPO("MlpPolicy", env, verbose=1)

# 시각화 설정
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title("🚗 Car Navigation with Obstacle")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)

ax.set_xticks(np.arange(0, 101, 1))
ax.set_yticks(np.arange(0, 101, 1))
ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.6)

# 장애물
for obs_min, obs_max in zip(env.obstacle_min, env.obstacle_max):
    width = obs_max[0] - obs_min[0]
    height = obs_max[1] - obs_min[1]
    rect = plt.Rectangle(obs_min, width, height, color='red', alpha=0.4)
    ax.add_patch(rect)

# 목표 (사각형)
goal_rect = plt.Rectangle(env.goal_min,
                          env.goal_max[0] - env.goal_min[0],
                          env.goal_max[1] - env.goal_min[1],
                          color='green', alpha=0.3, label='Goal')
ax.add_patch(goal_rect)

# 자동차 위치 및 경로
car_plot, = ax.plot([], [], 'bo', label='Car')
car_path, = ax.plot([], [], 'orange', alpha=0.6)
ax.legend()

# 학습
TIMESTEPS = 500_000
LOG_INTERVAL = 50_000

for step in range(0, TIMESTEPS, LOG_INTERVAL):
    print(f"\n🔁 학습 {step + LOG_INTERVAL}단계 중...")
    model.learn(total_timesteps=LOG_INTERVAL, reset_num_timesteps=False)

    # 시각화용 테스트
    obs, _ = env.reset()
    path_x, path_y = [obs[0]], [obs[1]]

    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)

        if isinstance(obs, np.ndarray) and obs.shape[0] >= 2:
            path_x.append(obs[0])
            path_y.append(obs[1])
            car_plot.set_data([obs[0]], [obs[1]])
            car_path.set_data(path_x, path_y)

            plt.draw()
            plt.pause(0.05)

        if done or truncated:
            break

plt.ioff()
plt.show()

# 저장
model.save("500_000_models/ppo_car_obstacle_live")
print("✅ 학습 완료!")
