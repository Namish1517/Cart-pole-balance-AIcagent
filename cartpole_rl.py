import gymnasium as gym
from stable_baselines3 import PPO
env=gym.make("CartPole-v1")
model=PPO("MlpPolicy",env,verbose=1)
model.learn(total_timesteps=10_000)
model.save("ppo_cartpole")
env=gym.make("CartPole-v1",render_mode="human")
obs,info=env.reset()

for _ in range(1000):
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)
    if terminated or truncated:
        obs,info=env.reset()
