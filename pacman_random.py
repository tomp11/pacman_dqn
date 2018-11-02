import gym
import matplotlib.pyplot as plt
from animation import plot_animation


env = gym.make('MsPacman-v0')
env.reset()

frames = []

n_max_steps = 1000
for _ in range(3):
    for _ in range(n_max_steps):
        #env.reset()ここだとだめだった
        img = env.render(mode="rgb_array")
        frames.append(img)
        env.step(env.action_space.sample())
        _, _, done, _ = env.step(env.action_space.sample())
        #obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()#1回閉めて
    env.reset()#リセットする
video = plot_animation(frames)
#plot_animation(frames)これだとだめだったなぜかはわからん
plt.show()


"""
もともとはこれ
これは１回＝３回死ぬまで
ここで言う"1000"はmax_sep
import gym
env = gym.make('MsPacman-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close()
"""
