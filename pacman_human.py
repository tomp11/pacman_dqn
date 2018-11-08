import gym
import matplotlib.pyplot as plt
#from animation import plot_animation


env = gym.make('MsPacman-v0')
env.reset()

frames = []

n_max_steps = 1000
for _ in range(n_max_steps):
    env.render()
    a = input()
    if a=="w":
        action=0
    if a=="a":
        action=1
    if a=="d":
        action=2
    if a=="s":
        action=3
    #env.step(env.action_space.sample())
    env.step(action)
    _, _, done, _ = env.step(env.action_space.sample())
    #obs, reward, done, info = env.step(action)
    if done:
        break
env.close()#1回閉めて
env.reset()#リセットする
