import tensorflow as tf
from pacman_dqn import *

import matplotlib.pyplot as plt
from animation import plot_animation


frames = []
n_max_steps = 10000
test_n = 3

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    obs = env.reset()

    for _ in range(test_n):
        for step in range(n_max_steps):
            state = preprocess_observation(obs)

            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = np.argmax(q_values)


            # Online DQN plays
            obs, reward, done, info = env.step(action)

            #env.render(mode="rgb_array")<-できない
            #env.render()<-できない
            #windows環境では止まるらしい
            img = env.render(mode="rgb_array")
            frames.append(img)

            if done:
                break
        env.close()#1回閉めて
        env.reset()#リセットする
video = plot_animation(frames, interval=5)
plt.show()
