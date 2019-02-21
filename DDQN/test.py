import tensorflow as tf
from pacman_dqn import *

import matplotlib.pyplot as plt
from animation import plot_animation



frames = []
n_max_steps = 10000
test_n = 3

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    step = global_step.eval()
    print("Training step {}/{} ({:.1f}%)".format(step, n_steps, step * 100 / n_steps))
    obs = env.reset()
    for i in range(test_n):
        for step in range(n_max_steps):
            state = preprocess_observation(obs)
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = np.argmax(q_values)
            obs, reward, done, info = env.step(action)
            img = env.render(mode="rgb_array")
            frames.append(img)
            if done:
                print("\rfinished:{}".format(i+1),end="")
                break
        env.close()
        env.reset()
video = plot_animation(frames, interval=5)
plt.show()
