+import numpy as np
+import matplotlib.pyplot as plt
+import tensorflow as tf
+import statsmodels.api as sm
+import imageio
+from base64 import b64encode
+from IPython.display import HTML
+
+# Global random seed for reproducibility
+SEED = 42
+
+# set numpy random seed for consistency
+np.random.seed(SEED)
+
+# Constants used by helper functions
+BATCH_SIZE = 64
+EPSILON_MIN = 0.01
+EPSILON_DECAY = 0.995
+TAU = 1e-3
+
+
+def display_table(initial_state, action, next_state, reward, done):
+    """Return a table summarizing a single environment transition.
+
+    Parameters
+    ----------
+    initial_state : array_like
+        State before taking the action.
+    action : int
+        Action index taken by the agent.
+    next_state : array_like
+        Resulting state after taking the action.
+    reward : float
+        Reward obtained after the transition.
+    done : bool
+        Whether the episode terminated after the transition.
+
+    Returns
+    -------
+    statsmodels.iolib.table.SimpleTable
+        HTML table representing the transition.
+    """
+    action_desc = {
+        0: "Do nothing",
+        1: "Fire right engine",
+        2: "Fire main engine",
+        3: "Fire left engine",
+    }.get(int(action), str(action))
+
+    data = [
+        ["Initial State:", np.array2string(np.asarray(initial_state), precision=3)],
+        ["Action:", action_desc],
+        ["Next State:", np.array2string(np.asarray(next_state), precision=3)],
+        ["Reward Received:", f"{reward:.3f}"],
+        ["Episode Terminated:", str(bool(done))],
+    ]
+    table = sm.iolib.table.SimpleTable(data)
+    return table
+
+
+def get_action(q_values, epsilon):
+    """Select an action according to an epsilon-greedy policy."""
+    if np.random.random() < epsilon:
+        return np.random.randint(q_values.shape[-1])
+    return int(np.argmax(q_values))
+
+
+def check_update_conditions(t, num_steps_for_update, memory_buffer):
+    """Check if conditions are met to perform a learning update."""
+    enough_samples = len(memory_buffer) >= BATCH_SIZE
+    time_to_update = (t + 1) % num_steps_for_update == 0
+    return enough_samples and time_to_update
+
+
+def get_experiences(memory_buffer):
+    """Sample a batch of experiences from the memory buffer."""
+    indices = np.random.choice(len(memory_buffer), size=BATCH_SIZE, replace=False)
+    states = []
+    actions = []
+    rewards = []
+    next_states = []
+    dones = []
+    for i in indices:
+        exp = memory_buffer[i]
+        states.append(exp.state)
+        actions.append(exp.action)
+        rewards.append(exp.reward)
+        next_states.append(exp.next_state)
+        dones.append(exp.done)
+    states = tf.convert_to_tensor(np.array(states, dtype=np.float32))
+    actions = tf.convert_to_tensor(np.array(actions, dtype=np.int32))
+    rewards = tf.convert_to_tensor(np.array(rewards, dtype=np.float32))
+    next_states = tf.convert_to_tensor(np.array(next_states, dtype=np.float32))
+    dones = tf.convert_to_tensor(np.array(dones, dtype=np.float32))
+    return states, actions, rewards, next_states, dones
+
+
+def get_new_eps(epsilon):
+    """Update epsilon according to an exponential decay schedule."""
+    return max(EPSILON_MIN, epsilon * EPSILON_DECAY)
+
+
+def plot_history(total_point_history, ma_window=100):
+    """Plot episode rewards with moving average."""
+    plt.figure(figsize=(8, 5))
+    plt.plot(total_point_history, label="Total Points")
+    if len(total_point_history) >= ma_window:
+        mov_avg = np.convolve(total_point_history, np.ones(ma_window)/ma_window, mode="valid")
+        plt.plot(range(ma_window-1, len(total_point_history)), mov_avg, label=f"{ma_window}-Episode MA")
+    plt.xlabel("Episode")
+    plt.ylabel("Total Points")
+    plt.legend()
+    plt.grid(True)
+    plt.show()
+
+
+def create_video(filename, env, q_network, fps=30):
+    """Create a mp4 video of one episode using the given Q-network."""
+    frames = []
+    state = env.reset()
+    done = False
+    frames.append(env.render(mode="rgb_array"))
+    while not done:
+        q_values = q_network(np.expand_dims(state, axis=0))
+        action = int(np.argmax(q_values))
+        state, _, done, _ = env.step(action)
+        frames.append(env.render(mode="rgb_array"))
+    imageio.mimsave(filename, frames, fps=fps)
+
+
+def embed_mp4(filename):
+    """Embed a saved mp4 video in a Jupyter notebook."""
+    video = open(filename, 'rb').read()
+    b64 = b64encode(video).decode()
+    return HTML(f"<video width='840' height='480' controls><source src='data:video/mp4;base64,{b64}' type='video/mp4'></video>")
+
+
+def update_target_network(q_network, target_q_network, tau=TAU):
+    """Perform a soft update of the target network weights."""
+    for target, source in zip(target_q_network.weights, q_network.weights):
+        target.assign(tau * source + (1.0 - tau) * target)
