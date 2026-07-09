# Reinforcement Learning Algorithm Design Comparison

This README explains the notebook `rl_algorithm_design_comparison.ipynb`.

The goal of this extension is not only to make the CartPole animation look better, but to ask a more reinforcement-learning-focused question:

> Under the same CartPole environment and evaluation protocol, which algorithmic design choices improve learning speed, stability, and final policy quality?

The experiment starts from the original course DQN and compares several modifications:

1. Baseline DQN
2. Reward Shaping DQN
3. Double DQN
4. Dueling Double DQN + Reward Shaping
5. Softmax Double DQN + Reward Shaping

All agents are trained on `CartPole-v1`, where the maximum episode length is 500 steps.

---

## 1. Original Baseline: DQN

The original notebook uses a Deep Q-Network, or DQN, to approximate the action-value function:

```text
Q(s, a; theta)
```

For CartPole:

- State `s` has 4 values:
  - cart position
  - cart velocity
  - pole angle
  - pole angular velocity
- Action `a` has 2 choices:
  - push left
  - push right

The greedy policy is:

```math
\pi(s) = \arg\max_a Q(s, a; \theta)
```

The baseline DQN target is:

```math
y = r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-)
```

where:

- `r` is the reward
- `gamma` is the discount factor
- `s'` is the next state
- `theta` is the policy network parameter
- `theta^-` is the target network parameter

The loss is the Huber loss between the predicted Q-value and the target:

```math
\mathcal{L}
= \text{Huber}\left(Q_{\text{policy}}(s,a;\theta) - y\right)
```

The baseline uses:

- replay memory
- target network
- epsilon-greedy exploration
- Huber TD loss

---

## 2. Algorithm Variants

### A. Baseline DQN

This is the direct filled-in version of the original course algorithm.

It uses the environment reward directly:

```math
r_t = 1
```

as long as the pole has not fallen.

This reward is simple, but it is also coarse. A nearly falling pole and a perfectly upright pole both receive the same reward if the episode has not ended.

---

### B. Reward Shaping DQN

Reward shaping changes the training reward while keeping the same environment and evaluation score.

The original reward is:

```math
r_t = 1
```

The shaped reward used in the notebook is:

```math
r'_t
= 1
- 0.45 \cdot \frac{|\theta|}{\theta_{\max}}
- 0.20 \cdot \frac{|x|}{x_{\max}}
- 0.01 \cdot |\dot{\theta}|
```

where:

- `x` is cart position
- `theta` is pole angle
- `theta_dot` is pole angular velocity
- `x_max = 2.4`
- `theta_max = 12 degrees = 12 * pi / 180`

If the pole falls, the reward is set to:

```math
r'_t = -1
```

This gives the agent denser feedback:

- the pole should stay upright
- the cart should stay near the center
- the pole should not rotate too quickly

Important caveat:

Reward shaping can improve learning speed, but it can also bias the agent away from the original objective if designed badly. Therefore, the notebook still evaluates all agents using the original CartPole score: number of steps balanced.

---

### C. Double DQN

Standard DQN uses the same target network expression to both select and evaluate the next action:

```math
y_{\text{DQN}}
= r + \gamma \max_{a'} Q_{\text{target}}(s', a')
```

This can overestimate Q-values because the `max` operator tends to select actions whose values are accidentally overestimated.

Double DQN separates action selection and action evaluation:

```math
a^*
= \arg\max_{a'} Q_{\text{policy}}(s', a')
```

```math
y_{\text{DoubleDQN}}
= r + \gamma Q_{\text{target}}(s', a^*)
```

So:

- policy network chooses the next action
- target network evaluates that chosen action

This is designed to reduce over-optimistic Q-value estimates.

---

### D. Dueling Double DQN + Reward Shaping

Dueling DQN changes the network architecture.

Instead of directly predicting all Q-values, it decomposes Q into:

```math
V(s)
```

and:

```math
A(s,a)
```

where:

- `V(s)` is the value of the state
- `A(s,a)` is the advantage of action `a` in that state

The final Q-value is:

```math
Q(s,a)
= V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')
```

The subtraction of the mean advantage makes the decomposition identifiable and stable.

This variant combines:

- Double DQN target
- reward shaping
- dueling network architecture

The motivation is that in many CartPole states, the most important question is not only "which action is best?", but also "how safe is this state overall?".

---

### E. Softmax Double DQN + Reward Shaping

The baseline uses epsilon-greedy exploration:

```math
\pi(a|s)
=
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
```

Softmax exploration instead samples actions according to their Q-values:

```math
P(a|s)
=
\frac{\exp(Q(s,a)/T)}
{\sum_{a'} \exp(Q(s,a')/T)}
```

where `T` is the temperature.

- high `T`: more exploration
- low `T`: more greedy behavior

This is less blindly random than epsilon-greedy, because actions with higher Q-values are more likely to be sampled.

However, it can be sensitive to Q-value scale and temperature schedule.

---

## 3. Evaluation Metrics

The notebook does not judge performance from one training curve only.

It compares several metrics:

| Metric | Meaning |
|---|---|
| `train_last20` | Average episode length over the last 20 training episodes |
| `train_best` | Best training episode length |
| `eval_mean` | Mean greedy evaluation score over 10 seeds |
| `eval_min` | Worst greedy evaluation score over 10 seeds |
| `eval_max` | Best greedy evaluation score over 10 seeds |
| `episodes_to_solve_train20` | First episode where 20-episode average reaches the solved range |
| `auc_train_steps` | Sum of training episode lengths; rough learning-efficiency measure |
| `recent_loss` | Recent Huber loss |
| `recent_td_abs` | Recent absolute TD error |
| `q_abs_probe` | Mean absolute Q-value on fixed probe states |

The most important metric is:

```text
eval_mean
```

because it evaluates the learned greedy policy without random exploration.

For CartPole:

```text
500 steps = maximum score
475+ steps = usually considered solved
```

---

## 4. Results From the Executed Notebook

The notebook was executed on AutoDL using CUDA.

Observed result table:

| Agent | Episodes Run | Train Last-20 | Eval Mean | Eval Min | Eval Max | Episodes to Solve |
|---|---:|---:|---:|---:|---:|---:|
| A. Baseline DQN | 260 | 104.95 | 98.8 | 96 | 102 | not solved |
| B. Reward Shaping DQN | 171 | 456.75 | 500.0 | 500 | 500 | 171 |
| C. Double DQN | 260 | 158.10 | 116.1 | 114 | 121 | not solved |
| D. Dueling Double DQN + Shaping | 201 | 489.25 | 500.0 | 500 | 500 | 201 |
| E. Softmax Double DQN + Shaping | 260 | 487.45 | 500.0 | 500 | 500 | 257 |

Best agent selected for animation:

```text
B. Reward Shaping DQN
```

Greedy rollout scores:

```text
[(0, 500), (1, 500), (2, 500), (3, 500), (4, 500),
 (5, 500), (6, 500), (7, 500), (8, 500), (9, 500)]
```

Mean score:

```text
500.0
```

This means the selected agent solved all 10 evaluation seeds.

---

## 5. Interpretation

### Main finding

The biggest improvement came from reward shaping.

The baseline DQN only reached:

```text
eval_mean = 98.8
```

while Reward Shaping DQN reached:

```text
eval_mean = 500.0
```

The improvement was:

```math
500.0 - 98.8 = 401.2
```

steps.

This suggests that, under this training budget, the main bottleneck was not network capacity but the weakness of the original reward signal.

---

### Why Double DQN alone did not solve the task

Double DQN improved the target definition, but its result was:

```text
eval_mean = 116.1
```

This is only slightly better than the baseline.

Interpretation:

Double DQN helps with Q-value overestimation, but in this experiment the bigger issue was that the original reward did not tell the agent early enough that the pole was becoming dangerous.

So Double DQN alone was not enough.

---

### Why the more complex models also solved the task

Both of these solved the task:

```text
D. Dueling Double DQN + Shaping
E. Softmax Double DQN + Shaping
```

However, they did not clearly outperform the simpler Reward Shaping DQN.

This is an important conclusion:

> More complex algorithms are not automatically better. They should be judged by evaluation score, stability, and training cost.

Reward Shaping DQN solved the task in 171 episodes, while:

- Dueling Double DQN + Shaping solved in 201 episodes
- Softmax Double DQN + Shaping solved in 257 episodes

So the simplest successful modification was also the most efficient in this run.

---

## 6. What This Says About Algorithm Design

This project compares three important RL design choices:

### Reward function design

Reward shaping changed:

```math
r_t = 1
```

into:

```math
r'_t
= 1
- 0.45 \cdot \frac{|\theta|}{\theta_{\max}}
- 0.20 \cdot \frac{|x|}{x_{\max}}
- 0.01 \cdot |\dot{\theta}|
```

This had the largest practical impact.

### Q-learning target design

Double DQN changed:

```math
\max_{a'} Q_{\text{target}}(s',a')
```

into:

```math
Q_{\text{target}}\left(s',
\arg\max_{a'} Q_{\text{policy}}(s',a')\right)
```

This is theoretically better for reducing overestimation, but it was not the main bottleneck here.

### Policy / decision design

Softmax exploration changed action selection from:

```math
\epsilon\text{-greedy}
```

to:

```math
P(a|s)
=
\frac{\exp(Q(s,a)/T)}
{\sum_{a'} \exp(Q(s,a')/T)}
```

It solved the task, but more slowly than simple reward shaping.

---

## 7. Final Conclusion

In this CartPole experiment, the best practical improvement was not the most complicated algorithm.

The most effective change was:

```text
Reward Shaping
```

because it gave the agent a clearer learning signal:

- keep the pole upright
- keep the cart near the center
- reduce dangerous angular velocity

Double DQN and Dueling DQN are important algorithmic ideas, and they are well supported by RL literature. However, in this experiment, they only became clearly successful when combined with reward shaping.

Therefore, the main lesson is:

> In reinforcement learning, improving the reward signal can matter as much as, or more than, increasing algorithmic complexity.

---

## 8. References

- Mnih et al. (2015), *Human-level control through deep reinforcement learning*. Introduced the DQN recipe of replay memory and target networks.
- Van Hasselt et al. (2016), *Deep Reinforcement Learning with Double Q-learning*. Introduced Double DQN to reduce overestimation bias.
- Wang et al. (2016), *Dueling Network Architectures for Deep Reinforcement Learning*. Introduced the value/advantage decomposition.
- Schaul et al. (2015), *Prioritized Experience Replay*. Proposed sampling high-TD-error transitions more often.

