# DRL_P1_Navigation

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


## 1 Introduction

The goal of this project is to train an agent to navigate and collect bananas in an Unity environment as showed in the following image:  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## 2 Algorithm Implementation

### 2.1 Random Actions

Initially, I tried selecting a random action at each time step to see how much reward I could get. After running the following code several times, I found that the rewards were always below 3, which is clearly insufficient to solve the environment. Therefore, a better strategy is needed, which I will explain in the next sections.
```python
env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))
```

### 2.2 Deep Q-Network (DQN) (Including model description, target network)

Q-Learning is a reinforcement learning algorithm used to find the optimal action-selection policy for any given finite Markov decision process (MDP). It's a simple-to-implement algorithm and can be applied in a wide range of problems. However, it becomes impractical or even impossible when dealing with large or continous state and actions spaces.

As the state space is continuous in this navigation task, Deep Q-Network (DQN) becomes a better choice which is an extension of Q-Learning that uses deep neural networks to approximate the Q-values.

The neural network is implemented in the `model.py` file which can be found [here](https://github.com/yijun-deng/DRL_P1_Navigation/blob/main/model.py#L5). The following figure shows the structure of this neural network:

<img src="image/neural_network.png" width="50%" align="top-left" alt="" title="neural network" />

### 2.3 Epsilon Greedy Search

### 2.4 Replay Buffer

## 3 Experiment Result

The bolow figure shows the final result of the training. The agent was able to solve the evironment within 405 episodes with an average score of 13.05.

<img src="image/result.png" width="50%" align="top-left" alt="" title="result" />

## 4 Future Improvements

### 4.1 Double Deep Q-Network

### 4.2 Dueling Deep Q-Network

### 4.3 Prioritized Experience Replay

