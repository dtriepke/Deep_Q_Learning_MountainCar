# Mountain Car with Deep Q-Learning
***


# Introduction
***

This project is about solving a reinforcement problem with an algorithm so called deep Q-Learning. 


## Deep Q-Network
DQN is introduced in 2 papers, Playing Atari with Deep Reinforcement Learning on NIPS in 2013 and Human-level control through deep reinforcement learning on Nature in 2015. Interestingly, there were only few papers about DRN between 2013 and 2015. I guess that the reason was people couldn’t reproduce DQN implementation without information in Nature version.


## Game Environment

The playbox from `openAI` for developing and comparing reinforcement learning algorithms is the library called `gym`.
This library inclued several environments or test problems that can be solved with reinforcement algorithm. 
It provides easy shared interfaces, which enables to skip the complex manual feature engineering. 


This project captures the learning problem `MountainCar`. 
Here is the challenge that a car, stocked between two hills, need to climb the right hill, but the a single impulse cause a to less momentum. The only way to solve the problem is that the agent drives front and back to generate a stronger momentum. 
Typically, the agent does not know about this approach and how to solve this efficiently.
A Moore, Efficient Memory-Based Learning for Robot Control, PhD thesis, University of Cambridge, 1990. first discribed the problem.

![](pic/mountainCar.png)

This is the `MountainCar` evironment from gym.


The spaces for the action is disrcet and there are 3 possible actions availible.
$$
a \in \mathcal{A} = \{0, 1, 2\}
$$


number | action  
-------|-------  
0      | push left
1      | no operation
2      | push right


The observation space $\boldsymbol{i}$ is an `2` dimensional vector. The first dimension tells the position of the car and the second the velocity, fall into two intervalls:

$$
\boldsymbol{i} = (i_1, i_2)' \in \mathcal{S} = [-1.2, 0.6] \times [-0.07, 0.07]
$$

number | sate  
-------|-------  
$i_1$  | position
$i_2$  | velocity


## Reward 

The reward is set to be -1 for each time step except the goal position of $0.5$ is reached.
$$
r \in \mathcal{R} = \{-1, 0\}
$$

and adjusted to:
$$
r_t = 
	\begin{cases}
	i_{1t} & \text{if } i_t \neq (0.5, \cdot) \\
	10     & \text{otherwise}
	\end{cases}.
$$

## Terminal State
The terminal state determnines the end of an epsiode and is either when the car is in state $\boldsymbol{i}_{500}$ or in the state $\boldsymbol{i}_t = (0.5, i_{t2})$ with $t \leq 200$.