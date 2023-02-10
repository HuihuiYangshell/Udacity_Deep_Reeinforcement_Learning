# Project 1: Navigation

## Introduction
This project is for Udacity, which is part of the [Deep Reinforcement Learning Nanodegree Program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).  

The goal in this project is to implement Deep Reinforcement Learning method to create and train an agent to navigate and collect bananas in a large, square world.  
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"



## Environment
One will learn how to use Python API to control an agent.
One will not need to install Unity - this is because we have already built.
One can also build the enviroment using the **Unity Machine Learning Agents Toolkit (ML-Agents)**, which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).  

The project environment provided by Udacity is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).  

In this environment, an Agent navigates a large, square world collecting bananas. Each episode of this task is limited to 300 steps. A reward of **+1** is provided for collecting a yellow banana, and a reward of **-1** is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible, avoiding the blue ones.

#### State and action spaces
The state-space has **37 dimensions** and contains the agent's velocity, along with the ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. **Four discrete actions** are available, corresponding to:

- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

#### Solving the environment
To solve the environment, the Agent must obtain an average score of **>13** over 100 consecutive episodes.


## Included in this repository

* The code used to create and train the Agent
  * Navigation_success.ipynb
  * dqn_agent.py
  * model.py
* A file describing all the packages required to set up the environment
  * environment.yml
* The trained model
  * dqn_success.pth
* A Report.pdf file describing the development process and the learning algorithm, along with ideas for future work
* This README.md file

## Setting up the environment

This section describes how to get the code for this project and configure the environment.

### Create (and activate) a new environment drlnd
```
conda create --name drlnd python=3.6 
activate drlnd
```
### Getting the code
You have two options to get the code contained in this repository:
#####  Clone this repository using Git version control system
Make sure you install Git, if not, follow [this link](https://git-scm.com/downloads) to install it.

Having Git installed in your system, you can clone this repository by running the following command, and navigate to the python/ folder. Then, install several dependencies.
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install 
```
### Create an IPython kernel 
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```


## Train the Agent
The environment you have just set up has the files and tools to allow the training of the agent.  

Start the Jupyter Notebook, Navigate to the root of the project in your system and click on the `Navigation.ipynb` notebook.  
![](./img/jupyter_notebook_workspace.png)  

You must to set your operational system by changing the value of the variable `operational_system` in the second cell code.  
The options available are:

* mac
* windows_x86
* windows\_x86\_64
* linux_x86
* linux\_x86\_64  

You can train the agent clicking on the menu bar `Kernel` and then `Restart & Run All`.  
 


### Adjusting the Hyperparameters
To experiment with how the Agent learns through distinct parameters, you can tune these variables:  

**1.** In the **Navigation.ipynb** file  

* n_episodes: Maximum number of training episodes
* max_t: Maximum number of timesteps per episode
* eps_start: Starting value of epsilon, for epsilon-greedy action selection
* eps_end: Minimum value of epsilon
* eps_decay: Multiplicative factor (per episode) for decreasing epsilon  

**2.** In the **dqn_agent.py** file

* BUFFER_SIZE: Replay buffer size
* BATCH_SIZE: Minibatch size
* GAMMA: Discount factor for expected rewards
* TAU: Multiplicative factor for updating the target network weights
* LR: Learning rate
* UPDATE_EVERY: How often to update the network

**3.** In the **model.py** file
* we use 2 layers each has 64 neurons; the activation function is RELU. 


