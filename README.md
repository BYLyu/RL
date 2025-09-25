# Classical and Deep Reinforcement Learning Algorithms

## Classical Reinforcement Learning (RL)

### The algorithm simulations in this folder are reproductions based on the descriptions from the reinforcement learning course by Professor Shiyu Zhao of Westlake University: "Mathematical Foundations of Reinforcement Learning".

### [Link to Professor Shiyu Zhao's Course on Bilibili](https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.1387.favlist.content.click&vd_source=db006d00775ee58ab55fc2a1a1894557)

### [Link to Course Materials on GitHub](https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning)

### This section includes:

1.  Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent (MBGD).
2.  The Sarsa algorithm.
3.  The Q-learning algorithm.
4.  Q-learning based on value function approximation.

### Algorithms 2-4 are designed to train an agent to navigate from a starting point to a designated goal within a grid environment. Algorithm 1, which is also a type of machine learning algorithm, tasks an agent with moving from a starting point to the average position of 100 randomly generated points using both SGD and MBGD. The resulting plots clearly demonstrate the stochastic nature of SGD and the advantages of MBGD. For detailed mathematical principles, please refer to Professor Zhao's thorough course lectures.

## Deep Reinforcement Learning (DRL)

### The algorithms in this folder are primarily enhancements of the code from the book "Hands-on Reinforcement Learning". The original source code was based on `gym`, whereas the code in this project has been updated to use `gymnasium`. Some hyperparameters have been adjusted to ensure the algorithms perform well in the new environments. The `rl_utils` file is a custom library from the original source, providing functionalities such as plotting and curve smoothing.

### [Link to "Hands-on Reinforcement Learning" Course Materials](https://hrl.boyuai.com/)

# Prerequisites

### It is highly recommended to run this project in an Anaconda virtual environment. As this project utilizes Jupyter, please remember to install it before running the code.

```bash
pip install jupyter notebook
```