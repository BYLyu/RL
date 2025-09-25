# 经典强化学习（Reinforce Learing,RL）与深度强化学习（Deep Reinforce Learing,DRL）算法

## 经典强化学习算法

### 本文件夹中的算法仿真是基于西湖大学赵世钰老师的强化学习课程《【强化学习的数学原理】课程：从零开始到透彻理解（完结）》中的相关描述所进行的复现。

### [赵世钰老师B站课程学习链接](https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.1387.favlist.content.click&vd_source=db006d00775ee58ab55fc2a1a1894557)

### [课程资料Github链接](https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning)

### 包括：1.随机梯度下降（SGD）、小批次梯度下降（MBGD）；2.Sarsa算法；3.Q-learing算法；4.基于值函数估计的Q-learing算法。

### 其中算法2-4是基于网格环境训练一个智能体agent从起点到达指定终点。算法1中的随机梯度下降算法实际上也是机器学习算法的一种，指定了一个智能体的起点，生成随机的100个点，要求智能体从起点使用两种梯度下降算法到达这些点的平均值点。结果图很直观地展示了随机梯度下降算法的随机性以及小批次梯度下降的优点。相关数学原理在此不做赘述，感兴趣的朋友可以观看赵老师的课程，讲的很透彻。


## 深度强化学习算法

### 本文件夹中的算法主要是对书籍《动手学强化学习》中代码的完善。源代码是基于gym，本项目中的代码是基于gymnasium。修改了一些超参数使得相关算法在新的环境中也能取得不错的效果。其中文件rl_utils是源文件中的自定义库，主要功能包括绘图、曲线平滑等功能。

### [动手学强化学习课程资料链接](https://hrl.boyuai.com/)

# 注意事项
### 最好使用Anaconda虚拟环境运行，因为本项目文件有使用jupyter，所以在运行代码时不要忘记
```bash
pip install jupyter notebook
```