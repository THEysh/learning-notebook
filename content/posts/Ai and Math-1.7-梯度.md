---
author: 杨盛晖
data: 2024-11-04T09:23:00+08:00
title: Ai and Math-1.7-梯度
featured: true
draft: false
tags: ['人工智能数学基础']
categories: ['数学']
---

<style>
  img {
    width: 80%;
  }
</style>
# pip安装numpy

```bath
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sympy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 方向导数

![](https://pic.imgdb.cn/item/66f90036f21886ccc0680c5c.png)

![](https://pic.imgdb.cn/item/66f906aaf21886ccc0743905.png)

![](https://pic.imgdb.cn/item/66f9200ef21886ccc0968972.png)

### 关于单位向量

单位向量是一个长度（或模）为 1 的向量，用于表示方向。单位向量通常用于简化计算和表示方向，因为无论实际向量的长度如何，其方向始终保持不变。

**特点**

1. **长度为 1**：单位向量的模（长度）等于 1。
2. **表示方向**：单位向量主要用于表示某个特定方向，而不关心其具体的大小。
3. **标准化**：任何非零向量都可以通过将其除以自身的长度来转换为单位向量，这个过程称为标准化。

**计算单位向量**

- 给定一个向量 $ \vec{v} = (x, y) $，其单位向量 $ \hat{u} $ 的计算公式为：
    $$
    \hat{u} = \frac{\vec{v}}{|\vec{v}|}
    $$
    其中，$ |\vec{v}| $ 是向量的长度，计算方式为：
    $$
    |\vec{v}| = \sqrt{x^2 + y^2}
    $$

-----

**示例**

假设有一个向量 $ \vec{v} = (3, 4) $：

1. **计算其长度**：
   $$
   |\vec{v}| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = 5
   $$

2. **计算单位向量**：
   $$
   \hat{u} = \left( \frac{3}{5}, \frac{4}{5} \right) = (0.6, 0.8)
   $$

这表示单位向量 $ \hat{u} $ 指向与 $ \vec{v} $ 相同的方向，但长度为 1。

----
**计算向量的长度**：
- 已经知道AB坐标，计算$\overrightarrow{AB}$**
    $$
    |\overrightarrow{AB}| = \sqrt{(dx)^2 + (dy)^2}
    $$

- **计算单位向量**：
    $$
    \hat{u} = \frac{\overrightarrow{AB}}{|\overrightarrow{AB}|}
    $$
    $$
    \hat{u} = \left( \frac{dx}{|\overrightarrow{AB}|} \right)i + \left( \frac{dy}{|\overrightarrow{AB}|} \right)j
    $$

    单位向量 $\hat{u}$ 为：
    $$
    \hat{u} = \frac{x_2 - x_1}{\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}} i + \frac{y_2 - y_1}{\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}} j
    $$

![](https://pic.imgdb.cn/item/66f922e4f21886ccc0996a4a.png)

## 梯度向量

![](https://pic.imgdb.cn/item/66f9241cf21886ccc09aaf3c.png)

![](https://pic.imgdb.cn/item/66f92492f21886ccc09b3a1b.png)

## 方向导数最大化

![](https://pic.imgdb.cn/item/66f91d1ef21886ccc0938db3.png)

![](https://pic.imgdb.cn/item/66f924f6f21886ccc09bacdd.png)

**例6中$\overrightarrow{PQ}$向量单位向量$\overrightarrow{u}$的计算方法：**

$$|\overrightarrow{PQ}| = \sqrt{(\frac{1}{2}-2)^2+(2-0)^2}=\frac{5}{2}$$
所以单位向量$\overrightarrow{PQ}$:
$$\overrightarrow{u}=\frac{\frac{1}{2}-2}{\frac{5}{2}}i + \frac{2-0}{\frac{5}{2}}j =\langle -\frac{3}{5},\frac{4}{5} \rangle $$

![](https://pic.imgdb.cn/item/66f97727f21886ccc0f0c7f0.png)

<div style="text-align: right;">
    <img src="https://pic.imgdb.cn/item/66f9774cf21886ccc0f0eb0b.png" alt="Image" style="max-width: 70%; height: auto;"/>
</div>


## 梯度的形象理解

![](https://pic.imgdb.cn/item/66f979d8f21886ccc0f3bce4.png)

## 多元函数极值和最值

![](https://pic.imgdb.cn/item/66f97b4ef21886ccc0f54755.png)

![](https://pic.imgdb.cn/item/66f97c20f21886ccc0f620f9.png)

![](https://pic.imgdb.cn/item/66f97d00f21886ccc0f7019e.png)

![](https://pic.imgdb.cn/item/66f97d23f21886ccc0f726f1.png)

```python
    # 这里是解出v的表达式
    # 定义变量
    x, y, z, v= sp.symbols('x y z v')
    # 定义方程
    eq1 = sp.Eq(x*y*z,v)
    eq2 = sp.Eq(2*x*z+2*y*z+x*y, 12)
    # 解第二个方程，求 z
    solution_z = sp.solve(eq2, z)
    # 将 z 代入第一个方程
    v_expr = sp.solve(eq1.subs(z, solution_z[0]), v)
    # 输出结果
    v_function = v_expr[0]
    print(v_function)
```

## 拉格朗日乘子

![](https://pic.imgdb.cn/item/66f97fdbf21886ccc0f9c2f2.png)

![](https://pic.imgdb.cn/item/66f9802df21886ccc0fa0edd.png)
后续可以使用代码解方程，$x,y,z$均需要是正数，$x,y,z$的计算结果为$(2, 2, 1)$

```python
import sympy as sp
# 定义变量
x, y, z, w= sp.symbols('x y z w')
# 定义方程
eq1 = sp.Eq(w*(2*z+y), y*z)
eq2 = sp.Eq(w*(2*z+x), x*z)
eq3 = sp.Eq(w*(2*x+2*y),x*y)
eq4 = sp.Eq(2*x*z+2*y*z+x*y,12)
# 求解方程组
solution = sp.solve((eq1, eq2, eq3, eq4), (x, y, z, w))
# 输出解
print(solution)
```

![](https://pic.imgdb.cn/item/66f98243f21886ccc0fbefc0.png)
同理用代码求解方程

```python
   # 定义变量
    x, y, w= sp.symbols('x y w')
    # 定义方程
    eq1 = sp.Eq(2*x*w,2*x)
    eq2 = sp.Eq(2*y*w, 4*y)
    eq3 = sp.Eq(x**2+y**2,1)
    # 求解方程组
    solution = sp.solve((eq1, eq2, eq3), (x, y, w))
    # 输出解
    print(solution)
```

### 多个约束条件

![](https://pic.imgdb.cn/item/66f98505f21886ccc0fe49fa.png)
![](https://pic.imgdb.cn/item/66f98604f21886ccc0ff1f4f.png)

```python
        # 定义变量
    x, y, z, w, v= sp.symbols('x y z, w, v')
    # 定义方程
    eq1 = sp.Eq(w+2*x*v,1)
    eq2 = sp.Eq(-w+2*y*v, 2)
    eq3 = sp.Eq(w,3)
    eq4 = sp.Eq(x-y+z,1)
    eq5 = sp.Eq(x**2+y**2,1)
    # 求解方程组
    solution = sp.solve((eq1, eq2, eq3, eq4, eq5), 
                        (x, y, z, w, v))
    # 输出解
    print(solution)
```

## 梯度下降算法

1. **目标函数**：假设我们有一个需要最小化的目标函数 $ f(x) $。这个函数通常是连续的，并且在某个区域内可导。

2. **梯度**：梯度是一个向量，表示函数在某一点的变化率。对于一维函数 $ f(x) $，梯度就是导数 $ f'(x) $。

3. **迭代更新**：

   - 从一个初始点 $ x_0 $ 开始。

   - 计算当前点的梯度 $ \nabla f(x) $。

   - 根据梯度的信息更新当前点：  
     $$
     x_{new} = x_{old} - \alpha \nabla f(x_{old})
     $$
     其中 $ \alpha $ 是学习率，控制每次更新的步长。

4. **收敛**：重复上述过程，直到达到预定的迭代次数或梯度足够小，即认为已经接近最优解。


5. **学习率**（$ \alpha $）的选择非常重要：
   - 如果太大，可能导致跳过最优解，甚至发散。
   - 如果太小，收敛速度慢，计算成本高。


梯度下降法在许多领域都有应用，如：

- **线性回归**：通过最小化损失函数来找到最佳拟合线。
- **神经网络**：用于训练模型，通过反向传播计算梯度并更新权重。

**拓展**

1. **批量梯度下降**：使用整个数据集计算梯度，适用于小数据集。
2. **随机梯度下降（SGD）**：每次只使用一个样本计算梯度，适用于大数据集，能够加速收敛。
3. **小批量梯度下降**：结合了批量和随机的优点，使用数据的一个小批次进行更新。

- **优点**：
  - 实现简单，易于理解。
  - 可以处理大规模数据集。

- **缺点**：
  - 对初始值敏感，可能陷入局部最小值。
  - 学习率的选择对收敛速度影响大。
  - 

## 梯度下降算法寻优示例

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GradientDescentOptimizer1D:
    def __init__(self, starting_point, learning_rate, num_iterations):
        self.starting_point = starting_point
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.history = []

    def f(self, x):
        """目标函数"""
        return (x - 3) ** 2

    def gradient(self, x):
        """计算目标函数的梯度"""
        return 2 * (x - 3)

    def optimize(self):
        """执行梯度下降"""
        x = self.starting_point
        self.history.append(x)  # 记录初始点

        for _ in range(self.num_iterations):
            grad = self.gradient(x)  # 计算当前点的梯度
            x = x - self.learning_rate * grad  # 更新 x 值
            self.history.append(x)

        return x

    def plot_optimization(self):
        """可视化优化过程"""
        x_values = np.linspace(-1, 7, 100)
        y_values = self.f(x_values)

        plt.plot(x_values, y_values, label='f(x) = (x - 3)^2')
        plt.scatter(self.history, self.f(np.array(self.history)), color='red', label='Gradient Descent Steps')
        plt.title('Gradient Descent Optimization')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid()
        plt.show()

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.1, num_iterations=30):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.history = []

    def f(self, x, y):
        """目标函数"""
        return (x - 1)**2 + (y - 2)**2

    def gradient(self, x, y):
        """计算目标函数的梯度"""
        df_dx = 2 * (x - 1)
        df_dy = 2 * (y - 2)
        return np.array([df_dx, df_dy])

    def optimize(self, starting_point):
        """执行梯度下降"""
        point = np.array(starting_point)
        self.history.append(point.copy())

        for _ in range(self.num_iterations):
            grad = self.gradient(point[0], point[1])  # 计算当前点的梯度
            point = point - self.learning_rate * grad  # 更新点
            self.history.append(point.copy())

        return point

    def plot_optimization(self):
        """可视化优化过程"""
        x_values = np.linspace(-1, 3, 100)
        y_values = np.linspace(0, 4, 100)
        X, Y = np.meshgrid(x_values, y_values)
        Z = self.f(X, Y)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

        history = np.array(self.history)
        ax.scatter(history[:, 0], history[:, 1], self.f(history[:, 0], history[:, 1]), color='red', label='Gradient Descent Steps')
        ax.set_title('Gradient Descent Optimization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
        plt.legend()
        plt.show()

    def plot_optimization2(self):
        """可视化优化过程"""
        x_values = np.linspace(-1, 3, 100)
        y_values = np.linspace(0, 4, 100)
        X, Y = np.meshgrid(x_values, y_values)
        Z = self.f(X, Y)

        # 绘制等高线图
        plt.figure(figsize=(10, 6))
        contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour)

        # 绘制梯度下降的路径
        history = np.array(self.history)
        plt.plot(history[:, 0], history[:, 1], marker='o', color='red', label='Gradient Descent Steps')
        plt.scatter(history[-1, 0], history[-1, 1], color='blue', s=100, label='Optimal Point')

        plt.title('Gradient Descent Optimization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid()
        plt.show()

def fun1():
        # 设置参数
    starting_point = 0.0  # 初始点
    learning_rate = 0.1   # 学习率
    num_iterations = 20   # 迭代次数
    optimizer = GradientDescentOptimizer1D(starting_point, learning_rate, num_iterations)
    min_x = optimizer.optimize()
    # 打印结果
    print(f"找到的最小值点: {min_x}")
    print(f"对应的函数值: {optimizer.f(min_x)}")
    # 可视化过程
    optimizer.plot_optimization()

def fun2():
    optimizer = GradientDescentOptimizer(learning_rate=0.1, num_iterations=30)
    starting_point = [0.0, 0.0]  # 初始点
    min_point = optimizer.optimize(starting_point)
    print(f"找到的最小值点: {min_point}")
    print(f"对应的函数值: {optimizer.f(min_point[0], min_point[1])}")
    optimizer.plot_optimization()
    optimizer.plot_optimization2()

if __name__ == "__main__":
   # 使用示例
   fun1()
   fun2()
```