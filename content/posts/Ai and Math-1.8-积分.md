---
author: 杨盛晖
data: 2024-11-04T09:24:00+08:00
title: Ai and Math-1.8-积分
featured: true
draft: false
tags: ['人工智能数学基础']
categories: ['数学']
---

## 积分
<style>
  img {
    width: 80%;
  }
</style>

![](https://pic.imgdb.cn/item/66fa1aacf21886ccc0686b3a.png)

![](https://pic.imgdb.cn/item/66fa1ad5f21886ccc0689454.png)

![](https://pic.imgdb.cn/item/66fa1aedf21886ccc068acc1.png)

![](https://pic.imgdb.cn/item/66fa1b75f21886ccc0692b4f.png)

## 定积分

所以积分可以用极限求和的形式来定义：$ \int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x $

其中：

- $\Delta x = \frac{b-a}{n}$
- $x_i^*$ 是在每个小区间内的某个点（可以选择为右端点、左端点或中点等）。

1. **划分区间**：
   将区间 $[a, b]$ 划分为 $n$ 个小区间，每个小区间的宽度为：
   $$
   \Delta x = \frac{b - a}{n}
   $$

2. **确定样本点**：
   选择每个小区间的样本点 $x_i^*$，通常选择右端点：
   $$
   x_i^* = a + i \Delta x
   $$

3. **构造和式**：
   构造 Riemann 和：
   $$
   S_n = \sum_{i=1}^n f(x_i^*) \Delta x = \sum_{i=1}^n f\left(a + i \frac{b - a}{n}\right) \cdot \frac{b - a}{n}
   $$

4. **求极限**：
   取 $n$ 趋近于无穷大：
   $$
   \int_a^b f(x) \, dx = \lim_{n \to \infty} S_n
   $$

假设我们要计算函数 $f(x) = x^2$ 在区间 $[1, 3]$ 上的定积分：

1. 划分区间：
   $$
   \Delta x = \frac{3 - 1}{n} = \frac{2}{n}
   $$

2. 确定样本点（右端点）：
   $$
   x_i^* = 1 + i \cdot \frac{2}{n}
   $$

3. 构造和式：
   $$
   S_n = \sum_{i=1}^n \left(1 + i \cdot \frac{2}{n}\right)^2 \cdot \frac{2}{n}
   $$

   展开：
   $$
   S_n = \sum_{i=1}^n \left(1 + \frac{4i}{n} + \frac{4i^2}{n^2}\right) \cdot \frac{2}{n}
   $$

   $$
   S_n = \frac{2}{n} \sum_{i=1}^n 1 + \frac{8}{n^2} \sum_{i=1}^n i + \frac{8}{n^3} \sum_{i=1}^n i^2 
   $$

   使用公式：

   - $\sum_{i=1}^n 1 = n$
   - $\sum_{i=1}^n i = \frac{n(n+1)}{2}$
   - $\sum_{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}$

   代入后，简化并取极限：

4. **求极限**：
   $$
   \int_1^3 x^2 \, dx = \lim_{n \to \infty} S_n
   $$

   通过上述步骤，可以得到最终结果：
   $$
   \int_1^3 x^2 \, dx = \left[ \frac{x^3}{3} \right]_1^3 = \frac{27}{3} - \frac{1}{3} = \frac{26}{3}
   $$

### 黎曼和

黎曼和是一种用于近似计算定积分的方法。它通过将积分区间分割成小的子区间，并在每个子区间上取一个样本点，计算这些样本点的函数值与子区间宽度的乘积，最终将这些乘积加起来，从而得到近似的积分值。设 $ f(x) $ 是在区间 $[a, b]$ 上的连续函数。我们将这个区间分成 $ n $ 个小的子区间，每个子区间的宽度为：$$\Delta x = \frac{b-a}{n}$$。

对于每个子区间 $[x_{i-1}, x_i]$，我们选择一个点 $ x_i^* $（可以是左端点、右端点或中点），然后黎曼和 $ S_n $ 定义为:
   $$S_n = \sum_{i=1}^{n} f(x_i^*) \Delta x$$


1. **左侧黎曼和**：选择每个子区间的左端点作为样本点：
   $$
   S_n = \sum_{i=0}^{n-1} f(x_i) \Delta x
   $$

2. **右侧黎曼和**：选择每个子区间的右端点作为样本点：
   $$
   S_n = \sum_{i=1}^{n} f(x_i) \Delta x
   $$

3. **中点黎曼和**：选择每个子区间的中点作为样本点：
   $$
   S_n = \sum_{i=0}^{n-1} f\left(x_i + \frac{\Delta x}{2}\right) \Delta x
   $$

![](https://pic.imgdb.cn/item/66fa2412f21886ccc071d90a.png)

![](https://pic.imgdb.cn/item/66fa271ff21886ccc07554e8.png)

![](https://pic.imgdb.cn/item/66fa273bf21886ccc075677c.png)

### 积分和导数之间的关系(牛顿-莱布尼兹公式，微积分的2个基本理论)

![](https://pic.imgdb.cn/item/66fa3aeaf21886ccc085b2a2.png)

![](https://pic.imgdb.cn/item/66fa3b21f21886ccc085e430.png)

<div >
    <img src="https://pic.imgdb.cn/item/66fa3bb3f21886ccc08667c9.png" style="transform: translateY(31.5%);">
</div>


![](https://pic.imgdb.cn/item/66fa3cfef21886ccc0877567.png)

![](https://pic.imgdb.cn/item/66fa3c4cf21886ccc086e56d.png)

![](https://pic.imgdb.cn/item/66fa3c6ff21886ccc08704f8.png)

![](https://pic.imgdb.cn/item/66fa3cc0f21886ccc08744c1.png)

### 积分表

![](https://pic.imgdb.cn/item/66fa3f5df21886ccc089e715.png)

### 使用python计算积分

```python
import sympy as sp
# 定义符号
x = sp.symbols('x')
# 定义函数
f = x**2
# 计算不定积分
indefinite_integral = sp.integrate(f, x)
print(f'不定积分: {indefinite_integral}')
# 计算定积分
a, b = 1, 3
definite_integral = sp.integrate(f, (x, a, b))
print(f'定积分从 {a} 到 {b}: {definite_integral}')
```
![](https://pic.imgdb.cn/item/66fa425bf21886ccc08c77aa.png")

### 积分应用

![](https://pic.imgdb.cn/item/66fa43a6f21886ccc08d9f94.png)

![](https://pic.imgdb.cn/item/66fa4420f21886ccc08e0c8e.png)

![](https://pic.imgdb.cn/item/66fa448ef21886ccc08e7a44.png)

![](https://pic.imgdb.cn/item/66fa44e9f21886ccc08edfc6.png)

![](https://pic.imgdb.cn/item/66fa44faf21886ccc08ef0f7.png)

![](https://pic.imgdb.cn/item/66fa4609f21886ccc08ff7a8.png)