---
author: 杨盛晖
data: 2024-11-04T09:19:00+08:00
title: Ai and Math-1.3-极限
featured: true
draft: false
tags: ['人工智能数学基础']
categories: ['数学']
---


## 极限
极限强调了“逼近”的思想。我们通常并不直接计算函数在某一点的值，而是研究函数在该点附近的行为。这个过程帮助我们理解复杂函数的特性。

以下是一些常见的一些数列，我们开始理解，什么是极限？
1. 数列：$ a_n = \frac{1}{n} $
   - 极限：$ lim_{n \to \infty} a_n = 0 $

2. 数列：$ b_n = \frac{(-1)^n}{n} $
   - 极限：$ lim_{n \to \infty} b_n = 0 $

3. 数列：$\frac{n}{1+2n}$
   - 极限：$ lim_{n \to \infty} c_n = \frac{1}{2} $

![](https://pic.imgdb.cn/item/66f67943f21886ccc0012784.png)

极限定义中的“$x\neq a$”这句话，这表示，在求$x$趋于$a$ 时$f(x)$的极限时，我们从不考虑$x=a.$事实上，$f(x)$甚至无需在$x=a$时有定义.惟一重要的事是$f(x)$在$a$的附近是如何定义的.

极限也有可以能**不存在**,例如:
$$f(x)=sin\frac{\pi}{x}$$
$$f(x)=\frac{|x|}{x}$$
左右极限不相等，说明极限是不存在的

---

<div style="text-align: left;">
    <img src="https://pic.imgdb.cn/item/66f68cf9f21886ccc01aead9.png" alt="Image" style="max-width: 45%; height: auto;"/>
</div>

![](https://pic.imgdb.cn/item/66f69079f21886ccc01f4b6d.png)

可以使用夹逼定理证明$\lim_{x\to0}x^{2}\sin\frac{1}{x}=0$

![](https://pic.imgdb.cn/item/66f692d3f21886ccc021c387.png)

---


## 1.1.极限的定义

<div style="text-align: left;">
    <img src="https://pic.imgdb.cn/item/66f6aeacf21886ccc0477b7a.png" alt="Image" style="max-width: 60%; height: auto;"/>
</div>

![](https://pic.imgdb.cn/item/66f6b091f21886ccc04a28a8.png)

![](https://pic.imgdb.cn/item/66f6b110f21886ccc04af126.png)

<div style="text-align: center;">
    <img src="https://pic.imgdb.cn/item/66f6b1c7f21886ccc04bf465.png" alt="Image" style="max-width: 80%; height: auto;"/>
</div>

设函数$f(x)$在点 $x_0$ 的某一去心邻域内有定义.若存在常数 $A$, 对 于 任 意 给 定 的 $\varepsilon > 0$ (不论它多么小),总存在正数 $\delta$,使得当 $0<|x-x_0|<\delta$时，对应的函数值$f(x)$都满足不等式$|f(x)-A|<\varepsilon$,则$A$ 叫作函数$f(x)$当$x\to x_0$时的极限，记为

$$lim_{x\to x_0}f(x)=A\text{或}f(x)\to A(x\to x_0)\:.$$

### 1.2 单侧极限 

**示例**
- **左侧极限**：$lim_{x \to 0^-} \frac{1}{x} = -\infty$

- **右侧极限**：$lim_{x \to 0^+} \frac{1}{x} = +\infty$

如果在某一点附近的左侧极限和右侧极限相等，则导数存在。例如: $lim_{x \to 1} (x^2 - 1) $ 

---

### 1.3 无穷小和无穷大

#### 1.3.1 无穷小的性质

1. 无限个无穷小相加，不一定是无穷小.

**示例：**

- 考虑数列：
  $$
  \begin{array}{rcl}
  \sum_{n=1}^\infty\frac{1}{n^2} & = & \frac{1}{n^2} + \frac{2}{n^2} + \frac{3}{n^2} + \cdots + \frac{n}{n^2} 
  \\
  \\
  & = & \lim_{n \to \infty} \frac{\frac{n^2 + n}{2}}{n^2} = \frac{1}{2}
  \end{array}
  $$
  所以无限个无穷小相加，不一定是无穷小.

---
#### 1.3.2 无穷小的比阶
![](https://pic.imgdb.cn/item/66f6ac8cf21886ccc044d182.png)


**下面示例尝试使用python编程完成：**

**示例 1: 高阶无穷小**
设 $\alpha(x) = x^2$ 和 $\beta(x) = x$。
- 计算 $lim_{x \to 0} \frac{\alpha(x)}{\beta(x)}$:
  
  $$
  lim_{x \to 0} \frac{x^2}{x} = lim_{x \to 0} x = 0
  $$

- 因此，$\alpha(x) = o(\beta(x))$。

**示例 2: 低阶无穷小**
设 $\alpha(x) = x$ 和 $\beta(x) = x^2$。
- 计算 $lim_{x \to 0} \frac{\alpha(x)}{\beta(x)}$:
  
  $$
  lim_{x \to 0} \frac{x}{x^2} = lim_{x \to 0} \frac{1}{x} = \infty
  $$

- 因此，$\alpha(x)$ 是比 $\beta(x)$ 低阶的无穷小。



**示例 3: 同阶无穷小**
设 $\alpha(x) = x^2+1$ 和 $\beta(x) = 2x^2$。
- 计算 $lim_{x \to 0} \frac{\alpha(x)}{\beta(x)}$:
  
  $$
  lim_{x \to 0} \frac{x^2+1}{2x^2} = \frac{1}{2} \neq 0
  $$

- 因此，$\alpha(x)$ 和 $\beta(x)$ 是同阶无穷小。



**示例 4: 等价无穷小**
设 $\alpha(x) = x$ 和 $\beta(x) = x + x^2$。
- 计算 $lim_{x \to 0} \frac{\alpha(x)}{\beta(x)}$:
  
  $$
  lim_{x \to 0} \frac{x}{x + x^2} = lim_{x \to 0} \frac{1}{1 + x} = 1
  $$

- 因此，$\alpha(x)$ 和 $\beta(x)$ 是等价无穷小。

### 1.4 尝试使用sympy库计算函数的极限

1. $lim_{x \to 0} \frac{\sin(x)}{x}$  
2. $lim_{x \to 1} (x^2 - 1)$  
3. $lim_{x \to \infty} \frac{1}{x}$  
4. $lim_{x \to 2}(3x + 1)$  
5. $lim_{x \to -1} \frac{x^2-1}{x-1}$  
6. $lim_{x \to 0} (e^x - 1)$  
7. $lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$
8. $lim_{x \to 0} \frac{x - \sin(x)}{x^3}$
9. $lim_{x\to4}\frac{\sqrt{x}-2}{x-4}$
10. $lim_{x\to0}\frac{e^{2}-1}{x}$
11. $lim_{x\to1}{\frac{x^{3}-1}{x-1}}$

**一个简单的示例：**
可以使用 sympy 库来计算极限。再次之前，在vscode安装python，安装python解释器，配置python环境变量。随后安装这个库：
```bash
pip install sympy
```
使用清华镜像下载，速度更快
```bash
pip install sympy -i https://pypi.tuna.tsinghua.edu.cn/simple
```
使用python计算极限
```python
import sympy as sp
# 定义变量
x = sp.symbols('x')
# 定义函数
f = sp.sin(x) / x
# 计算极限
limit_result = sp.limit(f, x, 0)
print(limit_result)
```
---