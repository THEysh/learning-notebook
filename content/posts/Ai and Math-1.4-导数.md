---
author: 杨盛晖
data: 2024-11-04T09:20:00+08:00
title: Ai and Math-1.4-导数
featured: true
draft: false
tags: ['人工智能数学基础']
categories: ['数学']
---


## 什么是导数？
从分析一个问题开始：

<div style="text-align: left;">
    <img src="https://pic.imgdb.cn/item/66f675d0f21886ccc0fcc8ad.png" alt="Image" style="max-width: 100%; height: auto;"/>
</div>

<div style="text-align: center;">
    <img src="https://pic.imgdb.cn/item/66f67775f21886ccc0fedb33.png" alt="Image" style="max-width: 70%; height: auto;"/>
</div>

瞬时经过降程:$$\Delta s=s\left(t_{0}+\Delta t\right)-s\left(t_{0}\right)$$
瞬时的平均速度:$$v=\frac{\Delta s}{\Delta t}=\frac{s\left(t_{0}+\Delta t\right)-s\left(t_{0}\right)}{\Delta t}$$
当$\Delta t\to0$时,对应的v就是瞬时速度。在时刻$t_0$的解时速度为$$v=\lim_{\Delta t\to0}\overline{v}=\lim_{\Delta t\to0}\frac{s\left(t_{0}+\Delta t\right)-s\left(t_{0}\right)}{\Delta t}$$从公式可以看出,解时速度就是变化率的问题。

- **注：**
   5秒末的时刻通常指的是在某个时间点的最后一瞬间。例如，如果我们说“5秒末”，它可以表示为5秒的结束时刻，即5秒整的那一刻。在实际的时间表示中，5秒末可以被理解为：5.000....秒

![](https://pic.imgdb.cn/item/66f6b4f6f21886ccc0506656.png)


<div style="text-align: center;">
    <img src="https://pic.imgdb.cn/item/66f6b609f21886ccc0520bdc.png" alt="Image" style="max-width: 100%; height: auto;"/>
</div>


## 导数定义

$$f'(x_0)=\lim_{\Delta x\to0}\frac{\Delta y}{\Delta x}=\lim_{\Delta x\to0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}\:.$$

当然，$\left.\frac{\mathrm{d}y}{\mathrm{d}x}\right|_{x=x_{0}},\left.\frac{\mathrm{d}[f(x)]}{\mathrm{d}x}\right|_{x=x_{0}},y^{\prime}(x_{0})$\text{或}$y^\prime|_{x=x_{0}}$这些符号记法与$f^{\prime}(x_{0})$等价。“导数”这个名词被认为是拉格朗日最先使用的，记号$f^{\prime}(x_0),y^{\prime}|_{x=x_0}$多次出现在拉格朗日的文章中。

![](https://pic.imgdb.cn/item/66f6b76af21886ccc053f478.png)

![](https://pic.imgdb.cn/item/66f6b7b7f21886ccc054558e.png)

![](https://pic.imgdb.cn/item/66f6b8a6f21886ccc0558145.png)

![](https://pic.imgdb.cn/item/66f6bbdcf21886ccc059d9f5.png)

![](https://pic.imgdb.cn/item/66f6bc09f21886ccc05a1006.png)

![](https://pic.imgdb.cn/item/66f6bc4bf21886ccc05a660e.png)

## 导数的常见公式：

$$1.(e^x)^{^{\prime}}=e^x$$

$$2.(\ln x)^{'}=\frac1x$$

$$3.(log_ax)^{'}=\frac1{x\cdot\ln a}$$

$$4.(a^x)^{^{\prime}}=a^x\cdot\ln a$$

$$5.(x^n)^{^{\prime}}=n\cdot x^{n-1}$$

$$7.(\sin x)^{'}=\cos x$$

$$8.(\cos x)^{'}=-\sin x$$

$$9.(\arcsin x)^{\prime}=\frac{1}{\sqrt{1-x^{2}}}$$

$$10.(\arccos x)^{\prime}=-\frac{1}{\sqrt{1-x^{2}}} $$

$$11.(\tan x)^{\prime}=\sec^{2}x$$

$$12.(\arctan x)^{\prime}=\frac{1}{1+x^{2}}$$

$$13.(\cot x)^{\prime}=-\csc^{2}x$$

![](https://pic.imgdb.cn/item/66f6c133f21886ccc0609d74.png)


## 洛必达法则

![](https://pic.imgdb.cn/item/66f6ad13f21886ccc045911e.png)


## 洛必达法则求极限示例

- 求极限:
   $$
   \lim_{x \to 0} \frac{\sin(x)}{x}
   $$
   **步骤：**
   1. 当 $ x \to 0 $ 时，分子 $ \sin(x) \to 0 $，分母 $ x \to 0 $，形成 $ \frac{0}{0} $ 型。
   2. 应用洛必达法则：
      $$
      \lim_{x \to 0} \frac{\sin(x)}{x} = \lim_{x \to 0} \frac{\cos(x)}{1}
      $$
   3. 计算极限：
      $$
      = \frac{\cos(0)}{1} = 1
      $$

   **结果：**
   $$
   \lim_{x \to 0} \frac{\sin(x)}{x} = 1
   $$

---

- 求极限：
   $$
   \lim_{x \to 1} \frac{x^2 - 1}{x - 1}
   $$

   **步骤：**
   1. 当 $ x \to 1 $ 时，分子 $ x^2 - 1 \to 0 $，分母 $ x - 1 \to 0 $，形成 $ \frac{0}{0} $ 型。
   2. 应用洛必达法则：
      $$
      \lim_{x \to 1} \frac{x^2 - 1}{x - 1} = \lim_{x \to 1} \frac{2x}{1}
      $$
   3. 计算极限：
      $$
      = 2 \cdot 1 = 2
      $$

   **结果：**
   $$
   \lim_{x \to 1} \frac{x^2 - 1}{x - 1} = 2
   $$

---

- 求极限:
   $$
   \lim_{x \to \infty} \frac{e^x}{x^2}
   $$

   **步骤：**
   1. 当 $ x \to \infty $ 时，分子 $ e^x \to \infty $，分母 $ x^2 \to \infty $，形成 $ \frac{\infty}{\infty} $ 型。
   2. 应用洛必达法则：
      $$
      \lim_{x \to \infty} \frac{e^x}{x^2} = \lim_{x \to \infty} \frac{e^x}{2x}
      $$
   3. 再次应用洛必达法则（因为仍为 $ \frac{\infty}{\infty} $ 型）：
      $$
      \lim_{x \to \infty} \frac{e^x}{2x} = \lim_{x \to \infty} \frac{e^x}{2}
      $$
   4. 计算极限：
      $$
      = \infty
      $$

   **结果：**
   $$
   \lim_{x \to \infty} \frac{e^x}{x^2} = \infty
   $$

## $e^x$求导等于自身
- 这是一个神奇的函数，它的导数等于它本身，通过这个函数，可以推出很多函数的导函数。但高中对的研究很少，现在的我们可以轻松证明它。
   $$
   (e^x)'=\lim_{\Delta x\to0}\frac{e^{x+\Delta x}-e^x}{\Delta x}=\lim_{\Delta x\to0}e^x\cdot\frac{e^{\Delta x}-1}{\Delta x} 
   $$
   洛必达法则:
   $$
   = \lim_{\Delta x\to0}e^x\cdot\frac{e^{\Delta x}}{1} = e^x
   $$

## 基本的求导法则
$\left(1\right)\left[u\left(x\right)\pm\nu\left(x\right)\right]^{\prime}=u^{\prime}\left(x\right)\pm\nu^{\prime}\left(x\right)$。

$\left(2\right)\left[u\left(x\right)\cdot\nu\left(x\right)\right]^{\prime}=u^{\prime}\left(x\right)\cdot\nu\left(x\right)+u\left(x\right)\cdot\nu^{\prime}\left(x\right)$。

$\left(3\right)\left[\frac{u\left(x\right)}{\nu\left(x\right)}\right]^{\prime}=\frac{u^{\prime}\left(x\right)\nu\left(x\right)-u\left(x\right)\cdot\nu^{\prime}\left(x\right)}{\nu^{2}\left(x\right)},\nu\left(x\right)\neq0$


## 可微

![](https://pic.imgdb.cn/item/66f6bd29f21886ccc05b7778.png)

![](https://pic.imgdb.cn/item/66f6bd6ef21886ccc05bcb7a.png)


## 复合函数求导

如果$u=g\left(x\right)$在点$x$处可导，$y=f\left(u\right)$在点$u=g\left(x\right)$处可导，那么复合函数$y=f\left[g\left(x\right)\right]$在点$x$处可导，且其导数为$\frac{dy}{dx}=f^{\prime}\left(u\right)\cdot g^{\prime}\left(x\right)\text{或}\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}$。

- **示例：**
   函数: $ f(x) = y = \ln(x^2 + 1) $
   令$u = x^2+1$,所以，有了一个新函数：$g(x)=y=ln(u)$
   $
   f'(x) =y' \cdot u' = g'(u)\cdot u'(x)
   $
   $$
   \rightarrow f'(x) =(ln(u))'= \frac{1}{u} \cdot \frac{du}{dx}=  \frac{2x}{x^2 + 1}
   $$

- 求导计算例题：

   1.求导数： $ f(x) = x^3 + 2x^2 + x $

   $$
   f'(x) = 3x^2 + 4x + 1
   $$

   ---

   2.求导数：$ g(x) = \sin(x) + \cos(x) $

   $$
   g'(x) = \cos(x) - \sin(x)
   $$

   ---

   3.求导数： $ h(x) = e^{2x} $

   $$
   h'(x) = 2e^{2x}
   $$

   ---
   4.求导数：$y=\ln\left(1+\mathrm{e}^{x}\right)-x.$ 

   $$
   y^\prime=\frac1{1+\mathrm{e}^x}\cdot(1+\mathrm{e}^x)^{\prime}-1=\frac{\mathrm{e}^x}{1+\mathrm{e}^x}-1
   $$

   ---

   5.求导数：$y=\sqrt{x^2+x}.$

   $$
   y^\prime=\frac1{2\sqrt{x^2+x}}\cdot(x^2+x)^{\prime}=\frac1{2\sqrt{x^2+x}}\cdot(2x+1)=\frac{2x+1}{2\sqrt{x^2+x}}
   $$

   ---
   6.求导数：$y=x^3-2x^2+\sin x$

   $$
   y^{\prime }= 3x^2- 4x+ \cos x
   $$

   ---
   7.求导数：$y= x\ln x$ 

   $$
   y'=(x\ln x)'=x\:(\ln x)'+(x)'\ln x
   =x\cdot\frac1x + 1 \cdot \ln x = 1 + \ln x
   $$

   ---

   8.求导数：$y=x^3+\frac7{x^4}-\frac2x+12.$

   $$
   y=x^3+7x^{-4}-2x^{-1}+12
   $$
   $$
   y'=3x^2+7\cdot(-4)x^{-5}-2\cdot(-1)x^{-2}+0=3x^2-\frac{28}{x^5}+\frac2{x^2}
   $$

   ---
   9.求导数: $y=\sin x\cos x$

   $$
   y =\sin x\cos x =\frac{1}{2} \cdot2 \sin x \cos x
   =\frac{1}{2} \sin2x
   $$
   
   $$
   y^{\prime}=\frac12\cos2x\cdot(2x)^{\prime}=\frac12\cdot\cos2x\cdot2=\cos2x.
   $$

   ---

   10.求导数:$y=5x^3-2^x+3e^x$

   $$
   y^{\prime}=5\cdot3x^2-2^x\ln2+3\operatorname{e}^x=15x^2-2^x\ln2+3\operatorname{e}^x
   $$

   ---
   11.求导数：$y=x^2\ln x$

   $$
   \begin{aligned}y^{\prime}=(x^2)^{\prime}\mathrm{ln}x+x^2\left(\mathrm{ln}x\right)^{\prime}=2x\ln x+x^2\cdot\frac1x=2x\ln x+x\end{aligned}
   $$

   ---

   12.求导数：$y=3\operatorname{e}^x\cos x$

   $$
   \begin{aligned}y^{\prime}=&(3\operatorname{e}^x)^{\prime}\mathrm{cos}x+3\operatorname{e}^x(\mathrm{cos}x)^{\prime}=3\operatorname{e}^x\cos x+3\operatorname{e}^x\left(-\mathrm{sin}x\right)
   \\=&3\operatorname{e}^x(\cos x-\sin x).\end{aligned}
   $$

## python求导
**使用sympy库求导**：$f(x)=x^2+3x+5$
```python
import sympy as sp
# 定义符号变量
x = sp.symbols('x')
# 定义一个函数
f = x**2 + 3*x + 5
# 进行求导
f_derivative = sp.diff(f, x)
# 输出结果
print("函数:")
sp.pprint(f)
print("\n导数:")
sp.pprint(f_derivative)
```
## 作业：

![](https://pic.imgdb.cn/item/66f6c24df21886ccc0621ff9.png)

**解题：（p213）**
![](https://pic.imgdb.cn/item/66f6c2bef21886ccc062b450.png)

![](https://pic.imgdb.cn/item/66f6c2a5f21886ccc062961c.png)