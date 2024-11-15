---
author: 杨盛晖
data: 2024-11-04T09:25:00+08:00
title: Ai and Math-1.9-泰勒多项式
featured: true
draft: false
tags: ['人工智能数学基础']
categories: ['数学']
---

## 泰勒多项式

观看一个视频了解：[泰勒公式](https://www.bilibili.com/video/BV1pkxDeqEya/?spm_id_from=333.999.0.0&vd_source=dfdc11f1f47a765503ad9d7cbf293f64)

泰勒多项式是一种用来逼近函数的多项式。给定一个在某点 $ a $ 上具有足够阶数可导的函数 $ f(x) $，其泰勒多项式可以表示为：
  
$$T_n(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \ldots + \frac{f^{(n)}(a)}{n!}(x - a)^n$$

- 上式，$ T_n(x) $ 是 $ f(x) $ 的 n 阶泰勒多项式。

- $$f(x)=\sum_{n=0}^{N}\frac{f^{(n)}(a)}{n!}(x-a)^{n}+o((x-a)^N)$$
其中$o((x-a)^N)$是佩亚诺余项，佩亚诺余项则强调了余项的渐进性质，常用于实际计算中，它给出具体的误差表达。泰勒多项式具备以下几种主要功能：

## 泰勒多项式作用

1. **函数逼近**：
   - 泰勒多项式能够在点 $ a $ 附近有效地逼近复杂函数，尤其是在 $ x $ 接近 $ a $ 时。

2. **简化计算**：
   - 在某些情况下，使用多项式代替复杂函数可以简化计算，特别是在数值分析和计算机科学中。

3. **应用广泛**：
   - 泰勒多项式在物理学、工程学、经济学等多个领域都有应用，例如在求解微分方程和优化问题时。

**示例**
- 考虑函数 $ f(x) = e^x $ 的泰勒多项式在点 $ a = 0 $ 处的展开：
    $$
    T_n(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \ldots + \frac{x^n}{n!}
    $$

当 $ n \to \infty $ 时，泰勒级数收敛到 $ e^x $。可以使用python可视化加深理解

```python
import numpy as np
import matplotlib.pyplot as plt

class TaylorSeriesExp:
    def __init__(self, x_range):
        """ 初始化函数，并设置 x 值范围 """
        self.x_values = np.linspace(x_range[0], x_range[1], 100)
        self.actual_values = np.exp(self.x_values)

    def taylor_series(self, n):
        """ 计算 e^x 的 n 阶泰勒多项式 """
        result = 0
        for i in range(n + 1):
            result += (self.x_values ** i) / np.math.factorial(i)
        return result

    def plot(self, max_order=15):
        """ 绘制 e^x 和不同阶数的泰勒多项式 """
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.actual_values, label='e^x (Actual)', color='blue')

        # 计算并绘制不同阶数的泰勒多项式
        for order in range(1, max_order + 1):
            if order % 2 == 1:  # 只绘制奇数阶数
                taylor_values = self.taylor_series(order)
                plt.plot(self.x_values, taylor_values, linestyle='--', label=f'{order}th Order Taylor')

        plt.title('Taylor Series Approximation of $e^x$')
        plt.xlabel('x')
        plt.ylabel('$f(x)$')
        plt.axhline(0, color='black', linewidth=0.5, ls='--')
        plt.axvline(0, color='black', linewidth=0.5, ls='--')
        plt.legend()
        plt.grid()
        plt.show()

def fun1_taylor_exp():
    # 使用示例
    taylor_exp = TaylorSeriesExp((-1, 8))
    taylor_exp.plot(max_order=11) #泰勒展开到11阶

if __name__ == '__main__':
    fun1_taylor_exp()
```