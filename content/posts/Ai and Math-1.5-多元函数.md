---
author: 杨盛晖
data: 2024-11-04T09:21:00+08:00
title: Ai and Math-1.5-多元函数
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
## 多元函数及偏导数

从一个多元函数开始，认识等高线
![](https://pic.imgdb.cn/item/66f82e50f21886ccc0b2e406.png)

![](https://pic.imgdb.cn/item/66f996a6f21886ccc00d3a7a.png)

多元函数的连续性质：

![](https://pic.imgdb.cn/item/66f82edcf21886ccc0b36bcb.png)

多元函数的极限定义：
![](https://pic.imgdb.cn/item/66f96eccf21886ccc0e85cef.png)

![](https://pic.imgdb.cn/item/66f833b8f21886ccc0b7eaed.png)

求极限：
$$\lim_{(x, y) \to (0, 0)} \frac{sin(x^2+y^2)}{x^2 + y^2}$$

证明一个极限不存在：
![](https://pic.imgdb.cn/item/66f83728f21886ccc0bb72d5.png)

![](https://pic.imgdb.cn/item/66f837b7f21886ccc0bbf3be.png)
导数不等，所以极限不存在。

![](https://pic.imgdb.cn/item/66f838f4f21886ccc0bd12ed.png)

![](https://pic.imgdb.cn/item/66f83980f21886ccc0bd8e16.png)

# 偏导数

![](https://pic.imgdb.cn/item/66f83a75f21886ccc0be5be5.png)

![](https://pic.imgdb.cn/item/66f83aa0f21886ccc0be8176.png)

![](https://pic.imgdb.cn/item/66f83c38f21886ccc0bfbd23.png)

![](https://pic.imgdb.cn/item/66f83c4af21886ccc0bfcba4.png)

# 求一元切线方程

要求函数 $ y = 2x^2 $ 在任意一点 $ (a, 2a^2) $ 的切线方程，步骤如下：


1. **求导**：计算函数的导数，以获取切线的斜率。
   $$
   y' = \frac{d}{dx}(2x^2) = 4x
   $$

2. **计算切线的斜率**：在点 $ x = a $ 处，切线的斜率为：
   $$
   m = 4a
   $$

3. **使用点斜式方程**：切线方程可以用点斜式来表示：
   $$
   y - y_1 = m(x - x_1)
   $$
   其中 $ (x_1, y_1) = (a, 2a^2) $，代入得：
   $$
   y - 2a^2 = 4a(x - a)
   $$

4. **整理方程**：
   $$
   y - 2a^2 = 4ax - 4a^2
   $$

   $$
   y = 4ax - 2a^2
   $$

   因此，函数 $ y = 2x^2 $ 在任意点 $ (a, 2a^2) $ 的切线方程为：

   $$
   y = 4ax - 2a^2
   $$

# 进一步，求二元切平面方程

要求函数 $ z = x^2 + y^2 $ 在任意一点 $ (a, b, z_0) $ 的切平面方程，步骤如下：

- **计算函数值**：在点 $ (a, b) $ 处，函数值为：
   $$
   z_0 = a^2 + b^2
   $$

- **求偏导数**：

   - 对 $ x $ 的偏导数：
     $$
     \frac{\partial z}{\partial x} = 2x
     $$

   - 对 $ y $ 的偏导数：
     $$
     \frac{\partial z}{\partial y} = 2y
     $$

- **计算切平面的斜率**：在点 $ (a, b) $ 处，偏导数值为：
   $$
   \frac{\partial z}{\partial x}\bigg|_{(a,b)} = 2a
   $$

   $$
   \frac{\partial z}{\partial y}\bigg|_{(a,b)} = 2b
   $$

- **切平面方程**：

   **切平面方程的线性近似**

- 在点 $ (a, b) $ 附近，函数 $ f(x, y) $ 的增量可以用偏导数表示：
  $$
  dz \approx \frac{\partial f}{\partial x}(a, b)(x - a) + \frac{\partial f}{\partial y}(a, b)(y - b)
  $$

- 这里，$ dz $ 是 $ z $ 的微小变化，$ (x - a) $ 和 $ (y - b) $ 分别是 $ x $ 和 $ y $ 相对于 $ a $ 和 $ b $ 的变化量。

- 将上面的增量公式整理为切平面方程：
  $$
  z - z_0 = dz
  $$

- 替换 $ dz $ 的表达式，得到：
  $$
  z - z_0 = \frac{\partial f}{\partial x}(a, b)(x - a) + \frac{\partial f}{\partial y}(a, b)(y - b)
  $$

  $$
   z - (a^2 + b^2) = 2ax - 2a^2 + 2by - 2b^2
  $$

  $$
   z = 2ax + 2by - 2a^2 - 2b^2 + a^2 + b^2
  $$

  $$
   z = 2ax + 2by - a^2 - b^2
  $$

- 因此，函数 $ z = x^2 + y^2 $ 在任意点 $ (a, b, a^2 + b^2) $ 的切平面方程为：
  $$
  z = 2ax + 2by - a^2 - b^2
  $$

![](https://pic.imgdb.cn/item/66f8491af21886ccc0ca4f37.png)

## 综上研究后，我们进一步理解偏导数

![](https://pic.imgdb.cn/item/66f8478ef21886ccc0c93474.png)

## 多元函数偏导数

![](https://pic.imgdb.cn/item/66f8482ff21886ccc0c9a3d4.png)

![](https://pic.imgdb.cn/item/66f84856f21886ccc0c9be76.png)

## 线性近似

![](https://pic.imgdb.cn/item/66f849d0f21886ccc0cace77.png)

## 多元微分

![](https://pic.imgdb.cn/item/66f84aa4f21886ccc0cb5ce5.png)

以一个示例理解多元微分
![](https://pic.imgdb.cn/item/66f84a34f21886ccc0cb101b.png)

<div style="text-align: center;">
    <img src="https://pic.imgdb.cn/item/66f84b6bf21886ccc0cbdff5.png" alt="Image" style="max-width: 70%; height: auto;"/>
</div>


用一个图来理解多元微分
![](https://pic.imgdb.cn/item/66f84ae5f21886ccc0cb85be.png)

