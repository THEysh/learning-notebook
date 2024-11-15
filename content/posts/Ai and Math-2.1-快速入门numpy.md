---
author: 杨盛晖
data: 2024-11-04T09:26:00+08:00
title: Ai and Math-2.1-快速入门numpy
featured: true
draft: false
tags: ['人工智能数学基础','numpy']
categories: ['数学']
---


## 1. **NumPy 基础**

### 什么是 NumPy？
NumPy（Numerical Python）是一个用于科学计算的开源库，提供了一个高效的多维数组对象 `ndarray`，以及执行快速数值计算的各种函数。NumPy 支持向量化操作，可以大大提高计算效率。它是许多科学计算库（如 SciPy、Pandas、Matplotlib）的基础。

### 安装 NumPy
NumPy 可以通过 Python 包管理工具 `pip` 安装。打开命令行终端并运行以下命令：

#### 使用 `pip` 安装
```bash
pip install numpy
```
**创建 NumPy 数组 (`np.array()`)**
  在 Python 脚本或 Jupyter Notebook 中导入 NumPy 库：
  ```python
  import numpy as np
  a = np.array([1, 2, 3])
  print(a)
  ```
**数组的数据类型**
- 整数类型：
  int8：8位整数（范围：-128 到 127）
  int16：16位整数（范围：-32,768 到 32,767）
  int32：32位整数（范围：-2,147,483,648 到 2,147,483,647）
  int64：64位整数（范围：-9,223,372,036,854,775,808 到 9,223,372,036,854,775,807）
- 浮点数类型
  float16：16位浮点数
  float32：32位浮点数
  float64：64位浮点数（通常是默认类型）
- 布尔类型：
  bool：布尔值，表示 True 或 False
- 复数类型：
  complex64：64位复数
  complex128：128位复数036,854,775,807）
```python
import numpy as np

# 整数类型数组
int_array = np.array([1, 2, 3], dtype=int)

# 浮点类型数组
float_array = np.array([1.1, 2.2, 3.3], dtype=float)

# 布尔类型数组
bool_array = np.array([True, False, True], dtype=bool)

# 复数类型数组
complex_array = np.array([1+2j, 3+4j], dtype=complex)
```

**数组的形状与维度**
- 形状（Shape）表示数组的每个维度的大小。例如，一个二维数组的形状 (3, 4) 表示该数组有 3 行 4 列。
- 维度（Dimension）表示数组的轴的数量。一个一维数组有 1 个轴，二维数组有 2 个轴，以此类推。
```python
import numpy as np

# 一维数组，维度是 1，形状是 (5,)
array_1d = np.array([1, 2, 3, 4, 5])

# 二维数组，维度是 2，形状是 (2, 3)
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 三维数组，维度是 3，形状是 (2, 3, 4)
array_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 
                     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

# 查看维度和形状
print("一维数组的维度:", array_1d.ndim)  # 输出 1
print("一维数组的形状:", array_1d.shape)  # 输出 (5,)

print("二维数组的维度:", array_2d.ndim)  # 输出 2
print("二维数组的形状:", array_2d.shape)  # 输出 (2, 3)

print("三维数组的维度:", array_3d.ndim)  # 输出 3
print("三维数组的形状:", array_3d.shape)  # 输出 (2, 3, 4)
```
**解释：**
一维数组 array_1d 的形状是 (5,)，表示它有 5 个元素。
二维数组 array_2d 的形状是 (2, 3)，表示它有 2 行 3 列。
三维数组 array_3d 的形状是 (2, 3, 4)，表示它有 2 个矩阵，每个矩阵有 3 行 4 列。

**数组的大小**
```python
import numpy as np

# 一维数组
array_1d = np.array([1, 2, 3, 4, 5])

# 查看数组的大小
print("一维数组的大小:", array_1d.size)  # 输出 5

# 二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 查看数组的大小
print("二维数组的大小:", array_2d.size)  # 输出 6

# 三维数组
array_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

# 查看数组的大小
print("三维数组的大小:", array_3d.size)  # 输出 24

# 四维数组
array_4d = np.array([[[[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]]],
                     [[[13], [14], [15], [16]], [[17], [18], [19], [20]], [[21], [22], [23], [24]]]])

# 查看数组的大小
print("四维数组的大小:", array_4d.size)  # 输出 48

```

## 2. **数组操作**
- 数组索引与切片
-  一维数组索引
-  多维数组索引
```python
import numpy as np
# 创建一个一维数组
arr_1d = np.array([10, 20, 30, 40, 50])
# 通过索引访问单个元素
print("第一个元素:", arr_1d[0])  # 输出 10
print("第三个元素:", arr_1d[2])  # 输出 30
# 使用负索引，-1 表示最后一个元素
print("最后一个元素:", arr_1d[-1])  # 输出 50
# 创建一个 2x3 的二维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 访问第一行第一列的元素
print("第一行第一列元素:", arr_2d[0, 0])  # 输出 1

# 访问第二行第三列的元素
print("第二行第三列元素:", arr_2d[1, 2])  # 输出 6

# 访问第二行的所有元素
print("第二行的所有元素:", arr_2d[1, :])  # 输出 [4 5 6]

# 访问第一列的所有元素
print("第一列的所有元素:", arr_2d[:, 0])  # 输出 [1 4]

```

- 切片操作

```python
# 创建一个一维数组
arr_1d = np.array([10, 20, 30, 40, 50, 60, 70])

# 获取从第2个到第5个元素（不包括第5个）
print("从第2个到第5个元素:", arr_1d[1:5])  # 输出 [20 30 40 50]

# 获取从第3个元素到最后的元素
print("从第3个元素到最后:", arr_1d[2:])  # 输出 [30 40 50 60 70]

# 获取前4个元素
print("前4个元素:", arr_1d[:4])  # 输出 [10 20 30 40]

# 每隔一个取一个元素
print("每隔一个元素:", arr_1d[::2])  # 输出 [10 30 50 70]

# 反向切片，获取所有元素的反向
print("反向切片:", arr_1d[::-1])  # 输出 [70 60 50 40 30 20 10]

```
在多维数组中，切片操作是通过分别对每个维度进行切片来完成的。
```python
# 创建一个 3x3 的二维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 获取前两行
print("前两行:", arr_2d[:2, :])  # 输出 [[1 2 3], [4 5 6]]

# 获取每行的前两列
print("每行的前两列:", arr_2d[:, :2])  # 输出 [[1 2], [4 5], [7 8]]

# 获取第二行的后两列
print("第二行的后两列:", arr_2d[1, 1:])  # 输出 [5 6]

# 获取对角线元素
print("对角线元素:", arr_2d[::2, ::2])  # 输出 [[1 3], [7 9]]

```

**总结**
- 一维数组索引: 可以使用整数索引或负数索引来访问数组元素。
- 多维数组索引: 对于多维数组，使用逗号分隔的多个索引来访问特定的元素或行列。
- 切片操作: 可以通过切片语法获取数组的一部分，包括指定开始、结束位置和步长。对于多维数组，每个维度都可以使用切片操作。



**数组的重塑与转置**
- `reshape()`
```python
import numpy as np

# 创建一个一维数组
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print("原始数组:")
print(arr)
print("原始形状:", arr.shape)  # 输出: (12,)

# 使用 reshape 重塑为 3 行 4 列的二维数组
reshaped_arr = arr.reshape(3, 4)
print("\n重塑后的数组:")
print(reshaped_arr)
print("重塑后的形状:", reshaped_arr.shape)  # 输出: (3, 4)

```
- `flatten()`
```python
# 创建一个 3x4 的二维数组
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("原始数组:")
print(arr)

# 使用 flatten() 展平数组
flattened_arr = arr.flatten()
print("\n展平后的数组:")
print(flattened_arr)
print("展平后的形状:", flattened_arr.shape)  # 输出: (12,)

```
- `T` (转置)
```python
# 创建一个 2x3 的二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("原始数组:")
print(arr)

# 使用 T 进行转置
transposed_arr = arr.T
print("\n转置后的数组:")
print(transposed_arr)

```
结合使用：
```python
# 创建一个 2x3 的二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("原始数组:")
print(arr)

# 重塑为 3 行 2 列的数组
reshaped_arr = arr.reshape(3, 2)
print("\n重塑后的数组:")
print(reshaped_arr)

# 展平数组
flattened_arr = reshaped_arr.flatten()
print("\n展平后的数组:")
print(flattened_arr)

# 转置数组
transposed_arr = reshaped_arr.T
print("\n转置后的数组:")
print(transposed_arr)

```
- 数组合并与拆分
- `hstack()`, `vstack()`, `concatenate()`
hstack() 用于沿水平方向（列的方向）合并多个数组。:
```python
import numpy as np

# 创建两个二维数组
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print("arr1:")
print(arr1)

print("\narr2:")
print(arr2)

# 水平堆叠 arr1 和 arr2
hstacked_arr = np.hstack((arr1, arr2))
print("\n水平堆叠后的数组:")
print(hstacked_arr)
```
vstack() 用于沿垂直方向（行的方向）合并多个数组。:
```python
# 创建两个二维数组
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print("arr1:")
print(arr1)

print("\narr2:")
print(arr2)

# 垂直堆叠 arr1 和 arr2
vstacked_arr = np.vstack((arr1, arr2))
print("\n垂直堆叠后的数组:")
print(vstacked_arr)

```
concatenate() 用于沿指定的轴（行或列）合并多个数组。它可以在水平方向（axis=1）或垂直方向（axis=0）合并。
```python
# 创建两个二维数组
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print("arr1:")
print(arr1)

print("\narr2:")
print(arr2)

# 使用 concatenate 沿轴 0（垂直方向）合并
concatenated_arr_v = np.concatenate((arr1, arr2), axis=0)
print("\n沿轴 0 合并后的数组:")
print(concatenated_arr_v)

# 使用 concatenate 沿轴 1（水平方向）合并
concatenated_arr_h = np.concatenate((arr1, arr2), axis=1)
print("\n沿轴 1 合并后的数组:")
print(concatenated_arr_h)
```
- `split()`
split() 用于将一个数组沿指定的轴拆分成多个子数组。
```python
# 创建一个一维数组
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print("原始数组:")
print(arr)

# 将数组拆分成 3 个子数组
split_arr = np.split(arr, 3)
print("\n拆分后的数组:")
for sub_arr in split_arr:
    print(sub_arr)

# 创建一个二维数组
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("\n二维数组:")
print(arr2d)

# 将二维数组沿轴 0 拆分
split_arr2d = np.split(arr2d, 3, axis=0)
print("\n沿轴 0 拆分后的二维数组:")
for sub_arr in split_arr2d:
    print(sub_arr)

# 将二维数组沿轴 1 拆分
split_arr2d_axis1 = np.split(arr2d, 3, axis=1)
print("\n沿轴 1 拆分后的二维数组:")
for sub_arr in split_arr2d_axis1:
    print(sub_arr)
```
**总结**
- hstack(): 沿水平方向（列方向）合并数组。
- vstack(): 沿垂直方向（行方向）合并数组。
- concatenate(): 沿指定轴合并多个数组。
- split(): 将数组拆分为多个子数组，支持按指定轴拆分。


## 3. **数组运算**
- 数学运算
- 加法、减法、乘法、除法
```python
import numpy as np

# 创建两个数组
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

# 数组之间的逐元素加法
sum_arr = arr1 + arr2
print("数组加法结果:")
print(sum_arr)

# 数组与标量加法
scalar_sum = arr1 + 10
print("\n数组与标量加法结果:")
print(scalar_sum)

# 数组之间的逐元素减法
diff_arr = arr1 - arr2
print("数组减法结果:")
print(diff_arr)

# 数组与标量减法
scalar_diff = arr1 - 5
print("\n数组与标量减法结果:")
print(scalar_diff)

# 数组之间的逐元素乘法
prod_arr = arr1 * arr2
print("数组乘法结果:")
print(prod_arr)

# 数组与标量乘法
scalar_prod = arr1 * 2
print("\n数组与标量乘法结果:")
print(scalar_prod)

# 数组之间的逐元素除法
div_arr = arr1 / arr2
print("数组除法结果:")
print(div_arr)

# 数组与标量除法
scalar_div = arr1 / 2
print("\n数组与标量除法结果:")
print(scalar_div)

# 数字的平方
square_arr = np.square(arr1)
print("数组平方结果:")
print(square_arr)

# 数字的平方根
sqrt_arr = np.sqrt(arr1)
print("\n数组平方根结果:")
print(sqrt_arr)

# 数字的指数
exp_arr = np.exp(arr1)
print("\n数组指数结果:")
print(exp_arr)

```
**总结**
- 加法（+）：逐元素相加，支持数组与标量运算。
- 减法（-）：逐元素相减，支持数组与标量运算。
- 乘法（*）：逐元素相乘，支持数组与标量运算。
- 除法（/）：逐元素相除，支持数组与标量运算。

**NumPy 广播机制 (Broadcasting)**
广播（Broadcasting）是 NumPy 中用于处理不同形状的数组之间进行算术运算的一种技术。当我们执行数组之间的算术操作时，如果它们的形状不同，NumPy 会尝试对较小的数组进行“广播”使得它们的形状相同，从而可以进行元素级别的运算。这种机制使得我们可以轻松地处理不同形状的数组，而不需要显式地复制数据。

**广播的基本概念**
广播的核心思想是：当两个数组的形状不同，但又能通过某些方式使它们的形状一致时，NumPy 会自动进行相应的调整。较小的数组会通过“广播”机制被扩展，重复其元素直到它们的形状与较大的数组匹配。广播操作通常是为了避免创建额外的数组副本，从而提高运算效率。

**NumPy 的广播机制遵循以下规则：**

如果两个数组的维度不同，维度较小的数组会在最左侧加上“1”直到两个数组的维度相同。两个数组在某一维度上的长度要么相等，要么其中一个数组的长度是 1。
如果两个数组在某个维度上的长度相等或其中一个数组的长度是 1，NumPy 会认为它们在该维度上兼容，可以进行广播。如果两个数组的维度在某一维度上不兼容（即该维度的长度都不是 1，也不相等），则不能进行广播。

假设我们有一个一维数组和一个二维数组，NumPy 会自动扩展一维数组以匹配二维数组的形状。
```python
import numpy as np

# 一维数组 (形状为 (3,))
arr1 = np.array([1, 2, 3])

# 二维数组 (形状为 (2, 3))
arr2 = np.array([[10, 20, 30],
                 [40, 50, 60]])

# 逐元素相加，arr1 会被广播成 (2, 3)
result = arr1 + arr2

print("数组 arr1:")
print(arr1)
print("\n数组 arr2:")
print(arr2)
print("\n广播结果:")
print(result)

```
在这个例子中，arr1 是一个形状为 (3,) 的一维数组，而 arr2 是一个形状为 (2, 3) 的二维数组。由于 arr1 的维度较小，NumPy 会在 arr1 上广播，使得其形状变为 (2, 3)，即 arr1 会被复制两次，使得它能够与 arr2 进行逐元素相加。

```python
# 标量与二维数组相加
scalar = 5
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

result = scalar + arr

print("标量值:", scalar)
print("\n数组 arr:")
print(arr)
print("\n广播结果:")
print(result)

```
在这个例子中，scalar 是一个标量值 5，它会被广播到整个数组 arr 上。这个过程相当于将标量值 5 加到数组 arr 中的每一个元素上。

```python
# 2D 数组和 3D 数组的形状
arr1 = np.array([[1, 2], [3, 4]])  # 形状为 (2, 2)
arr2 = np.array([[[5, 6]], [[7, 8]]])  # 形状为 (2, 1, 2)

# 广播运算
result = arr1 + arr2

print("数组 arr1:")
print(arr1)
print("\n数组 arr2:")
print(arr2)
print("\n广播结果:")
print(result)
```

在这个例子中，arr1 是一个形状为 (2, 2) 的二维数组，而 arr2 是一个形状为 (2, 1, 2) 的三维数组。NumPy 会根据广播规则将 arr2 的形状扩展成 (2, 2, 2)，使得它与 arr1 可以进行逐元素加法。

**总结**
- 广播允许不同形状的数组进行算术运算，避免了显式复制数据，从而提高了运算效率。
- 广播规则要求数组在每个维度上的形状要么相等，要么其中一个维度为 1，这样才能进行广播操作。
- 通过广播，NumPy 可以方便地处理形状不同的数组进行加法、减法、乘法等操作，而不需要手动调整数组的形状或进行数据复制。

- 统计运算
- `mean()`, `sum()`, `std()`, `var()`, `min()`, `max()`
```python
import numpy as np

# 创建一个示例数组
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 1. mean()：计算数组元素的均值（平均值）
mean_value = np.mean(arr)
print(f"数组的均值 (mean): {mean_value}")
# 解释：mean() 计算数组中所有元素的算术平均值，即所有元素的总和除以元素的个数。

# 2. sum()：计算数组元素的总和
sum_value = np.sum(arr)
print(f"数组的总和 (sum): {sum_value}")
# 解释：sum() 计算数组中所有元素的加总值。

# 3. std()：计算数组元素的标准差
std_value = np.std(arr)
print(f"数组的标准差 (std): {std_value}")
# 解释：std() 计算数组中元素的标准差，标准差衡量数据的分散程度。标准差越大，数据分布越广。

# 4. var()：计算数组元素的方差
var_value = np.var(arr)
print(f"数组的方差 (var): {var_value}")
# 解释：var() 计算数组中元素的方差，方差是标准差的平方，它表示数据偏离均值的程度。

# 5. min()：计算数组中的最小值
min_value = np.min(arr)
print(f"数组的最小值 (min): {min_value}")
# 解释：min() 返回数组中的最小元素值。

# 6. max()：计算数组中的最大值
max_value = np.max(arr)
print(f"数组的最大值 (max): {max_value}")
# 解释：max() 返回数组中的最大元素值。

# 可以选择应用上述函数于多维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 对二维数组应用这些函数
mean_2d = np.mean(arr_2d, axis=0)  # 计算每列的均值
sum_2d = np.sum(arr_2d, axis=1)    # 计算每行的总和
std_2d = np.std(arr_2d, axis=1)    # 计算每行的标准差
var_2d = np.var(arr_2d, axis=1)    # 计算每行的方差
min_2d = np.min(arr_2d, axis=0)    # 计算每列的最小值
max_2d = np.max(arr_2d, axis=0)    # 计算每列的最大值

print("\n对于二维数组的运算:")
print(f"每列的均值 (mean, axis=0): {mean_2d}")
print(f"每行的总和 (sum, axis=1): {sum_2d}")
print(f"每行的标准差 (std, axis=1): {std_2d}")
print(f"每行的方差 (var, axis=1): {var_2d}")
print(f"每列的最小值 (min, axis=0): {min_2d}")
print(f"每列的最大值 (max, axis=0): {max_2d}")
```
- 排序与索引
- `argsort()`, `sort()`
```python
import numpy as np

# 创建一个示例数组
arr = np.array([5, 2, 9, 1, 5, 6])

# 1. sort()：对数组进行排序，返回排序后的数组
sorted_arr = np.sort(arr)
print(f"排序后的数组 (sort): {sorted_arr}")
# 解释：sort() 会返回一个新的数组，该数组中的元素是按升序排列的。原数组不会被改变。

# 2. argsort()：返回排序后数组元素的索引
sorted_indices = np.argsort(arr)
print(f"排序后的索引 (argsort): {sorted_indices}")
# 解释：argsort() 返回的是数组元素排序后对应的索引值。也就是说，它返回一个新的数组，该数组中的元素是原数组中元素按升序排列后的索引。

# 示例：通过argsort()来重排原数组
arr_sorted_by_argsort = arr[sorted_indices]
print(f"通过argsort重排后的数组: {arr_sorted_by_argsort}")
# 解释：我们可以利用argsort()返回的索引值，重新排列原数组，从而得到一个排序后的数组。

# 示例：处理二维数组
arr_2d = np.array([[3, 2, 1], [6, 5, 4]])

# 对二维数组应用sort()
sorted_arr_2d = np.sort(arr_2d, axis=1)  # 按行排序
print(f"二维数组按行排序后的结果 (sort, axis=1): \n{sorted_arr_2d}")
# 解释：通过设置axis=1，sort() 会按行进行排序。

# 对二维数组应用argsort()
sorted_indices_2d = np.argsort(arr_2d, axis=1)  # 按行的索引排序
print(f"二维数组按行索引排序后的结果 (argsort, axis=1): \n{sorted_indices_2d}")
# 解释：argsort() 返回的是二维数组中每一行元素排序后的索引。

# 使用argsort()对二维数组的列进行排序
sorted_indices_2d_by_column = np.argsort(arr_2d, axis=0)  # 按列排序
print(f"二维数组按列索引排序后的结果 (argsort, axis=0): \n{sorted_indices_2d_by_column}")
# 解释：通过设置axis=0，argsort() 返回每列排序后的索引。

```

- 随机数生成
- `np.random.rand()`, `np.random.randn()`, `np.random.randint()`,随机种子 (`np.random.seed()`)

```python
import numpy as np

# 设置随机种子，确保每次运行代码生成的随机数相同
np.random.seed(42)  # 42是任意选择的数字，使用固定的随机种子

# 1. 使用 np.random.rand() 生成随机浮点数
# np.random.rand() 生成一个或多个 [0, 1) 区间的随机浮点数

# 生成一个单一随机数
rand_single = np.random.rand()
print(f"单个随机浮点数 (np.random.rand()): {rand_single}")

# 生成一个 2x3 的数组，元素为 [0, 1) 区间的随机浮点数
rand_array = np.random.rand(2, 3)
print(f"\n2x3 随机浮点数数组 (np.random.rand(2, 3)):\n{rand_array}")

# 2. 使用 np.random.randn() 生成随机数
# np.random.randn() 生成标准正态分布（均值为 0，标准差为 1）的随机数

# 生成一个单一随机数
randn_single = np.random.randn()
print(f"\n单个标准正态分布随机数 (np.random.randn()): {randn_single}")

# 生成一个 2x3 的数组，元素为标准正态分布的随机数
randn_array = np.random.randn(2, 3)
print(f"\n2x3 标准正态分布随机数数组 (np.random.randn(2, 3)):\n{randn_array}")

# 3. 使用 np.random.randint() 生成随机整数
# np.random.randint(low, high, size) 生成给定区间内的随机整数
# low: 最小值，high: 最大值（不包含），size: 输出的形状

# 生成一个单一的随机整数，范围从 0 到 10（不包括 10）
randint_single = np.random.randint(0, 10)
print(f"\n单个随机整数 (np.random.randint(0, 10)): {randint_single}")

# 生成一个 2x3 的整数数组，范围从 0 到 100（不包括 100）
randint_array = np.random.randint(0, 100, size=(2, 3))
print(f"\n2x3 随机整数数组 (np.random.randint(0, 100, size=(2, 3))):\n{randint_array}")

# 4. 设置随机种子
# np.random.seed(seed) 用于设置随机数生成的种子，保证每次运行代码生成的随机数相同

# 设置种子为 42
np.random.seed(42)

# 再次生成随机数，这次会得到相同的结果，因为我们使用了固定的种子
rand_single_seeded = np.random.rand()
randn_single_seeded = np.random.randn()
randint_single_seeded = np.random.randint(0, 10)

print(f"\n使用相同种子后的结果：")
print(f"rand_single_seeded: {rand_single_seeded}")
print(f"randn_single_seeded: {randn_single_seeded}")
print(f"randint_single_seeded: {randint_single_seeded}")

# 如果不设置种子，生成的随机数将每次都不同。为了验证这一点，我们重新运行一下并改变种子。
np.random.seed(24)

rand_single_new_seed = np.random.rand()
randn_single_new_seed = np.random.randn()
randint_single_new_seed = np.random.randint(0, 10)

print(f"\n使用不同种子后的结果：")
print(f"rand_single_new_seed: {rand_single_new_seed}")
print(f"randn_single_new_seed: {randn_single_new_seed}")
print(f"randint_single_new_seed: {randint_single_new_seed}")

```






