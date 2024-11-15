---
author: 杨盛晖
data: 2024-11-05T09:37:00+08:00
title: JavaScript-3.1-递归函数
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>

## 递归函数

### 概念

如果一个函数在内部调用这个函数自身，这个函数就是递归函数。

递归在数据结构和算法中经常用到，可以将很多复杂的数据模型拆解为简单问题进行求解。一定要掌握。

### 递归的要素

- 递归模式：把大问题拆解为小问题进行分析。也称为递归体。
- 边界条件：需要确定递归到何时结束。也称为递归出口。

### 代码演示：计算阶乘

提问：求一个正整数的阶乘。

**普通写法：**

```js
// 函数：计算一个正整数的阶乘
function factorial(n) {
  let result = 1;
  for (let i = 1; i <= n; i++) {
    result *= i;
  }
  return result;
}

console.log(factorial(5)); // 120
```

现在，我们学习了递归函数之后，会有更简洁的写法。

**递归写法：**

```js
// 递归函数：计算一个正整数的阶乘
function factorial(n) {
  // 递归出口：如果计算1的阶乘，就不用递归了
  if (n == 1) return 1;

  // 开始递归：如果当前这个 n 不是1，就返回 n * (n-1)!
  return n * factorial(n - 1);
}
console.log(factorial(5)); // 120
```



## 递归函数的案例

### 寻找所有的喇叭花数

题目：喇叭花数是一个**三位数**，其每一位数字的阶乘之和恰好等于它本身，即`abc＝a! + b! + c!`，其中abc表示一个三位数。请找出所有的喇叭花数。

思路：将计算某个数字的阶乘封装成函数。

代码实现：

```js
// 递归函数：计算一个数的阶乘
function factorial(n) {
  // 递归出口：如果计算1的阶乘，就不用递归了
  if (n == 1) return 1;

  // 开始递归：如果当前这个 n 不是1，就返回 n * (n-1)!
  return n * factorial(n - 1);
}

// 穷举法，从100到999遍历，寻找喇叭花数
for (let i = 100; i <= 999; i++) {
  // 将数字i转为字符串
  const i_str = i.toString();
  // abc分别表示百位、十位、个位
  const a = Number(i_str[0]);
  const b = Number(i_str[1]);
  const c = Number(i_str[2]);

  // 根据喇叭花数的条件进行判断
  if (factorial(a) + factorial(b) + factorial(c) == i) {
    console.log(i);
  }
}
```

打印结果：

```
145
```

### 斐波那契数列

斐波那契数列是这样一个数列：1、1、2、3、5、8、13、21、34......最早是由意大利数学家斐波那契开始研究的。它的规律是：下标为0和1的项，值为1；从下标为2的项开始，每一项等于前面两项之和。

提问：请找出斐波那契数列的前10项。

代码实现：

```js
// 递归函数：返回斐波那契数列中下标为n的那一项的值
function fib(n) {
  // 下标为0和1的项，值为1
  if (n == 0 || n == 1) return 1;
  // 从下标为2的项开始，每一项等于前面两项之和
  return fib(n - 1) + fib(n - 2);
}

// 循环语句：打印斐波那契数列的前10项
for (let i = 0; i < 15; i++) {
  console.log(fib(i));
}
```

### 小结

关于递归的案例，今后我们还会学习更多的应用场景。比如**深拷贝**就会用到递归。


