---
author: 杨盛晖
data: 2024-11-05T09:34:00+08:00
title: JavaScript-2.8-数组简介
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>


> 之前学习的数据类型，只能存储一个值（字符串也为一个值）。如果我们想存储多个值，就可以使用数组。

## 数组简介

数组（Array）是属于**内置对象**，数组和普通对象的功能类似，都可以用来存储一些值。不同的是：

-   普通对象是使用字符串作为属性名，而数组是使用数字作为**索引**来操作元素。索引：从 0 开始的整数就是索引。

数组的存储性能比普通对象要好。在实际开发中我们经常使用数组存储一些数据（尤其是**列表数据**），使用频率非常高。

![](http://img.smyhvae.com/20200612_1707.png)

比如说，上面这个页面的列表数据，它的数据结构就是一个数组。

数组中的元素可以是任意的数据类型，可以是对象，可以是函数，也可以是数组。数组的元素中，如果存放的是数组，我们就称这种数组为二维数组。

接下来，我们讲一讲数组的基本操作。

## 创建数组对象

### 方式一：使用字面量创建数组

举例：

```javascript
let arr1 = []; // 创建一个空的数组

let arr2 = [1, 2, 3]; // 创建带初始值的数组
```

方式一最简单，也用得最多。

### 方式二：使用构造函数创建数组

语法：

```js
let arr = new Array(参数);

let arr = Array(参数);
```

如果**参数为空**，表示创建一个空数组；如果参数是**一个数值**，表示数组的长度；如果**有多个参数**，表示数组中的元素内容。

举个例子：

```javascript
// 方式一
let arr1 = [11, 12, 13];

// 方式二
let arr2 = new Array(); // 参数为空：创建空数组
let arr3 = new Array(4); // 参数为 size
let arr4 = new Array(15, 16, 17); // 参数为多个数值：创建一个带数据的数组

console.log(typeof arr1); // 打印结果：object

console.log('arr1 = ' + JSON.stringify(arr1));
console.log('arr2 = ' + JSON.stringify(arr2));
console.log('arr3 = ' + JSON.stringify(arr3));
console.log('arr4 = ' + JSON.stringify(arr4));
```

打印结果：

```javascript
object;

arr1 = [11, 12, 13];
arr2 = [];
arr3 = [null, null, null, null];
arr4 = [15, 16, 17];
```

从上方打印结果的第一行可以看出，数组的类型是属于**对象**。

### 数组中的元素的类型

数组中可以存放**任意类型**的数据，例如字符串、数字、布尔值、对象等。

比如：

```javascript
const arr = ['qianguyihao', 28, true, { name: 'qianguyihao' }];
```

我们甚至可以在数组里存放数组。比如：

```js
const arr2 = [
    [11, 12, 13],
    [21, 22, 23],
];
```

## 数组的基本操作

### 数组的索引

**索引** (下标) ：用来访问数组元素的序号，代表的是数组中的元素在数组中的位置（下标从 0 开始算起）。

数组可以通过索引来访问、修改对应的数组元素。我们继续看看。

### 向数组中添加元素

语法：

```javascript
数组[索引] = 值;
```

代码举例：

```javascript
const arr = [];

// 向数组中添加元素
arr[0] = 10;
arr[1] = 20;
arr[2] = 30;
arr[3] = 40;
arr[5] = 50;

console.log(JSON.stringify(arr));
```

打印结果：

```
[10,20,30,40,null,50]
```

### 获取数组中的元素

语法：

```javascript
数组[索引];
```

如果读取不存在的索引（比如元素没那么多），系统不会报错，而是返回 undefined。

代码举例：

```javascript
const arr = [21, 22, 23];

console.log(arr[0]); // 打印结果：21
console.log(arr[5]); // 打印结果：undefined
```

### 获取数组的长度

可以使用`length`属性来获取数组的长度(即“元素的个数”)。

数组的长度是元素个数，不要跟索引号混淆。

语法：

```javascript
数组的长度 = 数组名.length；
```

代码举例：

```javascript
const arr = [21, 22, 23];

console.log(arr.length); // 打印结果：3
```

补充：

对于连续的数组，使用 length 可以获取到数组的长度（元素的个数）；对于非连续的数组（即“稀疏数组”，本文稍后会讲），length 的值会大于元素的个数。因此，尽量不要创建非连续的数组。

### 修改数组的长度

可以通过修改length属性修改数组的长度。

-   如果修改的 length 大于原长度，则多出部分会空出来，置为 null。

-   如果修改的 length 小于原长度，则多出的元素会被删除，数组将从后面删除元素。

-   （特例：伪数组 arguments 的长度可以修改，但是不能修改里面的元素，以后单独讲。）

代码举例：

```javascript
const arr1 = [11, 12, 13];
const arr2 = [21, 22, 23];

// 修改数组 arr1 的 length
arr1.length = 1;
console.log(JSON.stringify(arr1));

// 修改数组 arr2 的 length
arr2.length = 5;
console.log(JSON.stringify(arr2));
```

打印结果：

```javascript
[11]
[21, 22, 23, null, null]
```

### 遍历数组

**遍历**: 就是把数组中的每个元素从头到尾都访问一次。

最简单的做法是通过 for 循环，遍历数组中的每一项。举例：

```javascript
const arr = [10, 20, 30, 40, 50];

for (let i = 0; i < arr.length; i++) {
    console.log(arr[i]); // 打印出数组中的每一项
}
```

下一篇文章，会学习数组的各种方法，到时候，会有更多的做法去遍历数组。

## JS语言中，数组的注意点

> 和其他编程语言相比，JS语言中的数组比较灵活，有许多与众不同的地方。

1、如果访问数组中不存在的索引时，不会报错，会返回undefined。

2、当数组的存储空间不够时，数组会自动扩容。其它编程语言中数组的大小是固定的，不会自动扩容。

3、数组可以存储不同类型数据，其它编程语言中数组只能存储相同类型数据。

4、数组分配的存储空间不一定是连续的。其它语言数组分配的存储空间是连续的。

JS中的数组采用"哈希映射"的方式分配存储空间，我们可以通过索引找到对应空间。各大浏览器也对数组分配的存储空间进行了优化：如果存储的都是相同类型的数据，则会尽量分配连续的存储空间；如果存储的不是相同的数据类型，则不会分配连续的存储空间。

## 数组的解构赋值

解构赋值是ES6中新增的一种赋值方式。

ES5中，如果想把数组中的元素赋值给其他变量，是这样做的：

```js
const arr = [1, 2, [3,4]];
let a = arr[0]; // 1
let b = arr[1]; // 2
let c = arr[2]; // [3, 4]
```

上面这种写法比较啰嗦。通过ES6中的结构复制，我们可以像下面这样做。

1、数组解构赋值，代码举例：

```js
let [a, b, c] = [1, 2, [3, 4]];
console.log(a); // 1
console.log(b); // 2
console.log(c); // [3, 4]
```

注意点：

（1）等号左边的个数和格式，必须和右边的一模一样，才能完全解构。

（2）当然，左边的个数和右边的个数，可以不一样。

2、默认值。在赋值之前，我们可以给左边的变量指定**默认值**：

```js
let [a, b = 3, c = 4] = [1, 2];
console.log(a); // 1
console.log(b); // 2。默认值被覆盖。
console.log(c); // 4。继续保持默认值。
```

3、我们可以使用ES6中新增的**扩展运算符**打包剩余的数据。如果使用了扩展运算符, 那么扩展运算符只能写在最后。代码举例：

```js
let [a, ...b] = [1, 2, 3];
console.log(a); // 1
console.log(b); // [2, 3]
```

## 稀疏数组与密集数组

>  这个知识点，简单了解即可。

- 稀疏数组：索引不连续、数组长度大于元素个数的数组，可以简单理解为有 `empty`（有空隙）的数组。

- 密集数组：索引连续、数组长度等于元素个数的数组。


参考链接：

- [JavaScript 之稀疏数组与密集数组](https://juejin.cn/post/6975531514444562462)

- [JS 稀疏数组](https://github.com/JunreyCen/blog/issues/10)

- [JS 中的稀疏数组和密集数组](https://juejin.cn/post/6844904050152964109)

- [译]JavaScript中的稀疏数组与密集数组：https://www.cnblogs.com/ziyunfei/archive/2012/09/16/2687165.html

- [JavaScript || 数组](https://segmentfault.com/a/1190000008533942)

## 案例

### 例 1：翻转数组

代码实现：

```javascript
const arr = [10, 20, 30, 40, 50]; // 原始数组
const newArr = []; // 翻转后的数组
for (let i = 0; i < arr.length; i++) {
    newArr[i] = arr[arr.length - i - 1];
}
console.log(JSON.stringify(newArr));
```

打印结果：

```
    [50,40,30,20,10]
```

### 例 2：冒泡排序

代码实现：

```javascript
const arr = [20, 10, 50, 30, 40];
for (let i = 0; i < arr.length - 1; i++) {
    for (let j = 0; j < arr.length - i - 1; j++) {
        if (arr[j] > arr[j + 1]) {
            let temp = arr[j];
            arr[j] = arr[j + 1];
            arr[j + 1] = temp;
        }
    }
}
console.log(JSON.stringify(arr));
```

打印结果：

```
    [10,20,30,40,50]
```

