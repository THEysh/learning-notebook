---
author: 杨盛晖
data: 2024-11-05T09:29:00+08:00
title: JavaScript-2.3-对象简介
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>

## 对象简介

### 对象的概念

在 JavaScript 中，对象是一组**无序**的相关属性和方法的集合。

**对象的作用是：封装信息**。比如 Student 类里可以封装学生的姓名、年龄、成绩等。

对象具有**特征**（属性）和**行为**（方法）。

### 对象包括哪些数据类型

我们知道，JS 中的数据类型，包括以下几种：

-   **基本数据类型（值类型）**：String 字符串、Number 数值、BigInt 大型数值、Boolean 布尔值、Null 空值、Undefined 未定义、Symbol。

-   **引用数据类型（引用类型）**：Object 对象。

只要不是那七种基本数据类型，就全都是对象。对象属于一种复合的数据类型，在对象中可以保存多个不同数据类型的属性。

### 对象的分类

1、内置对象：

-   由 ES 标准中定义的对象，在任何的 ES 的实现中都可以使用。

-   比如：Object、Math、Date、String、Array、Number、Boolean、Function 等。

2、宿主对象：

-   由 JS 的运行环境提供的对象，目前来讲主要指由浏览器提供的对象。

-   比如 BOM、DOM，比如`console`、`document`。

3、自定义对象：

-   由开发人员自己创建的对象。

通过 new 关键字创建出来的对象实例，都是属于对象类型。

## 自定义对象

### 为什么需要自定义对象

保存一个值时，可以使用**变量**，保存多个值（一组值）时，可以使用**数组**。

比如，如果要保存一个人的信息，通过数组的方式可以这样保存：

```javascript
const arr = ['王二', 35, '男', '180'];
```

上面这种表达方式比较乱。而如果用 JS 中的**自定义对象**来表达，**结构会更清晰**。如下：

```javascript
const person = {
    name: 'qianguyihao',
    age: 30,
    sex: '男',
    favor: ['阅读', '羽毛球'],
    sayHi: function () {
        console.log('qianguyihao');
    },
};
```

由此可见，自定义对象里面的属性均是**键值对（key: value）**，表示属性和值的映射关系：

-   键/属性：属性名。

-   值：属性值，可以是任意类型的值（数字类型、字符串类型、布尔类型，函数类型等）。

### 自定义对象的语法

语法如下：

```js
const obj = {
    key: value,
    key: value,
    key: value,
};
```

key 和 value 之间用冒号分隔，每组 key:vaue 之间用逗号分隔，最后一对 key:value 的末尾可以写逗号，也可以不写逗号。

问：对象的属性名是否需要加引号？

答：如果属性名不符合 JS 标识符的命名规范，则需要用引号包裹。比如：

```js
const person = {
    'my-name': 'qianguyihao',
};
```

补充：其实，JS 的内置对象、宿主对象，底层也是通过自定义对象的形式（也就是键值对的形式）进行封装的。

## 对象的属性值补充

### 什么叫对象的方法【重要】

对象的属性值可以是任意的数据类型，也可以是个**函数**（也称之为方法）。换而言之，**如果对象的属性值是函数，则这个函数可被称之为对象的“方法”**。

```javascript
const obj = new Object();
obj.sayName = function () {
    console.log('qianguyihao');
};

// 没加括号，就是获取方法
console.log(obj.sayName);
console.log('-----------');
// 加了括号，就是调用方法。即：执行函数内容，并执行函数体的内容
console.log(obj.sayName());
```

打印结果：

![](https://img.smyhvae.com/20221014_1130.png)

### 对象中的属性值，也可以是一个对象

举例：

```javascript
//创建对象 obj1
var obj1 = new Object();
obj1.test = undefined;

//创建对象 obj2
var obj2 = new Object();
obj2.name = 'qianguyihao';

//将整个 obj2 对象，设置为 obj1 的属性
obj1.test = obj2;

console.log(obj1.test.name);
```

打印结果为：qianguyihao

## 传值和传址的区别

### 对象保存在哪里

1、基本数据类型的值直接保存在**栈内存**中，变量与变量之间是独立的，值与值之间是独立的，修改一个变量不会影响其他的变量。

2、对象是保存到**堆内存**中的，每创建一个新的对象，就会在堆内存中开辟出一个新的空间。变量保存的是对象的内存地址（对象的引用）。换而言之，对象的值是保存在**堆内存**中的，而对象的引用（即变量）是保存在**栈内存**中的。

**如果两个变量保存的是同一个对象引用，当一个通过一个变量修改属性时，另一个也会受到影响**。这句话很重要，我们来看看下面的例子。

### 传值

代码举例：

```js
let a = 1;

let b = a; // 将 a 赋值给 b

b = 2; // 修改 b 的值
```

上方代码中，当我修改 b 的值之后，a 的值并不会发生改变。这个大家都知道。我们继续往下看。

### 传址（一个经典的例子）

代码举例：

```javascript
var obj1 = new Object();
obj1.name = '孙悟空';

var obj2 = obj1; // 将 obj1 的地址赋值给 obj2。从此， obj1 和 obj2 指向了同一个堆内存空间

//修改obj2的name属性
obj2.name = '猪八戒';
```

上面的代码中，当我修改 obj2 的 name 属性后，会发现，obj1 的 name 属性也会被修改。因为 obj1 和 obj2 指向的是堆内存中的同一个地址。

这个例子要尤其注意，实战开发中，很容易忽略。

对于引用类型的数据，赋值相当于地址拷贝，a、b 指向了同一个堆内存地址。所以改了 b，a 也会变；本质上 a、b 就是一个东西。

如果你打算把引用类型 A 的值赋值给 B，让 A 和 B 相互不受影响的话，可以通过 Object.assign() 来复制对象。效果如下：

```js
var obj1 = { name: '孙悟空' };

// 复制对象：把 obj1 赋值给 obj3。两者之间互不影响
var obj3 = Object.assign({}, obj1);
```

