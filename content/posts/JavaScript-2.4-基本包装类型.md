---
author: 杨盛晖
data: 2024-11-05T09:30:00+08:00
title: JavaScript-2.4-基本包装类型
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>




## 基本数据类型不能绑定属性和方法

属性和方法只能添加给对象，不能添加给基本数据类型。我们拿字符串来举例。

**1、基本数据类型：**

基本数据类型`string`是**无法绑定属性和方法**的。

```javascript
var str = 'qianguyihao';

str.aaa = 12;
console.log(typeof str); //打印结果为：string
console.log(str.aaa); //打印结果为：undefined
```

上方代码中，当我们尝试打印`str.aaa`的时候，会发现打印结果为：undefined。也就是说，不能给 `string` 绑定属性和方法。

当然，我们可以打印 str.length、str.indexOf("m")等等。因为这两个方法的底层做了数据类型转换（**临时**将 `string` 字符串转换为 `String` 对象，然后再调用内置方法），也就是我们在下一段将要讲到的**包装类**。

**2、引用数据类型：**

引用数据类型`String`是可以绑定属性和方法的。如下：

```javascript
var strObj = new String('smyhvae');
strObj.aaa = 123;
console.log(strObj);
console.log(typeof strObj); //打印结果：Object
console.log(strObj.aaa);
```

打印结果：

![](http://img.smyhvae.com/20180202_1351.png)

内置对象 Number 也有一些自带的方法，比如：

-   Number.MAX_VALUE;

-   Number.MIN_VALUE;

内置对象 Boolean 也有一些自带的方法，但是用的不多。

### 基本包装类型

### 介绍

我们都知道，JS 中的数据类型包括以下几种。

-   基本数据类型：String 字符串、Number 数值、BigInt 大型数值、Boolean 布尔值、Null 空值、Undefined 未定义、Symbol。

-   引用数据类型：Object 对象。

JS 为我们提供了三个**基本包装类**：

-   String()：将基本数据类型字符串，转换为 String 对象。

-   Number()：将基本数据类型的数字，转换为 Number 对象。

-   Boolean()：将基本数据类型的布尔值，转换为 Boolean 对象。

通过上面这这三个包装类，我们可以**将基本数据类型的数据转换为对象**。

代码举例：

```javascript
let str1 = 'qianguyihao';
let str2 = new String('qianguyihao');

let num = new Number(3);

let bool = new Boolean(true);

console.log(typeof str1); // 打印结果：string
console.log(typeof str2); // 注意，打印结果：object
```

**需要注意的是**：我们在实际应用中一般不会使用基本数据类型的**对象**。如果使用基本数据类型的对象，在做一些比较时可能会带来一些**不可预期**的结果。

比如说：

```javascript
var boo1 = new Boolean(true);
var boo2 = new Boolean(true);

console.log(boo1 === boo2); // 打印结果竟然是：false
```

再比如说：

```javascript
var boo3 = new Boolean(false);

if (boo3) {
    console.log('qianguyihao'); // 这行代码竟然执行了
}
```

### 基本包装类型的作用

当我们对一些基本数据类型的值去调用属性和方法时，JS引擎会**临时使用包装类将基本数据类型转换为引用数据类型**（即“隐式类型转换”），这样的话，基本数据类型就有了属性和方法，然后再调用对象的属性和方法；调用完以后，再将其转换为基本数据类型。

举例：

```js
var str = 'qianguyihao';
console.log(str.length); // 打印结果：11
```

比如，上面的代码，执行顺序是这样的：

```js
// 步骤（1）：把简单数据类型 string 转换为 引用数据类型  String，保存到临时变量中
var temp = new String('qianguyihao');

// 步骤（2）：把临时变量的值 赋值给 str
str = temp;

//  步骤（3）：销毁临时变量
temp = null;

```

## 在底层，字符串以字符数组的形式保存

在底层，字符串是以字符数组的形式保存的。代码举例：

```javascript
var str = 'smyhvae';
console.log(str.length); // 获取字符串的长度
console.log(str[2]); // 获取字符串中的第3个字符（下标为2的字符）
```

上方代码中，`smyhvae`这个字符串在底层是以`["s", "m", "y", "h", "v", "a", "e"]`的形式保存的。因此，我们既可以获取字符串的长度，也可以获取指定索引 index 位置的单个字符。这很像数组中的操作。

再比如，String 对象的很多内置方法，也可以直接给字符串用。此时，也是临时将字符串转换为 String 对象，然后再调用内置方法。


