---
author: 杨盛晖
data: 2024-11-05T09:38:00+08:00
title: JavaScript-3.2-立即执行函数
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>


## 立即执行函数 IIFE

### 概念

函数定义完，就立即被调用，这种函数叫做立即执行函数。英文是 IIFE（Immediately-invoked function expression），立即调用函数表达式。

### 语法格式

语法1：

```js
(function() {
  // 函数体
})();
```

语法2：（立即执行函数也可以传参）

```js
(function() {
  // 函数体
})(a, b);
```

语法解释：



- `function(){}`这种写法，需要再加一对圆括号，变成``

### 举例

现有匿名函数如下：

```javascript
	function(a, b) {
		console.log("a = " + a);
		console.log("b = " + b);
	};
```

立即执行函数如下：

```javascript
	(function(a, b) {
		console.log("a = " + a);
		console.log("b = " + b);
	})(123, 456);
```

立即执行函数往往只会执行一次。为什么呢？因为没有变量保存它，执行完了之后，就找不到它了。

## IIFE的作用

### 为变量赋值

当给变量赋值需要一些较复杂的计算时，使用IIFE显得语法更紧凑。

```js
const sex = 'male';
const nickName = (function () {
  if (sex == 'male') {
    return '帅哥';
  } else {
    return '美女';
  }
})();

console.log(nickName);
```

### 将全局变量变为局部变量

现有如下代码：

```js
var arr = [];
for (var i = 0; i < 5; i++) {
  arr.push(function () {
    console.log(i);
  });
}
arr[2](); // 打印5
```

我们知道，上方代码中，i 是全局变量，所有函数共享内存中的同一个变量i。

现在，我们通过立即执行函数进行改造：

```js
var arr = [];
for (var i = 0; i < 5; i++) {
  (function (i) {
    arr.push(function () {
      console.log(i);
    });
  })(i);
}
arr[2](); // 打印2
```

上方代码中，i作为传递给了IIFE的形参，让 i 得以成为 IIFE 的局部变量；并让 IIFE 并形成了闭包（`arr[2]()`打印出了IIFE内部变量 i 的值，说明形成了闭包）。

