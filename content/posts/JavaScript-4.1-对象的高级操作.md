---
author: 杨盛晖
data: 2024-11-05T09:47:00+08:00
title: JavaScript-4.1-对象的高级操作
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>

## hasOwnProperty()：判断对象中是否包含某个属性

hasOwnProperty() 是 Object 对象的一个方法，用于判断对象自身（即不包括从原型链继承来的属性）是否具有某个特定的属性。

语法：

```js
obj.hasOwnProperty(prop);
```

解释：

- obj 是要检查的对象。
- prop 是一个字符串，表示要检查的属性名。

返回值：如果对象 obj 自身包含名为 prop 的属性，则返回 true。否则，返回 false。

举例：

```js
const obj = {a: undefined, b: 2, c: 3};

console.log(obj.hasOwnProperty('a')); // true
console.log(obj.hasOwnProperty('b')); // true
console.log(obj.hasOwnProperty('d')); // false

```

## Object.freeze() 冻结对象

Object.freeze() 方法可以冻结一个对象。一个被冻结的对象再也不能被修改；冻结了一个对象则不能向这个对象添加新的属性，不能删除已有属性，不能修改该对象已有属性的可枚举性、可配置性、可写性，以及不能修改已有属性的值。此外，冻结一个对象后该对象的原型也不能被修改。freeze() 返回和传入的参数相同的对象。

代码举例：

```js
const params = {
    name: 'qianguyihao';
    port: '8899';
}

Object.freeze(params); // 冻结对象 params

params.port = '8080';// 修改无效

```

上方代码中，把 params 对象冻结后，如果想再改变 params 里面的属性值，是无效的。
