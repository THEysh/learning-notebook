---
author: 杨盛晖
data: 2024-11-05T09:46:00+08:00
title: JavaScript-4.1-浅拷贝和深拷贝
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>


## 概念

-   浅拷贝：只拷贝最外面一层的数据；更深层次的对象，只拷贝引用。

-   深拷贝：拷贝多层数据；每一层级别的数据都会拷贝。

**总结**：

拷贝引用的时候，是属于**传址**，而非**传值**。关于传值和传址的区别，是很基础的内容，详见《JavaScript 基础/对象简介.md》这篇文章。

深拷贝会把对象里**所有的数据**重新复制到新的内存空间，是最彻底的拷贝。

## 浅拷贝的实现方式

### 用 for in 实现浅拷贝（比较繁琐）

```js
const obj1 = {
    name: 'qianguyihao',
    age: 28,
    info: {
        desc: '很厉害',
    },
};

const obj2 = {};
//  用 for in 将 obj1 的值拷贝给 obj2
for (let key in obj1) {
    obj2[key] = obj1[key];
}

console.log('obj2:' + JSON.stringify(obj2));

obj1.info.desc = '永不止步'; // 当修改 obj1 的第二层数据时，obj2的值也会被改变。所以  for in 是浅拷贝

console.log('obj2:' + JSON.stringify(obj2));
```

上方代码中，用 for in 做拷贝时，只能做到浅拷贝。也就是说，在 obj2 中， name 和 age 这两个属性会单独存放在新的内存地址中，和 obj1 没有关系。但是，`obj2.info` 属性，跟 `obj1.info`属性，**它俩指向的是同一个堆内存地址**。所以，当我修改 `obj1.info` 里的值之后，`obj2.info`的值也会被修改。

打印结果如下：

```
obj2:{"name":"qianguyihao","age":28,"info":{"desc":"很厉害"}}

obj2:{"name":"qianguyihao","age":28,"info":{"desc":"永不止步"}}
```

### 用 Object.assgin() 实现浅拷贝（推荐的方式）

上面的 for in 方法做浅拷贝过于繁琐。ES6 给我们提供了新的语法糖，通过 `Object.assgin()` 可以实现**浅拷贝**。

`Object.assgin()` 在日常开发中，使用得相当频繁，非掌握不可。

**语法**：

```js
// 语法1
obj2 = Object.assgin(obj2, obj1);

// 语法2
Object.assign(目标对象, 源对象1, 源对象2...);
```

**解释**：将`obj1` 拷贝给 `obj2`。执行完毕后，obj2 的值会被更新。

**作用**：将 obj1 的值追加到 obj2 中。如果对象里的属性名相同，会被覆盖。

从语法2中可以看出，Object.assign() 可以将多个“源对象”拷贝到“目标对象”中。

**例 1**：

```js
const obj1 = {
    name: 'qianguyihao',
    age: 28,
    info: {
        desc: 'hello',
    },
};

// 浅拷贝：把 obj1 拷贝给 obj2。如果 obj1 只有一层数据，那么，obj1 和 obj2 则互不影响
const obj2 = Object.assign({}, obj1);
console.log('obj2:' + JSON.stringify(obj2));

obj1.info.desc = '永不止步'; // 由于 Object.assign() 只是浅拷贝，所以当修改 obj1 的第二层数据时，obj2 对应的值也会被改变。
console.log('obj2:' + JSON.stringify(obj2));
```

代码解释：由于 Object.assign() 只是浅拷贝，所以在当前这个案例中， obj2 中的 name 属性和 age 属性是单独存放在新的堆内存地址中的，和 obj1 没有关系；但是，`obj2.info` 属性，跟 `obj1.info`属性，**它俩指向的是同一个堆内存地址**。所以，当我修改 `obj1.info` 里的值之后，`obj2.info`的值也会被修改。

打印结果：

```
obj2:{"name":"qianguyihao","age":28,"info":{"desc":"hello"}}

obj2:{"name":"qianguyihao","age":28,"info":{"desc":"永不止步"}}
```

**例 2**：

```js
const myObj = {
    name: 'qianguyihao',
    age: 28,
};

// 【写法1】浅拷贝：把 myObj 拷贝给 obj1
const obj1 = {};
Object.assign(obj1, myObj);

// 【写法2】浅拷贝：把 myObj 拷贝给 obj2
const obj2 = Object.assign({}, myObj);

// 【写法3】浅拷贝：把 myObj 拷贝给 obj31。注意，这里的 obj31 和 obj32 其实是等价的，他们指向了同一个内存地址
const obj31 = {};
const obj32 = Object.assign(obj31, myObj);

```

上面这三种写法，是等价的。所以，当我们需要将对象 A 复制（拷贝）给对象 B，不要直接使用 `B = A`，而是要使用 Object.assign(B, A)。

**例 3**：

```js
let obj1 = { name: 'qianguyihao', age: 26 };
let obj2 = { city: 'shenzhen', age: 28 };
let obj3 = {};

Object.assign(obj3, obj1, obj2); // 将 obj1、obj2的内容赋值给 obj3
console.log(obj3); // {name: "qianguyihao", age: 28, city: "shenzhen"}
```

上面的代码，可以理解成：将多个对象（obj1和obj2）合并成一个对象 obj3。

**例4**：【重要】

```js
const obj1 = {
    name: 'qianguyihao',
    age: 28,
    desc: 'hello world',
};

const obj2 = {
    name: '许嵩',
    sex: '男',
};

// 浅拷贝：把 obj1 赋值给 obj2。这一行，是关键代码。这行代码的返回值也是 obj2
Object.assign(obj2, obj1);

console.log(JSON.stringify(obj2));
```

打印结果：

```
{
    "name":"qianguyihao",
    "sex":"男",
    "age":28,
    "desc":"hello world"
}
```

注意，**例 4 在实际开发中，会经常遇到，一定要掌握**。它的作用是：将 obj1 的值追加到 obj2 中。如果两个对象里的属性名相同，则 obj2 中的值会被 obj1 中的值覆盖。

**例5：**

```js
const a1 = undefined;
const a2 = null;

Object.assgin(a1, {name: 'qiangu'}); // 报错：TypeError. Cannot convert undefined or null to object
Object.assgin(a1, {name: 'yihao'}); // 报错：TypeError. Cannot convert undefined or null to object
```

Object.assign() 方法的第一个参数是目标对象，如果目标对象是 undefined 或 null，则会报错 TypeError。


所以，为了避免报错，我们要先确目标对象存在。比如使用短路运算符确保 a1 是存在的，就不会报错：

```js
const a1 = undefined || {}; // 短路苏奶奶福，确保 obj 是存在的对象
Object.assgin(a1, {name: 'qiangu'});
```

## 深拷贝的实现方式

深拷贝其实就是将浅拷贝进行递归。

### 用 for in 递归实现深拷贝

代码实现：

```js
let obj1 = {
    name: 'qianguyihao',
    age: 28,
    info: {
        desc: 'hello',
    },
    color: ['red', 'blue', 'green'],
};
let obj2 = {};

deepCopy(obj2, obj1);
console.log(obj2);
obj1.info.desc = 'github';
console.log(obj2);

// 方法：深拷贝
function deepCopy(newObj, oldObj) {
    for (let key in oldObj) {
        // 获取属性值 oldObj[key]
        let item = oldObj[key];
        // 判断这个值是否是数组
        if (item instanceof Array) {
            newObj[key] = [];
            deepCopy(newObj[key], item);
        } else if (item instanceof Object) {
            // 判断这个值是否是对象
            newObj[key] = {};
            deepCopy(newObj[key], item);
        } else {
            // 简单数据类型，直接赋值
            newObj[key] = item;
        }
    }
}
```

