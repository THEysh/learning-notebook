---
author: 杨盛晖
data: 2024-11-05T09:45:00+08:00
title: JavaScript-3.9-对象的基本操作
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>


## 对象的基本操作

### 创建对象

使用 new 关键字调用的函数，是构造函数 constructor。**构造函数是专门用来创建对象的函数**。

例如：

```javascript
const obj = new Object();
```

记住，使用`typeof`检查一个对象时，会返回`object`。

关于创建对象的更多方式，可以看上一篇文章《对象的创建&构造函数》。

### 向对象中添加属性

在对象中保存的值称为属性。

向对象添加属性的语法：

```javascript
对象.属性名 = 属性值;
```

举例：

```javascript
const obj = new Object();

//向obj中添加一个name属性
obj.name = '孙悟空';

//向obj中添加一个gender属性
obj.gender = '男';

//向obj中添加一个age属性
obj.age = 18;

console.log(JSON.stringify(obj)); // 将 obj 以字符串的形式打印出来
```

打印结果：

```
	{
		"name":"孙悟空",
		"gender":"男",
		"age":18
	}
```

这里我们也可以看出一个规律：如果对象里本身没有某个属性，则用点语法赋值时，这个属性会被创建出来。

### 获取对象中的属性

**方式 1**：

语法：

```javascript
对象.属性名;
```

如果获取对象中没有的属性，不会报错而是返回`undefined`。

举例：

```javascript
const obj = new Object();

//向obj中添加一个name属性
obj.name = '孙悟空';

//向obj中添加一个gender属性
obj.gender = '男';

//向obj中添加一个age属性
obj.age = 18;

// 获取对象中的属性，并打印出来
console.log(obj.gender); // 打印结果：男
console.log(obj.color); // 打印结果：undefined
```

**方式 2**：可以使用`[]`这种形式去操作属性

如果属性名的命名规范没有遵循标识符的命名规范，就不能采用`.`的方式来操作对象的属性，则必须用方括号的形式来访问。比如说，`123`这种属性名，如果我们直接写成`obj.123 = 789`来操作属性，是会报错的。那怎么办呢？办法如下：

语法格式如下：（读取时，也是采用这种方式）

```javascript
// 注意，括号里的属性名，必须要加引号

// 获取属性
对象['属性名']

// 设置属性值
对象['属性名'] = 属性值;
```

上面这种语法格式，举例如下：

```javascript
obj['123'] = 789;
```

当然，如果属性名遵循了标识符的命名规范，也可以使用方括号操作属性。

**重要**：使用`[]`这种形式去操作属性会更灵活，因为我们可以在`[]`中传递一个**变量**。也就是说，如果属性名以变量的形式存储，请记得也必须使用方括号的形式操作属性。这在日常开发中，使用得非常多。比如：

```js
const person = {
		name: '千古壹号',
    age: 30
}

const myKey = 'name';
// 错误的访问方式
console.log(obj.myKey); // undefined
// 正确的访问方式
console.log(obj[myKey]); // 千古壹号
```

### 修改对象的属性值

语法：

```javascript
对象.属性名 = 新值;
```

举例：

```javascript
obj.name = 'qiangu yihao';
```

### 删除对象的属性

语法：

```javascript
delete obj.name;
```

### in 运算符

通过该运算符可以检查一个对象中是否含有指定的属性。如果有则返回 true，没有则返回 false。

语法：

```javascript
'属性名' in 对象;
```

举例：

```javascript
//检查对象 obj 中是否含有name属性
console.log('name' in obj);
```

我们平时使用的对象不一定是自己创建的，可能是从接口获取的，这个时候，in 运算符可以派上用场。

当然，还有一种写法可以达到上述目的：

```js
if (obj.name) {
    // 如果对象 obj 中有name属性，我就继续做某某事情。
}
```

## for of：遍历数组


ES6 中，如果我们要遍历一个数组，可以这样做：

```js
let arr1 = [2, 6, 8, 5];

for (let value of arr1) {
    console.log(value);
}
```

打印结果：


```
2
6
8
5
```


for ... of 的循环可以避免我们开拓内存空间，增加代码运行效率，所以建议大家在以后的工作中使用 for…of 遍历数组。

注意，上面的数组中，`for ... of`获取的是数组里的值；如果采用`for ... in`遍历数组，则获取的是 index 索引值。

### Map 对象的遍历

`for ... of`既可以遍历数组，也可以遍历 Map 对象。


## for in：遍历对象的属性

> `for ... in`主要用于遍历对象，不建议用来遍历数组。

语法：

```javascript
for (const 变量 in 对象) {

}
```

解释：对象中有几个属性，循环体就会执行几次。每次执行时，会将对象中的**每个属性的 属性名 赋值给变量**。

语法举例：

```javascript
for (var key in obj) {
    console.log(key); // 这里的 key 是：对象属性的键（也就是属性名）
    console.log(obj[key]); // 这里的 obj[key] 是：对象属性的值（也就是属性值）
}
```

举例：

```html
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8" />
        <title></title>
        <script type="text/javascript">
            const obj = {
                name: 'qianguyihao',
                age: 28,
                gender: '男',
                address: 'shenzhen',
                sayHi: function () {
                    console.log(this.name);
                },
            };

            // 遍历对象中的属性
            for (const key in obj) {
                console.log('属性名:' + key);
                console.log('属性值:' + obj[key]); // 注意，因为这里的属性名 key 是变量，所以，如果想获取属性值，不能写成 obj.key，而是要写成 obj[key]
            }
        </script>
    </head>

    <body></body>
</html>
```

打印结果：

```
属性名:name
属性值:qianguyihao

属性名:age
属性值:26

属性名:gender
属性值:男

属性名:address
属性值:shenzhen

属性名:sayHi
属性值:function() {
                    console.log(this.name);
                }
```

### for in 遍历数组（不建议）

另外，for in 当然也可以用来遍历数组（只是不建议），此时的 key 是数组的索引。举例如下：

```js
const arr = ['hello1', 'hello2', 'hello3'];

for (const key in arr) {
    console.log('属性名：' + key);
    console.log('属性值：' + arr[key]);
}
```

打印结果：

```
属性名：0
属性值：hello1

属性名：1
属性值：hello2

属性名：2
属性值：hello3
```

