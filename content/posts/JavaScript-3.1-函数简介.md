---
author: 杨盛晖
data: 2024-11-05T09:36:00+08:00
title: JavaScript-3.1-函数简介
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>


## 函数的介绍

函数：就是一些功能或语句的**封装**。在需要的时候，通过**调用**的形式，执行这些语句。

补充：

- **函数也是一个对象**

- 使用`typeof`检查一个函数对象时，会返回 function

**函数的作用**：

- 一次定义，多次调用。将大量重复的语句抽取出来，写在函数里，以后需要这些语句时，可以直接调用函数，避免重复劳动。

- 简化代码，可读性更强，让编程模块化。高内聚、低耦合。

来看个例子：

```javascript
console.log("你好");
sayHello();	// 调用函数
sayHello();	// 再调用一次函数



// 定义函数
function sayHello(){
	console.log("欢迎");
	console.log("welcome");
}
```

## 函数的定义/声明

我们使用`function`关键字定义函数，中文含义是“函数”、“功能”。可以使用如下方式进行定义。

### 方式一：函数声明（命名函数）

使用`函数声明`来创建一个函数。语法：

```javascript
function 函数名([形参1,形参2...形参N]){  // 备注：语法中的中括号，表示“可选”
	// 函数体语句
}
```

举例：

```javascript
function sum(a, b){
	return a+b;
}
```

解释如下：

- 函数名：命名规定和变量的命名规定一样，必须符合JS标识符的命名规则。只能是字母、数字、下划线、美元符号，不能以数字开头。

- 圆括号里，是形参列表，可选。即使没有形参，也必须书写圆括号。

- 大括号里，是函数体语句。

PS：在有些编辑器中，方法写完之后，我们在方法的前面输入`/**`，然后回车，会发现，注释的格式会自动补齐。

### 方式二：函数表达式（匿名函数）

使用`函数表达式`来创建一个函数。语法：

```javascript
const 变量名  = function([形参1,形参2...形参N]){
	语句....
}
```

举例：

```javascript
const fun2 = function() {
	console.log("我是匿名函数中封装的代码");
};
```

解释如下：


- 上面的 fun2 是变量名，不是函数名。

- 函数表达式的声明方式跟声明变量类似，只不过变量里存的是值，而函数表达式里存的是函数。

- 函数表达式也可以传递参数。

从方式二的举例中可以看出：所谓的“函数表达式”，其实就是将匿名函数赋值给一个变量。因为，一个匿名函数终究还是要给它一个接收对象，进而方便地调用这个函数。

### 方式三：使用构造函数 new Function()

使用构造函数`new Function()`来创建一个对象。这种方式，用的少。

语法：

```javascript
const 变量名/函数名  = new Function('形参1', '形参2', '函数体');
```

注意，Function 里面的参数都必须是**字符串**格式。也就是说，形参也必须放在**字符串**里；函数体也是放在**字符串**里包裹起来，放在 Function 的最后一个参数的位置。

代码举例：

```javascript
const fun3 = new Function('a', 'b', 'console.log("我是函数内部的内容");  console.log(a + b);');

fun3(1, 2); // 调用函数
```

打印结果：

```
我是函数内部的内容
3
```

**分析**：

方式3的写法很少用，原因如下：

- 不方便书写：写法过于啰嗦和麻烦。

- 执行效率较低：首先需要把字符串转换为 js 代码，然后再执行。

### 小结

1、**所有的函数，都是 `Fuction` 的“实例”**（或者说是“实例对象”）。函数本质上都是通过 new Function 得到的。

2、函数既然是实例对象，那么，**函数也属于“对象”**。还可以通过如下特征，来佐证函数属于对象：

（1）我们直接打印某一个函数，比如 `console.log(fun2)`，发现它的里面有`__proto__`。（这个是属于原型的知识，后续再讲）

（2）我们还可以打印 `console.log(fun2 instanceof Object)`，发现打印结果为 `true`。这说明 fun2 函数就是属于 Object。

## 函数的调用

调用函数即：执行函数体中的语句。函数必须要等到被调用时才执行。

### 方式1：普通函数的调用

函数调用的语法：

```javascript
// 写法1（最常用）
函数名();

// 写法2
函数名.call();
```



代码举例：

```javascript
function fn1() {
	console.log('我是函数体里面的内容1');
}

function fn2() {
	console.log('我是函数体里面的内容2');
}

fn1(); // 调用函数

fn2.call(); // 调用函数

```

### 方式2：通过对象的方法来调用

```javascript
var obj = {
	a: 'qianguyihao',
	fn2: function() {
		console.log('千古壹号，永不止步!');
	},
};

obj.fn2(); // 调用函数
```

如果一个函数是作为一个对象的属性保存，那么，我们称这个函数是这个对象的**方法**。

PS：关于函数和方法的区别，本文的后续内容里有讲到，可以往下面翻。


### 方式3：立即执行函数

代码举例：

```javascript
(function() {
	console.log('我是立即执行函数');
})();

```

立即执行函数在定义后，会自动调用。

PS：关于立即执行函数，本文的后续内容里有讲到，可以往下面翻。


上面讲到的这三种方式，是用得最多的。接下来讲到的三种方式，暂时看不懂也没关系，可以等学完其他的知识点，再回过头来看。

### 方式4：通过构造函数来调用

代码举例：

```javascript
function Fun3() {
	console.log('千古壹号，永不止步~');
}

new Fun3();
```

这种方式用得不多。

### 方式5：绑定事件函数

代码举例：


```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Document</title>
    </head>
    <body>
        <div id="btn">我是按钮，请点击我</div>

        <script>
            var btn = document.getElementById('btn');
            //2.绑定事件
            btn.onclick = function() {
                console.log('点击按钮后，要做的事情');
            };
        </script>
    </body>
</html>

```

这里涉及到DOM操作和事件的知识点，后续再讲。

### 方式6：定时器函数

代码举例：（每间隔一秒，将 数字 加1）

```javascript
    let num = 1;
   setInterval(function () {
       num ++;
       console.log(num);
   }, 1000);
```

这里涉及到定时器的知识点。

## 函数的参数：形参和实参

### 定义

函数的参数包括形参和实参。形参是函数内的一些**待定值**。在调用函数时，需传入这些参数的具体值（即实参）。

可以在函数的`()`中指定一个或多个参数，也可以不指定参数。多个参数之间用英文逗号隔开。

举例：

```js
// a, b 是形参，表示待定值
function add(a, b) {
const sum = a + b;
console.log(sum);
}

// 1, 2 是实参，表示传入的具体值。调用函数时，传入实参
add(1, 2);
```

**形参：**

- 概念：形式上的参数。定义函数时传递的待定值（此时并不知道是什么值）。
- 声明形参相当于在函数内部声明了变量，但并不赋值。也可以说，**形参的默认值是 undefined**。

**实参**：

- 概念：实际上的参数。调用函数时传递的具体值。实参将传递给函数中对应的形参。

举例：

```javascript
// 调用函数
add(3, 4);
add("3", 4);
add("Hello", "World");

// 定义函数：求和
function add(a, b) {
	console.log(a + b);
}
```

控制台输出结果：

```
7
34
helloworld
```

### 形参和实参的个数

实际参数和形式参数的个数，可以不同。调用函数时，解析器不会检查实参的数量。

- 如果实参个数 > 形参个数，则末尾的实参是多余的，不会被赋值，因为没有形参能接收它。
- 如果实参个数 < 形参个数，则末尾的形参是多余的，值是 undefined，因为它没有接收到实参。（undefined参与运算时，表达式的运算结果为NaN）

代码举例：

```javascript
	function sum(a, b) {
		console.log(a + b);
	}

	sum(1, 2);
	sum(1, 2, 3);
	sum(1);
```

打印结果：

```
3
3
NaN
```

### 实参的数据类型

函数的实参可以是任意的数据类型。调用函数时，解析器不会检查实参类型，所以要注意，是否有可能会接收到非法的参数，如果有可能则需要对参数进行类型检查。

## 函数的返回值

### return 关键字

函数体内可以没有返回值，也可以根据需要加返回值。语法格式：`return 函数的返回值`。

举例：

```javascript
console.log(sum(3, 4)); // 将函数的返回值打印出来

//函数：求和
function sum(a, b) {
	return a + b;
}
```

return关键字的作用既可以是**终止函数**，也可以给函数添加返回值。

解释：

（1）return 后的返回值将会作为函数的执行结果返回，可以定义一个变量，来接收该返回值。

（2）在函数中，return后的语句都不会执行。也就是说，函数在执行完 return 语句之后，会立即退出函数。

（3）如果return语句后不跟任何值，就相当于返回一个undefined

（4）如果函数中不写return，则也会返回undefined

（5）返回值可以是任意的数据类型，可以是对象，也可以是函数。

（6）return 只能返回一个值。如果用逗号隔开多个值，则以最后一个为准。

### break、continue、return 的区别

- break ：结束当前的循环体（如 for、while）

- continue ：跳出本次循环，继续执行下次循环（如 for、while）

- return ：1、退出循环。2、返回 return 语句中的值，同时结束当前的函数体内的代码，退出当前函数。

## 函数名、函数体和函数加载问题（重要，请记住）

我们要记住：**函数名 == 整个函数**。举例：

```javascript
console.log(fn) == console.log(function fn(){alert(1)});

//定义fn方法
function fn(){
	alert(1)
};
```

我们知道，当我们在调用一个函数时，通常使用`函数()`这种格式；可如果，我们是直接使用`函数`这种格式，它的作用相当于整个函数。

**函数的加载问题**：JS加载的时候，只加载函数名，不加载函数体。所以如果想使用内部的成员变量，需要调用函数。

### fn()  和 fn 的区别【重要】

- `fn()`：调用函数。调用之后，还获取了函数的返回值。

- `fn`：函数对象。相当于直接获取了整个函数对象。


## 方法

函数也可以成为对象的属性。**如果一个函数是作为一个对象的属性保存，那么，我们称这个函数是这个对象的方法**。

调用这个函数就说调用对象的方法（method）。函数和方法，有什么本质的区别吗？它只是名称上的区别，并没有其他的区别。

函数举例：

```javascript
	// 调用函数
	fn();
```

方法举例：

```javascript
	// 调用方法
	obj.fn();
```

我们可以这样说，如果直接是`fn()`，那就说明是函数调用。如果是`XX.fn()`的这种形式，那就说明是**方法**调用。

## 类数组对象 arguments

> 这部分，初学者可能看不懂，可以以后再来看。

在调用函数时，浏览器每次都会传递进两个隐含的参数：

- 1.函数的上下文对象 this

- 2.**封装实参的对象** arguments

这一段，我们来讲一下 arguments。例如：

```javascript
function foo() {
    console.log(arguments);
    console.log(typeof arguments);
}

foo('a', 'b');
```

打印结果：

![](https://img.smyhvae.com/20220725_2000.png)


### 定义

函数内的 arguments 是一个**类数组对象**，里面存储的是它接收到的**实参列表**。所有函数都内置了一个 arguments 对象，有个讲究的地方是：只有函数才有arguments。

具体来说，在调用函数时，我们所传递的实参都会在 arguments 中保存。**arguments 代表的是所有实参**。

arguments 的展示形式是一个**伪数组**。意思是，它和数组有点像，但它并不是数组。它具有以下特点：

- 可以进行遍历；具有数组的 length 属性，可以获取长度。

- 可以通过索引（从0开始计数）存储数据、获取和操作数据。比如，我们可以通过索引访问某个实参。

- 不能调用数组的方法。比如push()、pop() 等方法都没有。

我们看一下 arguments 的使用。

### arguments.length 返回函数实参的个数

arguments.length 可以用来获取**实参的个数**。

举例：

```javascript
fn(2, 4);
fn(2, 4, 6);
fn(2, 4, 6, 8);

function fn(a, b) {
    console.log(arguments);
    console.log(fn.length); //获取形参的个数
    console.log(arguments.length); //获取实参的个数

    console.log('----------------');
}
```

打印结果：

![](http://img.smyhvae.com/20180125_2140.png)

此外，即使我们不定义形参，也可以通过 arguments 来获取实参：arguments[0] 表示第一个实参、arguments[1] 表示第二个实参，以此类推。

举例：将传入的实参进行求和，无论实参的个数有多少。代码实现：

```js
function foo() {
  let sum = 0;
  for (let i = 0; i < arguments.length; i++) {
    sum += arguments[i];
  }
  return sum;
}

const result = foo(1, 2);
console.log(result);
```


### arguments.callee 返回正在执行的函数

arguments 里边有一个属性叫做 callee，这个属性对应一个函数对象，就是当前正在指向的函数对象。

```javascript
function fun() {
    console.log(arguments.callee == fun); // 打印结果为true
}

fun('hello');
```

在使用函数**递归**调用时，推荐使用 arguments.callee 代替函数名本身。

### arguments 可以修改元素

arguments 还可以**修改元素，但不能改变数组的长度**。举例：

```javascript
fn(2, 4);
fn(2, 4, 6);
fn(2, 4, 6, 8);

function fn(a, b) {
    arguments[0] = 99; // 将实参的第一个数改为99
    arguments.push(8); // 此方法不通过，因为无法增加元素
}
```

### 使用场景举例

当我们不确定有多少个参数传递的时候，可以用 **arguments** 来获取。

**举例**：利用 arguments 求函数实参中的最大值。

代码实现：

```javascript
	function getMaxValue() {
		var max = arguments[0];
		// 通过 arguments 遍历实参
		for (var i = 0; i < arguments.length; i++) {
			if (max < arguments[i]) {
				max = arguments[i];
			}
		}
		return max;
	}

	console.log(getMaxValue(1, 3, 7, 5));

```


