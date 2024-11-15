---
author: 杨盛晖
data: 2024-11-05T09:25:00+08:00
title: JavaScript-1.9-数据类型转换
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>

## 前言

**变量的数据类型转换**：将一种数据类型转换为另外一种数据类型。

如果你需要在不同的数据类型之间进行某些操作，那就需要用到数据类型转换。比如：

- 将字符串类型转为数字类型。
- 将数字和字符串做减法操作。
- 判断非 Boolean类型的值，是真还是假。

通常有三种形式的数据类型转换：

-   转换为字符串类型

-   转换为数字型

-   转换为布尔型

我需要专门把某个数据类型转换成 null 或者 undefined 吗？因为这样做没有意义。

## 变量的类型转换的分类

类型转换分为两种：显式类型转换、隐式类型转换。

### 显式类型转换

显式类型转换：**手动**将某种数据类型，**强制**转换为另一种数据类型。也就是说，通过调用特定函数或运算符显式地将一个数据类型转换为另一个数据类型。

常见的显示类型转换方法，有这几种：

-   toString()

-   String()

-   Number()

-   parseInt(string)

-   parseFloat(string)

-   Boolean()

### 隐式类型转换

隐式类型转换：这是JS在运行时会**自动执行**的一种类型转换，不需要明确的代码指示。JS 在某些情况下会隐式地将一个数据类型转换为另一个数据类型，以完成某些操作或比较。

常见的隐式类型转换，包括下面这几种：

-   isNaN() 函数
-   自增/自减运算符：`++`、`—-`
-   运算符：正号`+a`、负号`-a`
-   运算符：加号`+`
-   运算符：`-`、`*`、`/`、`%`
-   比较运算符：`<`、`>`、 `<=`、 `>=`、`==`等。比较运算符的运算结果都是布尔值：要么是 true，要么是 false。
-   逻辑运算符：`&&`、`||`、`!` 。非布尔值进行**与或**运算时，会先将其转换为布尔值，然后再运算。`&&`、`||`的运算结果是**原值**，`!`的运算结果为布尔值。

重点：**隐式类型转换，内部调用的都是显式类型转换的方法**。

接下来详细讲讲各种数据类型转换。

## 一、转换为 String

### 1、调用 toString() 方法

语法：

```javascript
变量.toString();
常量.toString(); // 这里的常量，不要直接写数字，但可以是其它常量；下文会具体讲。

// 或者用一个新的变量接收转换结果
var result = 变量.toString();
```

该方法**不会影响到原变量**，它会将转换的结果返回。当然我们还可以直接写成`a = a.toString()`，这样的话，就是直接修改原变量。

当我们对一个字符串字面量调用 toString() 方法时，它实际上是调用了 String 构造函数，并将字符串字面量转换为一个 String 对象，然后调用该对象的 toString() 方法。String 对象的 toString() 方法返回调用它的原始字符串值。

举例：

```js
// 基本数据类型
var a1 = 'qianguyihao';
var a2 = 29;
var a3 = true;

// 引用数据类型
var a4 = [1, 2, 3];
var a5 = { name: 'qianguyihao', age: 29 };

// undefined 和 null
var a6 = null;
var a7 = undefined;

// 打印结果都是字符串
console.log(a1.toString()); // "qianguyihao"
console.log(a2.toString()); // "29"
console.log(a3.toString()); // "true"
console.log(a4.toString()); // "1,2,3"
console.log(a5.toString()); // "object"

// 下面这两个，打印报错
console.log(a6.toString()); // 报错：Uncaught TypeError: Cannot read properties of undefined'
console.log(a7.toString()); // 报错：Uncaught TypeError: Cannot read properties of null
```

小技巧：在 chrome 浏览器的控制台中，Number类型、Boolean类型的打印结果是蓝色的，String类型的打印结果是黑色的。

一起来看看 toString() 的注意事项：

（1）undefined 和 null 这两个值没有 toString() 方法，所以它们不能用 toString() 。如果调用，会报错。

```js
console.log(undefined.toString());
console.log(null.toString());
```

![](https://img.smyhvae.com/20211116_1458.png)

如果你不确定一个值是不是`null`或`undefined`，可以使用`String()`函数，下一小段会讲。

（2）多数情况下，`toString()`不接收任何参数；当然也有例外：Number 类型的变量，在调用 toString()时，可以在方法中传递一个整数作为参数。此时它会把数字转换为指定的进制，如果不指定则默认转换为 10 进制。例如：

```javascript
var a = 255;

//Number数值在调用toString()时，可以在方法中传递一个整数作为参数
//此时它将会把数字转换为指定的进制,如果不指定则默认转换为10进制
a = a.toString(2); // 转换为二进制

console.log(a); // "11111111"
console.log(typeof a); // string
```

（3）纯小数的小数点后面，如果紧跟连续6个或6个以上的“0”时，那么，将用e来表示这个小数。代码举例：

```js
const num1 = 0.000001; // 小数点后面紧跟五个零
console.log(num1.toString()); // 打印结果："0.000001"

const num2 = 0.0000001; // 小数点后面紧跟六个零
console.log(num2.toString()); // 【重点关注】打印结果："1e-7"

const num3 = 1.0000001;
console.log(num3.toString()); // 打印结果："1.0000001"

const num4 = 0.10000001;
console.log(num4.toString()); // 打印结果："0.10000001"
```

（4）常量可以直接调用 toString() 方法，但这里的常量，不允许直接写数字。举例如下：

```js
1.toString(); // 注意，会报错
1..toString(); // 合法。得到的结果是字符串"1"
1.2.toString(); // 合法。得到的结果是字符串"1.2"
(1).toString(); // 合法。得到的结果是字符串"1"
'1'.toString(); // 合法。得到的结果是字符串"1"
```

上方代码中，为何出现这样的打印结果？这是因为：

- 第一行代码：JS引擎认为`1.toString()`中的`.`是小数点，**是数字字面量的一部分，而不是方法调用的分隔符**。小数点后面的字符是非法的。
- 第二行、第三行代码：JS引擎认为第一个`.`是小数点，第二个`.`是属性访问的语法，所以能正常解释实行。
- 第四行代码：用`()`排除了`.`被视为小数点的语法解释，所以这种写法也能正常解释执行。

小结：因为点号（.）被解释为数字字面量的一部分，而不是方法调用的分隔符。为了正确调用 toString 方法，可以使用括号或额外的点号。

如果想让数字调 toString() 方法，更推荐的做法是先把数字放到变量中存起来，然后通过变量调用 toString()。举例：

```js
const a = 1;
a.toString(); // 合法。得到的结果是字符串"1"
```


参考链接：[你不知道的toString方法](https://www.jianshu.com/p/88570529a03c)

（5）既然常量没有方法，那它为什么可以调用 toString() 呢？这是因为，除了 undefined、null 之外，其他的常量都有对应的特殊的引用类型——**基本包装类型**，所以代码在解释执行的时候，会将常量转为基本包装类型，这样就可以调用相应的引用类型的方法。

我们在后续的内容《JavaScritpt基础/基本包装类型》中会专门讲到基本包装类型。

### 2、使用 String() 函数

语法：

```javascript
String(变量/常量);
```

该方法**不会影响到原数值**，它会将转换的结果返回。

使用 String()函数做强制类型转换时：

-   对于 Number、Boolean、String、Object 而言，本质上就是调用 toString()方法，返回结果同 toString()方法。
-   但是对于 null 和 undefined，则不会调用 toString() 方法。它会，将 undefined 直接转换为 "undefined"，将 null 直接转换为 "null"。

使用String()函数转为字符串的规则如下：

| 原始值              | 转换后的值              |
| ------------------- | ----------------------- |
| 布尔值：true、false | 字符串：'true'、'false' |
| 数字                | 字符串                  |
| undefined           | 字符串：'undefined'     |
| null                | 字符串：'null'          |
| 对象                | 字符串：'object'        |

### 3、隐式类型转换：字符串拼接

如果加号的两边有一个是字符串，则另一边会自动转换成字符串类型进行拼接。

字符串拼接的格式：变量+"" 或者 变量+"abc"

举例：

```javascript
var a = 123; // Number 类型
// 使用空字符串进行拼接
console.log(a + ''); // 打印结果："123"
// 使用普通字符串进行拼接
console.log(a + 'haha'); // 打印结果："123haha"
```

上面的例子中，打印的结果，都是字符串类型的数据。实际上底层是调用的 String() 函数。

### 4、prompt()：用户的输入

我们在前面的《JavaScript基础/02-JavaScript书写方式：hello world》就讲过，`prompt()`就是专门用来弹出能够让用户输入的对话框。重要的是：用户不管输入什么，都当字符串处理。

## 二、转换为 Number

### 1、使用 Number() 函数

语法：

```js
const result = Number(变量/常量);
```

使用 Number() 函数转为数字的规则如下：

| 原始值    | 转换后的值                                                   |
| --------- | ------------------------------------------------------------ |
| 字符串    | （1）字符串去掉首尾空格后，剩余字符串的内容如果是纯数字，则直接将其转换为数字。<br/>（2）字符串去掉首尾空格后，剩余字符串包的内容只要含了其他非数字的内容（`小数点`按数字来算），则转换为 NaN。怎么理解这里的 **NaN** 呢？可以这样理解，使用 Number() 函数之后，**如果无法转换为数字，就会转换为 NaN**。<br />（3）如果字符串是一个**空串**或者是一个**全是空格**的字符串，则转换为 0。<br/> |
| 布尔值    | true 转成 1；false 转成 0                                    |
| undefined | NaN                                                          |
| null      | 0                                                            |

### 2、隐式类型转换——运算符：加号 `+`

（1）**字符串 + 其他数据类型 = 字符串**

任何数据类型和字符串做加法运算，都会先自动将那个数据类型调用 String() 函数转换为字符串，然后再做拼串操作。最终的运算结果是字符串。

比如：

```javascript
result1 = 1 + 2 + '3'; // 字符串：33

result2 = '1' + 2 + 3; // 字符串：123
```

某些函数在执行时也会自动将参数转为字符串类型，比如 `console.log()`函数。

（2）**Boolean + 数字 = 数字**

Boolean 型和数字型相加时， true 按 1 来算 ，false 按 0 来算。这里其实是先调 Number() 函数，将 Boolean 类型转为 Number 类型，然后再和 数字相加。

（3）**null + 数字 = 数字**

等价于：0 + 数字

（4）**undefined + 数字 = NaN**

计算结果：NaN

（5）任何值和 **NaN** 运算的结果都是 NaN。

### 3、隐式类型转换——运算符：`-`、`*`、`/`、`%`

任何非 Number 类型的值做`-`、`*`、`/`、`%`运算时，会将这些值转换为 Number 然后再运算(内部调用的是 Number() 函数），运算结果是 Number 类型。

任何数据和 NaN进行运算，结果都是NaN。

比如：

```js
var result1 = 100 - '1'; // 99

var result2 = true + NaN; // NaN
```

### 4、隐式类型转换：正负号 `+a`、`-a`

> 注意，这里说的是正号/负号，不是加号/减号。

任何值做`+a`、`-a`运算时， 底层调用的是 Number() 函数。不会改变原数值；得到的结果，会改变正负性。

代码举例：

```js
const a1 = '123';
console.log(+a1); // 123
console.log(-a1); // -123

const a2 = '123abc';
console.log(+a2); // NaN
console.log(-a2); // NaN

const a3 = true;
console.log(+a3); // 1
console.log(-a3); // -1


const a4 = false;
console.log(+a4); // 0
console.log(-a4); // -0

const a5 = null;
console.log(+a5); // 0
console.log(-a5); // -0

const a6 = undefined;
console.log(+a6); // NaN
console.log(-a6); // NaN
```



### 5、使用 parseInt()函数：字符串 -> 整数

语法：

```js
const result = parseInt(需要转换的字符串)
```

**parseInt()**：将传入的数据当作**字符串**来处理，从左至右提取数值，一旦遇到非数值就立即停止；停止时如果还没有提取到数值，就返回NaN。

parse 表示“转换”，Int 表示“整数”。例如：

```javascript
parseInt('5'); // 得到的结果是数字 5
```

按照上面的规律，使用 parseInt() 函数转为数字的规则如下：

| 原始值              | 转换后的值                                                   |
| ------------------- | ------------------------------------------------------------ |
| 字符串              | （1）**只保留字符串最开头的数字**，后面的中文自动消失。<br/>（2）如果字符串不是以数字开头，则转换为 NaN。<br/>（3）如果字符串是一个空串或者是一个全是空格的字符串，转换时会报错。 |
| 布尔值：true、false | NaN                                                          |
| undefined           | NaN                                                          |
| null                | NaN                                                          |

Number() 函数和 parseInt() 函数的区别：

就拿`Number()` 和 `parseInt()/parseFloat()`来举例，二者在使用时，是有区别的：

-   Number() ：千方百计地想转换为数字；如果转换不了则返回 NaN。

-   parseInt()/parseFloat() ：提取出最前面的数字部分（开头如果是空格，则自动忽略空格）；没提取出来，那就返回 NaN。

**parseInt()具有以下特性**：

（1）parseInt()、parseFloat()会将传入的数据当作**字符串**来处理。也就是说，如果对**非 String**使用 parseInt()、parseFloat()，它会**先将其转换为 String** 然后再操作。【重要】

比如：

```javascript
var a = 168.23;
console.log(parseInt(a)); //打印结果：168  （因为是先将 a 转为字符串"168.23"，然后然后再操作）

var b = true;
console.log(parseInt(b)); //打印结果：NaN （因为是先将 b 转为字符串"true"，然后然后再操作）

var c = null;
console.log(parseInt(c)); //打印结果：NaN  （因为是先将 c 转为字符串"null"，然后然后再操作）

var d = undefined;
console.log(parseInt(d)); //打印结果：NaN  （因为是先将 d 转为字符串"undefined"，然后然后再操作）
```


（2）**只保留字符串最开头的数字**，后面的中文自动消失。例如：

```javascript
console.log(parseInt('2017在公众号上写了6篇文章')); //打印结果：2017

console.log(parseInt('2017.01在公众号上写了6篇文章')); //打印结果仍是：2017   （说明只会取整数）

console.log(parseInt('aaa2017.01在公众号上写了6篇文章')); //打印结果：NaN （因为不是以数字开头）
```


（3）自动截断小数：**取整，不四舍五入**。

例 1：

```javascript
var a = parseInt(5.8) + parseInt(4.7);
console.log(a);
```

打印结果：

```
9
```

例 2：

```javascript
var a = parseInt(5.8 + 4.7);
console.log(a);
```

打印结果：

```javascript
10;
```

（4）带两个参数时，表示在转换时，包含了进制转换。

代码举例：

```javascript
var a = '110';

var num = parseInt(a, 16); // 【重要】将 a 当成 十六进制 来看待，转换成 十进制 的 num

console.log(num);
```

打印结果：

```
272
```

如果你对打印结果感到震惊，请仔细看上面的代码注释。就是说，无论 parseInt() 里面的进制参数是多少，最终的转换结果是十进制。

我们知道，八进制的数字是用0开头进行表示。比如`"070"`这个字符串，如果我调用 parseInt() 转成数字时，有些浏览器会当成 8 进制解析，有些会当成 10 进制解析。

所以，比较建议的做法是：可以在 parseInt()中传递第二个参数，来指定当前数字的进制。例如：

```javascript
var a = '070';

a = parseInt(a, 8); //将 070 当成八进制来看待，转换结果为十进制。
console.log(a); // 打印结果：56。这个地方要好好理解。
```

我们来看下面的代码，打印结果继续震惊：

```javascript
var a = '5';

var num = parseInt(a, 2); // 将 a 当成 二进制 来看待，转换成 十进制 的 num

console.log(num); // 打印结果：NaN。因为 二进制中没有 5 这个数，转换失败。
```

### 6、parseFloat()函数：字符串 --> 浮点数（小数）

parseFloat()的作用是：将字符串转换为**浮点数**。

parseFloat()和 parseInt()的作用类似，不同的是，parseFloat()可以获得小数部分。

代码举例：

```javascript
var a = '123.456.789px';
console.log(parseFloat(a)); // 打印结果：123.456
```

parseFloat() 的几个特性，可以参照 parseInt()。

## 三、转换为 Boolean

### 转换结果列举【重要】

其他的数据类型都可以转换为 Boolean 类型。无论是隐式转换，还是显示转换，转换结果都是一样的。有下面几种情况：

转换为 Boolean 类型的规则如下：

| 原始值    | 转换后的值                                                   |
| --------- | ------------------------------------------------------------ |
| 字符串    | 空串的转换结果是false，其余的都是 true。<br />全是空格的字符串，转换结果也是 true。<br />字符串`'0'`的转换结果也是 true。 |
| 数字      | 0 和 NaN的转换结果 false，其余的都是 true。比如 `Boolean(NaN)`的结果是 false。 |
| undefined | false                                                        |
| null      | false                                                        |
| 对象      | 引用数据类型会转换为 true。<br />注意，空数组`[]`和空对象`{}`，**转换结果也是 true**，这一点，很多人不知道。 |

小结：空字符串''、0、NaN、undefined、null会转换为 false；其他值会转换为 true。

**重中之重来了：**

转换为 Boolean 的上面这几种情况，**极其重要**，项目开发中会频繁用到。比如说，我们在项目开发中，经常需要对一些**非布尔值**做**逻辑判断或者逻辑运算**，符合条件后，才做下一步的事情。这个逻辑判断就是依据上面的四种情况。

举例：（接口返回的内容不为空，前端才做进一步的事情）

```js
const result1 = '';
const result2 = { a: 'data1', b: 'data2' };

// 逻辑判断
if (result1) {
    console.log('因为 result1的内容为空，所以代码进不了这里');
}

// 逻辑运算
if (result2 && result2.a) {
    // 接口返回了 result2，且 result2.a 里面有值，前端才做进一步的事情
    console.log('代码能进来，前端继续在这里干活儿');
}
```

这里再次强调一下，空数组`[]`和空对象`{}`转换为 Boolean 值时，转换结果为 true。

我们在下一篇内容《运算符》中，还会详细讲非布尔值的逻辑运算。

### 1. 隐式类型转换：逻辑运算

当非 Boolean 类型的数值和 Boolean 类型的数值做比较时，会先把前者**临时**进行隐式转换为 Boolean 类型，然后再做比较；且不会改变前者的数据类型。举例如下：

```js
const a = 1;

console.log(a == true); // 打印结果：true
console.log(typeof a); // 打印结果：number。可见，上面一行代码里，a 做了隐式类型转换，但是 a 的数据类型并没有发生变化，仍然是 Number 类型

console.log(0 == true); // 打印结果：false
```

### 2. 使用 `!!`

使用 `!!`可以显式转换为 Boolean 类型。比如 `!!3`的结果是 true。

### 3. 使用  Boolean()函数

使用 Boolean()函数可以显式转换为 Boolean 类型。

## 隐式类型转换：isNaN() 函数

语法：

```javascript
isNaN(参数)
```

解释：判断指定的参数是否**不是数字**（NaN，非数字类型），返回结果为 Boolean 类型。**不是数字时返回 true**，是数字时返回 false。

在做判断时，会进行隐式类型转换。也就是说：**任何不能被转换为数值的参数，都会让这个函数返回 true**。

**执行过程**：

（1）先调用`Number(参数)`函数；

（2）然后判断`Number(参数)`的返回结果是否为数值。如果不为数字，则返回结果为 true；如果为数字，则返回结果为 false。

代码举例：

```javascript
console.log(isNaN('123')); // 返回结果：false。

console.log(isNaN(null)); // 返回结果：false

console.log(isNaN('abc')); // 返回结果：true。因为 Number('abc') 的返回结果是 NaN

console.log(isNaN(undefined)); // 返回结果：true

console.log(isNaN(NaN)); // 返回结果：true
```

