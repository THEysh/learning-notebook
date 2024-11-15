---
author: 杨盛晖
data: 2024-11-05T09:42:00+08:00
title: JavaScript-3.6-闭包
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>

## 闭包的引入

我们知道，变量根据作用域的不同分为两种：全局变量和局部变量。

- 函数内部可以访问全局变量和局部变量。

- 函数外部只能访问全局变量，不能访问局部变量。

- 当函数执行完毕，本作用域内的局部变量会销毁。

比如下面这样的代码：

```js
function foo() {
    let a = 1;
}

foo();
console.log(a); // 打印报错：Uncaught ReferenceError: a is not defined
```

上方代码中，由于变量 `a` 是函数内的局部变量，所以外部无法访问。

但是，在有些场景下，我们就是想要在函数外部访问**函数内部作用域的局部变量**，那要怎么办呢？这就引入了闭包的概念。

## 什么是闭包

### 闭包（closure）的概念

**闭包**：如果**外部作用域**有权访问另外一个**函数内部**的**局部变量**时，那就产生了闭包。这个内部函数称之为闭包函数。注意，这里强调的是访问**局部变量**。

闭包代码举例：

```js
function fun1() {
  const a = 10;
  return function fun2() {
    console.log(a);
  };
}
fun1();
// 调用外部函数，就能得到内部函数，并用 变量 result 接收
const result = fun1();
// 在 fun1函数的外部，执行了内部函数 fun2，并访问到了 fun2的内部变量a
result(); // 10
```

打印结果：

```
10
```

上方代码中，外部作用域（即全局作用域） 访问了函数 fun1 中的局部变量，那么，在 fun1 中就产生了闭包，函数 fun1是闭包函数。

全局作用域中，并没有定义变量a。正常情况下作为函数内的局部变量 a，无法被外部访问到。但是通过闭包，我们最后还是可以在全局作用域中拿到局部变量 a 的值。

注意，闭包函数是fun1，不是fun2。fun2在这里的作用是让全局作用域访问到变量a，fun2只是一个桥梁。

### 闭包的生命周期

1. 产生：内部函数fn1被声明时（即被创建时，不是被调用时）就产生了。

2. 死亡：嵌套的内部函数成为垃圾对象时。（比如fun1 = null，就可以让 fun1 成为垃圾对象）

### 在 chrome 浏览器控制台中，调试闭包

上面的代码中，要怎么验证，确实产生了闭包呢？我们可以在 chrome 浏览器的控制台中设置断点，当代码执行到 `console.log(a)`这一行的时候，如下图所示：

![](http://img.smyhvae.com/20200703_2055.png)

上图中， Local 指的是局部作用域，Global 指的是全局作用域；而 Closure 则是**闭包**，fn1 是一个闭包函数。

## 闭包的表现形式

### 形式1：将一个函数作为另一个函数的返回值

```javascript
    function fn1() {
      var a = 2

      function fn2() {
        a++
        console.log(a)
      }
      return fn2
    }

    var f = fn1();   //执行外部函数fn1，返回的是内部函数fn2
    f() // 3       //执行fn2
    f() // 4       //再次执行fn2
```


当f()第二次执行的时候，a加1了，也就说明了：闭包里的数据没有消失，而是保存在了内存中。如果没有闭包，代码执行完倒数第三行后，变量a就消失了。

上面的代码中，虽然调用了内部函数两次，但是，闭包对象只创建了一个。

也就是说，要看闭包对象创建了几个，就看：**外部函数执行了几次**（与内部函数执行几次无关）。

### 形式2：将函数作为实参传递给另一个函数调用

在定时器、事件监听、Ajax 请求、Web Workers 或者任何异步中，只要使用了回调函数，实际上就是在使用闭包。

```javascript
    function showDelay(msg, time) {
      setTimeout(function() {  //这个function是闭包，因为是嵌套的子函数，而且引用了外部函数的变量msg
        alert(msg)
      }, time)
    }
    showDelay('qianguyihao', 2000)
```

上面的代码中，闭包是里面的function，因为它是嵌套的子函数，而且引用了外部函数的变量msg。


## 闭包的作用

- 作用1：延长局部变量的生命周期。

- 作用2：让函数外部可以操作（读写）函数内部的数据（变量/函数）。

代码演示：

```javascript
function fun1() {
  let a = 2

  function fun2() {
    a++
    console.log(a)
  }
  return fun2;
}

const foo = fun1();   //执行外部函数fn1，返回的是内部函数fn2
foo() // 3       //执行fun2
foo() // 4       //再次执行fun2
```

上方代码中，foo 代表的就是整个 fun2 函数。当执行了 `foo()` 语句之后，也就执行了fun2()函数，fun1() 函数内就产生了闭包。

**作用1分析**：

一般来说，在 fn1() 函数执行完毕后，它里面的变量 a 会立即销毁。但此时由于产生了闭包，所以 **fun1 函数中的变量 a 不会立即销毁，仍然保留在内存中，因为 fn2 函数还要继续调用变量 a**。只有等所有函数把变量 a 调用完了，变量 a 才会销毁。

**作用2分析：**

在执行 `foo()`语句之后，竟然能够打印出 `3`，这就完美通过闭包实现了：全局作用域成功访问到了局部作用域中的变量 a。

达到的效果是：**外界看不到变量a，但可以操作a**。当然，如果你真想看到a，可以在fun2中将a返回即可。

## 闭包的应用场景

### 场景1：高阶函数

题目：不同的班级有不同成绩检测标准。比如：A班的合格线是60分，B 班的合格线是70分。已知某个人班级和分数，请用闭包函数判断他的成绩是否合格。

思路：创建成绩检测函数 checkStandard(n)，检查成绩 n 是否合格，函数返回布尔值。

代码实现：

```js
// 高阶函数：判断学生的分数是否合格。形参 standardTemp 为标准线
function createCheckTemp(standardTemp) {
  // 形参 n 表示具体学生的分数
  function checkTemp(n) {
    if (n >= standardTemp) {
      alert('成绩合格');
    } else {
      alert('成绩不合格');
    }
  }
  return checkTemp;
}

// 创建一个 checkStandard_A 函数，它以60分为合格线
var checkStandard_A = createCheckTemp(60);
// 再创建一个 checkStandard_B 函数，它以70分为合格线
var checkStandard_B = createCheckTemp(70);

// 调用函数
checkStandard_A(65); // 成绩合格
checkStandard_B(65); // 成绩不合格
```

对于A班来说，它的闭包函数是createCheckTemp()，闭包范围是 checkTemp()函数和参数`standardTemp = 60`。对于B班来说，它的闭包函数是全新的createCheckTemp()，闭包范围是全新的checkTemp()函数和全新的参数`standardTemp = 70`。

因为有闭包存在，所以，并不会因为 createCheckTemp() 执行完毕后就销毁 standardTemp 的值；且A班和B班的standardTemp参数不会混淆。

备注：关于“高阶函数”的更多解释，我们在以后的内容中讲解。

### 场景2：封装JS模块

闭包的第二个使用场景是：定义具有特定功能的JS模块，将所有的数据和功能都封装在一个函数内部，只向外暴露指定的对象或方法。模块的调用者，只能调用模块暴露的对象或方法来实现对应的功能。

比如有这样一个需求：定义一个私有变量a，要求a只能被进行指定操作（加减），不能进行其他操作（乘除）。在  Java、C++ 等语言中，有私有属性的概念，但在JS中只能通过闭包模拟。

我们来看看下面的代码，如何通过闭包封装JS模块。

写法1：

（1）myModule.js：（定义一个模块，向外暴露多个方法，供外界调用）

```javascript
function myModule() {
    //私有数据
    var msg = 'Qinguyihao Haha'

    //操作私有数据的函数
    function doSomething() {
        console.log('doSomething() ' + msg.toUpperCase()); //字符串大写
    }

    function doOtherthing() {
        console.log('doOtherthing() ' + msg.toLowerCase()) //字符串小写
    }

    //通过【对象字面量】的形式进行包裹，向外暴露多个函数
    return {
        doSomething1: doSomething,
        doOtherthing2: doOtherthing
    }
}
```

上方代码中，外界只能通过doSomething1和doOtherthing2来操作里面的数据，但不让外界看到里面的具体实现。

（2）index.html:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>闭包的应用-自定义JS模块</title>
</head>
<body>
<!--
闭包应用举例: 定义JS模块
  * 具有特定功能的js文件
  * 将所有的数据和功能都封装在一个函数内部(私有的)
  * 【重要】只向外暴露一个包含n个方法的对象或方法
  * 模块的使用者, 只需要调用模块暴露的对象或者方法来实现对应的功能
-->
<script type="text/javascript" src="myModule.js"></script>
<script type="text/javascript">
    var module = myModule();
    module.doSomething1();
    module.doOtherthing2();
</script>
</body>
</html>
```

写法2：

同样是实现上面的功能，我们还采取另外一种写法，写起来更方便。如下：

（1）myModule2.js：（是一个立即执行的匿名函数）

```javascript
(function () {
    //私有数据
    var msg = 'Qinguyihao Haha'

    //操作私有数据的函数
    function doSomething() {
        console.log('doSomething() ' + msg.toUpperCase())
    }

    function doOtherthing() {
        console.log('doOtherthing() ' + msg.toLowerCase())
    }

    //外部函数是即使运行的匿名函数，我们可以把两个方法直接传给window对象
    window.myModule = {
        doSomething1: doSomething,
        doOtherthing2: doOtherthing
    }
})()
```


（2）index.html：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>闭包的应用-自定义JS模块</title>
</head>
<body>
<!--
闭包的应用2 : 定义JS模块
  * 具有特定功能的js文件
  * 将所有的数据和功能都封装在一个函数内部(私有的)
  * 只向外暴露一个包信n个方法的对象或函数
  * 模块的使用者, 只需要通过模块暴露的对象调用方法来实现对应的功能
-->

<!--引入myModule文件-->
<script type="text/javascript" src="myModule2.js"></script>
<script type="text/javascript">
    myModule.doSomething1()
    myModule.doOtherthing2()
</script>
</body>
</html>

```

上方两个文件中，我们在`myModule2.js`里直接把两个方法直接传递给window对象了。于是，在index.html中引入这个js文件后，会立即执行里面的匿名函数。在index.html中把myModule直接拿来用即可。

**小结：**

写法1和写法2都采用了闭包。

## 内存溢出和内存泄露

> 我们再讲两个概念，这两个概念和闭包有些联系。

### 内存泄漏

**内存泄漏**：**占用的内存**没有及时释放。

内存泄露的次数积累多了，就容易导致内存溢出。

**常见的内存泄露**：

1、意外的全局变量

2、没有及时清理的计时器或回调函数

3、闭包


情况1举例：

```javascript
// 意外的全局变量
function fn() {
  a = new Array(10000000);
  console.log(a);
}

fn();
```

情况2举例：

```javascript
// 没有及时清理的计时器或回调函数
var intervalId = setInterval(function () { //启动循环定时器后不清理
  console.log('----')
}, 1000)

// clearInterval(intervalId);  //清理定时器
```

情况3举例：

```js
function fn1() {
  var a = 4;
  function fn2() {
    console.log(++a)
  }
  return fn2
}
var f = fn1()
f()

// f = null //让内部函数成为垃圾对象-->回收闭包
```

### 内存溢出

**内存溢出**：程序运行时出现的错误。当程序运行**需要的内存**超过**剩余的内存**时，就抛出内存溢出的错误。

代码举例：

```javascript
    var obj = {};
    for (var i = 0; i < 10000; i++) {
    obj[i] = new Array(10000000);  //把所有的数组内容都放到obj里保存，导致obj占用了很大的内存空间
    console.log("-----");
    }
```

### 闭包是否会造成内存泄漏

一般来说，答案是否定的。因为内存泄漏是非预期情况，本来想回收，但实际没回收；而闭包是预期情况，一般不会造成内存泄漏。

但如果因代码质量不高，滥用闭包，也会造成内存泄漏。

## 闭包面试题

代码举例：

```js
function addCount() {
  let count = 0;
  return function () {
    count = count + 1;
    console.log(count);
  };
}

const fun1 = addCount();
const fun2 = addCount();
fun1();
fun2();

fun1();
fun2();
```

打印结果：

```
1
1
2
2
```

代码解释：

（1）fun1 和 fun2 这两个闭包函数是互不影响的，因此第一次调用时，count变量都是0，最终各自都输出1。

（2）第二次调用时，由于闭包有记忆性，所以各自会在上一次的结果上再加1，因此输出2。


