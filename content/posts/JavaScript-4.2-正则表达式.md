---
author: 杨盛晖
data: 2024-11-05T09:48:00+08:00
title: JavaScript-4.2-正则表达式
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>


## 正则表达式简介

**定义**：正则表达式用于定义一些字符串的规则。

**作用**：计算机可以根据正则表达式，来检查一个字符串是否符合指定的规则；或者将字符串中符合规则的内容提取出来。

如果你想查看正则更多的内容，可以查阅官方文档关于 RegExp 这个内置对象的用法。

## 创建正则表达式的对象

### 方式一：使用构造函数创建正则表达式的对象

语法：

```javascript
	var 变量 = new RegExp("正则表达式"); // 注意，参数是字符串

	var 变量 = new RegExp("正则表达式", "匹配模式"); // 注意，两个参数都是字符串
```

备注：`RegExp`的意思是 **Regular expression**。使用typeof检查正则对象，会返回object。

上面的语法中，既可以传一个参数，也可以传两个参数。

创建了正则表达式的对象后，该怎么使用呢？大致分为两个步骤：

- （1）创建正则表达式的对象 reg。

- （2）使用 reg 的test() 方法，判断指定字符串是否符合规则。

**正则表达式的`test()`方法**：【重要】

```javascript
	myReg.test(str); // 判断字符串 str 是否符合 指定的 myReg 这个正则表达式的规则
```

解释：使用`test()`这个方法可以用来检查一个字符串是否符合正则表达式的规则，**如果符合则返回true，否则返回false**。

我们来看看下面的例子。

**1、传一个参数时**：

构造函数 RegExp 中，可以只传一个参数。

代码举例：

```javascript
	var reg = new RegExp("a"); // 定义一个正则表达式：检查一个字符串中是否含有 a

	var str1 = "qianguyihao";
	var str2 = "smyh";

	// 通过 test()方法，判断字符串是否符合 上面定义的 reg 规则
	console.log(reg.test(str1)); // 打印结果：true
	console.log(reg.test(str2)); // 打印结果：false

```

注意，上面的例子中，我们是先定义了一个正则表达式的规则，然后通过正则表达式的`test()`方法来判断字符串是否符合之前定义的规则。

**2、传两个参数时**：匹配模式 【重要】

构造函数 RegExp 中，也可以传两个参数。我们可以传递一个**匹配模式**作为第二个参数。这个参数可以是：

- `i` 忽略大小写。这里的 i 指的是 ignore。

- `g` 全局匹配模式。这里的 g 指的是 global。

代码举例：

```javascript
    var reg = new RegExp('A', 'i');
    var str = 'qiangu';

    console.log(reg.test(str)); // 打印结果：true
```

### 方式二：使用字面量创建正则表达式

我们可以使用字面量来创建正则表达式。

语法：

```javascript
	var 变量 = /正则表达式/;  // 注意，这个语法里没有引号

	var 变量 = /正则表达式/匹配模式;  // 注意，这个语法里没有引号
```

代码举例：

```javascript
	var reg = /A/i; // 定义正则表达式的规则：检查一个字符串中是否含有 a。忽略大小写。
	var str = "qiangu";

	console.log(typeof reg);  // 打印结果：object
	console.log(reg.test(str)); // 打印结果：true
```

### 以上两种方式的对比

- 方式一：使用构造函数创建时，更加灵活，因为参数中还可以传递变量。

- 方式二：使用字面量的方式创建，更加简单。

代码举例：

```javascript
	var reg = new RegExp("a", "i"); // 方式一

	var reg = /a/i; // 方式二
```

上面这两行代码的作用是等价的。

### 避坑指南：全局匹配 g 慎用test()方法

对于非全局匹配的正则表达式，`test()`只会检测**是否存在某个目标字符串**（只要存在就为 true），多次检测的结果都相同。例如：

```javascript
const reg = /test/;
const str = '_test_test';

reg.test(str) // true
reg.test(str) // true
reg.test(str) // true
```

重点来了。

当设置全局标志 `/g` 时，一旦字符串中还存在匹配，test() 方法都将返回 true，同时匹配成功后将把 `lastIndex` 属性的值**设置为上次匹配成功结果之后的第一个字符所在的位置**，下次匹配将从 `lastIndex` 指示的位置开始；匹配不成功时返回 false，同时将 lastIndex 属性的值重置为 0。

举例：（很重要的例子，看仔细）

```javascript
const reg = /test/g;
const str = '_test_test';

console.log(reg.test(str)); // true
console.log(reg.lastIndex); // 5

console.log(reg.test(str)); // true
console.log(reg.lastIndex); // 10

console.log(reg.test(str)); // false
console.log(reg.lastIndex); // 0
```

**总结**：

全局匹配模式`g`一般用于 `exec()`、`match()`、`replace()`等方法。

全局匹配模式`g`如果用于test()方法会有问题。因为g模式会生成一个`lastindex`参数来存储匹配最后一次的位置。

参考链接：

- [JS正则表达式全局匹配的那些坑](https://juejin.im/post/5de9bd5fe51d45582c27b6f3)

- [javascript正则全局匹配g慎用test方法](https://blog.csdn.net/Leolu007/article/details/8576490)

- [issues](https://github.com/qianguyihao/Web/issues/69)


## 正则表达式的简单语法

### 检查一个字符串中是否包含 a或b

**写法1**：

```javascript
	var reg = /a|b/;
```

解释：使用 `|` 表示`或`的意思。

**写法2**：

```javascript
	var reg = /[ab]/;  // 跟上面的那行语法，是等价的
```

解释：这里的`[]`也是表示`或`的意思。

`[]`这个符号在正则还是比较常用的。我们接下来看几个例子。

### []表示：或

一些规则：

- `/[ab]/` 等价于 `/a|b/`：检查一个字符串中是否包含 **a或b**

- `/[a-z]/`：检查一个字符串那种是否包含**任意小写字母**

- `/[A-Z]/`：任意大写字母

- `/[A-z]/`：任意字母

- `/[0-9]/`：任意数字

- `/a[bde]c/`：检查一个字符串中是否包含 abc 或 adc 或 aec

### [^ ] 表示：除了

举例1：

```javascript
  var reg = /[^ab]/; // 规则：字符串中，除了a、b之外，还有没有其他的字符内容？
  var str = "acb";

  console.log(reg.test(str)); // 打印结果：true
```

举例2：（可以用来验证某字符串是否为 纯数字）

```javascript
	var reg = /[^0-9]/;  // 规则：字符串中，除了数字之外，还有没有其他的内容？
	var str1 = "1991";
	var str2 = "199a1";

	console.log(reg.test(str1)); // 打印结果：false （如果字符串是 纯数字，则返回 false）
	console.log(reg.test(str2)); // 打印结果：true
```

## 支持正则表达式的 String 对象的方法

 String对象的如下方法，是支持正则表达式的：

| 方法 | 描述 | 备注 |
|:-------------|:-------------|:-------------|
| split() | 将字符串拆分成数组  |   |
| search() | 搜索字符串中是否含有指定内容，返回索引 index |  |
| match() | 根据正则表达式，从一个字符串中将符合条件的内容提取出来 |  |
| replace()  | 将字符串中的指定内容，替换为新的内容并返回 |  |

下面来分别介绍和举例。

### split()

`split()`：将一个字符串拆分成一个数组。可以接受一个正则表达式作为参数。

备注：关于`split()`更详细的用法，可以看之前的关于《内置对象：String》这篇文章。

**正则相关的举例**：根据任意字母，将字符串拆分成数组。

代码实现：（通过正则）

```javascript
	var str = "1a2b3c4d5e6f7g";

	var result = str.split(/[A-z]/); // 参数是一个正则表达式：表示所有字母
	console.log(result);
```

打印结果：

```json
	["1", "2", "3", "4", "5", "6", "7", ""]
```

### search()

`search()`：搜索字符串中是否含有指定内容。如果搜索到指定内容，则会返回第一次出现的索引；否则返回-1。

`search()`方法可以接受一个正则表达式作为参数，然后会根据正则表达式去检索字符串。`serach()`只会查找第一个，即使设置全局匹配也没用。

**举例**：

```javascript
	var str = "hello abc hello aec afc";
	/*
	* 搜索字符串中是否含有abc 或 aec 或 afc
	*/
	result = str.search(/a[bef]c/);
	console.log(result); // 打印结果：6
```

### match()

`match()`：根据正则表达式，从一个字符串中将符合条件的内容提取出来，封装到一个数组中返回（即使只查询到一个结果）。

**注意**：默认情况下，`match()`方法只会找到**第一个**符合要求的内容，找到以后就停止检索。我们可以设置正则表达式为**全局匹配**模式，这样就会匹配到所有的内容，并以**数组**的形式返回。

另外，我们可以为一个正则表达式设置多个匹配模式，且匹配模式的顺序无所谓。

**代码举例**：

```javascript
	var str = "1a2a3a4a5e6f7A8B9C";

	var result1 = str.match(/[a-z]/);   // 找到符合要求的第一个内容，然后返回
	var result2 = str.match(/[a-z]/g);  // 设置为“全局匹配”模式，匹配字符串中 所有的小写字母
	var result3 = str.match(/[a-z]/gi); // 设置多个匹配模式，匹配字符串中 所有的字母（忽略大小写）

	console.log(result1); // 打印结果：["a"]
	console.log(result2); // 打印结果：["a", "a", "a", "a", "e", "f"]
	console.log(result3); // 打印结果：["a", "a", "a", "a", "e", "f", "A", "B", "C"]
```

**总结**：

match()这个方法还是很实用的，可以在一个很长的字符串中，提取出**有规则**的内容。这不就是爬虫的时候经常会遇到的场景么？

### replace()

`replace()`：将字符串中的指定内容，替换为新的内容并返回。不会修改原字符串。

语法：

```javascript
	新的字符串 = str.replace(被替换的内容，新的内容);
```

参数解释：

- 被替换的内容：可以接受一个正则表达式作为参数。

- 新的内容：默认只会替换第一个。如果需要替换全部符合条件的内容，可以设置正则表达式为**全局匹配**模式。

代码举例：

```javascript
    //replace()方法：替换
    var str2 = "Today is fine day,today is fine day !!!"

    console.log(str2);
    console.log(str2.replace("today","tomorrow"));  //只能替换第一个today
    console.log(str2.replace(/today/gi,"tomorrow")); //这里用到了正则，且为“全局匹配”模式，才能替换所有的today
```

## 常见正则表达式举例

### 检查一个字符串是否是一个合法手机号

手机号的规则：

- 以1开头（`^1` 表示1开头 , `[^1]`表示非1或除了1）

- 第二位是3~9之间任意数字

- 三位以后任意9位数字

正则实现：

```javascript
	var phoneStr = "13067890123";

	var phoneReg = /^1[3-9][0-9]{9}$/;

	console.log(phoneReg.test(phoneStr));
```

**备注**：如果在正则表达式中同时使用`^`和`$`符号，则要求字符串必须完全符合正则表达式。

### 去掉字符串开头和结尾的空格

正则实现：

```javascript
	str = str.replace(/^\s*|\s*$/g,"");
```

解释如下：

```javascript
	str = str.replace(/^\s*/, ""); //去除开头的空格

	str = str.replace(/\s*$/, ""); //去除结尾的空格
```

### 判断字符串是否为电子邮件

正则实现：

```javascript
	var emailReg = /^\w{3,}(\.\w+)*@[A-z0-9]+(\.[A-z]{2,5}){1,2}$/;

	var email = "abchello@163.com";

	console.log(emailReg.test(email));
```


