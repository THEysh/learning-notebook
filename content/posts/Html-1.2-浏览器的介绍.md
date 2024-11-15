---
author: 杨盛晖
data: 2024-11-06T09:18:00+08:00
title: Html-1.2-浏览器的介绍
featured: true
draft: false
tags: ['html']
categories: ['web']
---



<ArticleTopAd></ArticleTopAd>



## 常见的浏览器

浏览器是网页运行的平台，常见的浏览器有谷歌（Chrome）、Safari、火狐（Firefox）、IE、Edge、Opera等。如下图所示：

![](https://pic.imgdb.cn/item/66e3eec3d9c307b7e99dadb5.png)

我们重点需要学习的是 Chrome 浏览器。


## 浏览器的组成

浏览器分成两部分：

- 1、渲染引擎（即：浏览器内核）

- 2、JS 引擎

### 1、渲染引擎（浏览器内核）

浏览器所采用的「渲染引擎」也称之为「浏览器内核」，用于解析 HTML和CSS、布局、渲染等工作。渲染引擎决定了浏览器如何显示网页的内容以及页面的格式信息。

**渲染引擎是浏览器兼容性问题出现的根本原因。**

渲染引擎的英文叫做 Rendering Engine。通俗来说，它的作用就是：读取网页内容，计算网页的显示方式并显示在页面上。

常见浏览器的内核如下：

|浏览器 | 内核|
|:-------------:|:-------------:|
| chrome | Blink  |
| 欧鹏  | Blink  |
|360安全浏览器| Blink|
|360极速浏览器| Blink|
|Safari|Webkit|
|Firefox 火狐|Gecko|
|IE| Trident |

备注：360的浏览器，以前使用的IE浏览器的Trident内核，但是现在已经改为使用 chrome 浏览器的 Blink内核。

另外，移动端的浏览器内核是什么？大家可以自行查阅资料。


### 2、JS 引擎

也称为 JS 解释器。 用来解析和执行网页中的JavaScript代码。

浏览器本身并不会执行JS代码，而是通过内置 JavaScript 引擎(解释器) 来执行 JS 代码 。JS 引擎执行代码时会逐行解释每一句源码，转换为机器语言，然后由计算机去执行。

常见浏览器的 JS 引擎如下：

|浏览器 | JS 引擎|
|:-------------:|:-------------|
|chrome、欧鹏   | V8   |
|Mozilla Firefox 火狐|SpiderMonkey（1.0-3.0）/ TraceMonkey（3.5-3.6）/ JaegerMonkey（4.0-）|
|Safari|JavaScriptCore，也称为Nitro，是 WebKit 引擎的一部分|
|IE|Trident |
|Edge|Chakra。此外，ChakraCore是Chakra的开源版本，可以在不同的平台上使用。 |
|Opera|Linear A（4.0-6.1）/ Linear B（7.0-9.2）/ Futhark（9.5-10.2）/ Carakan（10.5-）|

补充说明：

1、SpiderMonkey 是第一款 JavaScript 引擎，由 JS语言的作者 Brendan Eich 开发。

2、先以WebKit为例，WebKit上由两部分组成：

- WebCore：负责解析HTML和CSS、布局、渲染等工作。
- JavaScriptCore：负责解析和执行JavaScript 代码。

参考链接：

- [主流浏览器内核及JS引擎](https://juejin.im/post/5ada727c518825670b33a584)

## 浏览器工作原理

> 这一小段有些深入，小白可以暂时跳过，以后学习JS的时候再回来看。

浏览器主要由下面这个七个部分组成：

![](http://img.smyhvae.com/20180124_1700.png)

1、User Interface（UI界面）：包括地址栏、前进/后退按钮、书签菜单等。也就是浏览器主窗口之外的其他部分。

2、Browser engine （浏览器引擎）：用来查询和操作渲染引擎。是UI界面和渲染引擎之间的**桥梁**。

3、Rendering engine（渲染引擎）：用于解析HTML和CSS，并将解析后的内容显示在浏览器上。

4、Networking （网络模块）：用于发送网络请求。

5、JavaScript Interpreter（JavaScript解析器）：用于解析和执行 JavaScript 代码。

6、UI Backend（UI后端）：用于绘制组合框、弹窗等窗口小组件。它会调用操作系统的UI方法。

7、Data Persistence（数据存储模块）：比如数据存储  cookie、HTML5中的localStorage、sessionStorage。

参考链接：（关于浏览器的工作管理，下面这篇文章，是精品中的精品，是必须要知道的）

- 英文版：[How Browsers Work: Behind the scenes of modern web browsers](https://www.html5rocks.com/en/tutorials/internals/howbrowserswork/)
