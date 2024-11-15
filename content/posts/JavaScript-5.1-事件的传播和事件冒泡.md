---
author: 杨盛晖
data: 2024-11-05T09:56:00+08:00
title: JavaScript-5.1-事件的传播和事件冒泡
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>




## DOM事件流

事件传播的三个阶段是：事件捕获、事件冒泡和目标。

- 事件捕获阶段：事件从祖先元素往子元素查找（DOM树结构），直到捕获到事件目标 target。在这个过程中，默认情况下，事件相应的监听函数是不会被触发的。

- 事件目标：当到达目标元素之后，执行目标元素该事件相应的处理函数。如果没有绑定监听函数，那就不执行。

- 事件冒泡阶段：事件从事件目标 target 开始，从子元素往冒泡祖先元素冒泡，直到页面的最上一级标签。

如下图所示：

![](http://img.smyhvae.com/20180204_1218.jpg)

PS：这个概念类似于 Android 里的 **touch 事件传递**。

### 事件捕获

addEventListener可以捕获事件：

```javascript
    box1.addEventListener("click", function () {
        alert("捕获 box3");
    }, true);
```

上面的方法中，参数为true，代表事件在捕获阶段执行。

代码演示：

```javascript
    //参数为true，代表事件在「捕获」阶段触发；参数为false或者不写参数，代表事件在「冒泡」阶段触发
    box3.addEventListener("click", function () {
        alert("捕获 child");
    }, true);

    box2.addEventListener("click", function () {
        alert("捕获 father");
    }, true);

    box1.addEventListener("click", function () {
        alert("捕获 grandfather");
    }, true);

    document.addEventListener("click", function () {
        alert("捕获 body");
    }, true);
```

效果演示：

![](http://img.smyhvae.com/20180204_1101.gif)

（如果上面的图片打不开，请点击：<http://img.smyhvae.com/20180204_1101.gif>）

**重点**：捕获阶段，事件依次传递的顺序是：window --> document --> html--> body --> 父元素、子元素、目标元素。

这几个元素在事件捕获阶段的完整写法是：

```javascript
    window.addEventListener("click", function () {
        alert("捕获 window");
    }, true);

    document.addEventListener("click", function () {
        alert("捕获 document");
    }, true);

    document.documentElement.addEventListener("click", function () {
        alert("捕获 html");
    }, true);

    document.body.addEventListener("click", function () {
        alert("捕获 body");
    }, true);

    fatherBox.addEventListener("click", function () {
        alert("捕获 father");
    }, true);

    childBox.addEventListener("click", function () {
        alert("捕获 child");
    }, true);

```

说明：

（1）第一个接收到事件的对象是 **window**（有人会说body，有人会说html，这都是错误的）。

（2）JS中涉及到DOM对象时，有两个对象最常用：window、doucument。它们俩是最先获取到事件的。

**补充一个知识点：**

在 js中：

- 如果想获取 `html`节点，方法是`document.documentElement`。

- 如果想获取 `body` 节点，方法是：`document.body`。

二者不要混淆了。

### 事件冒泡

**事件冒泡**: 当一个元素上的事件被触发的时候（比如说鼠标点击了一个按钮），同样的事件将会在那个元素的所有**祖先元素**中被触发。这一过程被称为事件冒泡；这个事件从原始元素开始一直冒泡到DOM树的最上层。

通俗来讲，冒泡指的是：**子元素的事件被触发时，父元素的同样的事件也会被触发**。取消冒泡就是取消这种机制。

代码演示：

```javascript
    //事件冒泡
    box3.onclick = function () {
        alert("child");
    }

    box2.onclick = function () {
        alert("father");
    }

    box1.onclick = function () {
        alert("grandfather");
    }

    document.onclick = function () {
        alert("body");
    }

```

![](http://img.smyhvae.com/20180204_1028.gif)

（如果上面的图片打不开，请点击：<http://img.smyhvae.com/20180204_1028.gif>）

上图显示，当我点击子元素 box3 的时候，它的父元素box2、box1、body都依次被触发了。即使我改变代码的顺序，也不会影响效果的顺序。

当然，上面的代码中，我们用 addEventListener 这种 DOM2的写法也是可以的，但是第三个参数要写 false，或者不写。

**冒泡顺序**：

一般的浏览器: （除IE6.0之外的浏览器）

- div -> body -> html -> document -> window

IE6.0：

- div -> body -> html -> document

### 不是所有的事件都能冒泡

以下事件不冒泡：blur、focus、load、unload、onmouseenter、onmouseleave。意思是，事件不会往父元素那里传递。

我们检查一个元素是否会冒泡，可以通过事件的以下参数：

```javascript
    event.bubbles
```

如果返回值为true，说明该事件会冒泡；反之则相反。

举例：

```javascript
    box1.onclick = function (event) {
        alert("冒泡 child");

        event = event || window.event;
        console.log(event.bubbles); //打印结果：true。说明 onclick 事件是可以冒泡的
    }
```

## 阻止冒泡

大部分情况下，冒泡都是有益的。当然，如果你想阻止冒泡，也是可以的。可以按下面的方法阻止冒泡。

### 阻止冒泡的方法

w3c的方法：（火狐、谷歌、IE11）

```javascript
    event.stopPropagation();
```

IE10以下则是：

```javascript
event.cancelBubble = true
```

兼容代码如下：

```javascript
   box3.onclick = function (event) {

        alert("child");

        //阻止冒泡
        event = event || window.event;

        if (event && event.stopPropagation) {
            event.stopPropagation();
        } else {
            event.cancelBubble = true;
        }
    }
```

上方代码中，我们对box3进行了阻止冒泡，产生的效果是：事件不会继续传递到 father、grandfather、body了。

### 阻止冒泡的举例

```html
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8" />
        <title></title>
        <style type="text/css">
            #box1 {
                width: 100px;
                height: 100px;
                background-color: red;
                /*
        * 开启box1的绝对定位
        */
                position: absolute;
            }
        </style>

        <script type="text/javascript">
            window.onload = function() {
                /*
                 * 使div可以跟随鼠标移动
                 */

                //获取box1
                var box1 = document.getElementById('box1');

                //给整个页面绑定：鼠标移动事件
                document.onmousemove = function(event) {
                    //兼容的方式获取event对象
                    event = event || window.event;

                    // 鼠标在页面的位置 = 滚动条滚动的距离 + 可视区域的坐标。
                    var pagex = event.pageX || scroll().left + event.clientX;
                    var pagey = event.pageY || scroll().top + event.clientY;

                    //   设置div的偏移量（相对于整个页面）
                    // 注意，如果想通过 style.left 来设置属性，一定要给 box1 开启绝对定位。
                    box1.style.left = pagex + 'px';
                    box1.style.top = pagey + 'px';
                };

                // 【重要注释】
                // 当 document.onmousemove 和 box2.onmousemove 同时触发时，通过  box2 阻止事件向 document 冒泡。
                // 也就是说，只要是在 box2 的区域，就只触发 document.onmousemove 事件
                var box2 = document.getElementById('box2');
                box2.onmousemove = function(event) {
                    //阻止冒泡
                    event = event || window.event;

                    if (event && event.stopPropagation) {
                        event.stopPropagation();
                    } else {
                        event.cancelBubble = true;
                    }
                };
            };

            // scroll 函数封装
            function scroll() {
                return {
                    //此函数的返回值是对象
                    left: window.pageYOffset || document.body.scrollTop || document.documentElement.scrollTop,
                    right: window.pageXOffset || document.body.scrollLeft || document.documentElement.scrollLeft,
                };
            }
        </script>
    </head>
    <body style="height: 1000px;width: 2000px;">
        <div id="box2" style="width: 300px; height: 300px; background-color: #bfa;"></div>
        <div id="box1"></div>
    </body>
</html>
```

关键地方可以看代码中的注释。

效果演示：

![](http://img.smyhvae.com/20191112_1650.gif)


