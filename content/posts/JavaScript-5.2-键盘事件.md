---
author: 杨盛晖
data: 2024-11-05T09:58:00+08:00
title: JavaScript-5.2-键盘事件
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>



## 鼠标的拖拽事件

拖拽的流程：

（1）`onmousedown`：当鼠标在被拖拽元素上按下时，开始拖拽；

（2）`onmousemove`：当鼠标移动时被拖拽元素跟随鼠标移动；

（3）`onmouseup`：当鼠标松开时，被拖拽元素固定在当前位置。

## 鼠标的滚轮事件

`onmousewheel`：鼠标滚轮滚动的事件，会在滚轮滚动时触发。但是火狐不支持该属性。

`DOMMouseScroll`：在火狐中需要使用 DOMMouseScroll 来绑定滚动事件。注意该事件需要通过addEventListener()函数来绑定。

## 键盘事件

### 事件名

`onkeydown`：按键被按下。

`onkeyup`：按键被松开。


**注意**：

- 如果一直按着某一个按键不松手，那么，`onkeydown`事件会一直触发。此时，松开键盘，`onkeyup`事件会执行一次。

- 当`onkeydown`连续触发时，第一次和第二次之间会间隔稍微长一点，后续的间隔会非常快。这种设计是为了防止误操作的发生。

键盘事件一般都会绑定给一些可以获取到焦点的对象或者是document。代码举例：

```html
    <body>
        <script>
            document.onkeydown = function(event) {
                event = event || window.event;
                console.log('qianguyihao 键盘按下了');
            };

            document.onkeyup = function() {
                console.log('qianguyihao 键盘松开了');
            };
        </script>

        <input type="text" />
    </body>
```


### 判断哪个键盘被按下

可以通过`event`事件对象的`keyCode`来获取按键的编码。


此外，`event`事件对象里面还提供了以下几个属性：

- altKey

- ctrlKey

- shiftKey


上面这三个属性，可以用来判断`alt`、`ctrl`、和`shift`是否被按下。如果按下则返回true，否则返回false。代码举例：

```html
    <body>
        <script>
            document.onkeydown = function(event) {
                event = event || window.event;
                console.log('qianguyihao：键盘按下了');

                // 判断y和ctrl是否同时被按下
                if (event.ctrlKey && event.keyCode === 89) {
                    console.log('ctrl和y都被按下了');
                }
            };
        </script>
    </body>
```


**举例**：input 文本框中，禁止输入数字。代码实现：


```html
    <body>
        <input type="text" />

        <script>
            //获取input
            var input = document.getElementsByTagName('input')[0];

            input.onkeydown = function(event) {
                event = event || window.event;

                //console.log('qianguyihao:' + event.keyCode);
                //数字 48 - 57
                //使文本框中不能输入数字
                if (event.keyCode >= 48 && event.keyCode <= 57) {
                    //在文本框中输入内容，属于onkeydown的默认行为
                    return false; // 如果在onkeydown中取消了默认行为，则输入的内容，不会出现在文本框中
                }
            };
        </script>
    </body>

```


## 举例：通过键盘的方向键，移动盒子

代码实现：

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
                position: absolute;
            }
        </style>
    </head>
    <body>
        <div id="box1"></div>

        <script type="text/javascript">
            // 使div可以根据不同的方向键向不同的方向移动
            /*
             * 按左键，div向左移
             * 按右键，div向右移
             * ...
             */

            //为document绑定一个按键按下的事件
            document.onkeydown = function(event) {
                event = event || window.event;

                //定义一个变量，来表示移动的速度
                var speed = 10;

                //当用户按了ctrl以后，速度加快
                if (event.ctrlKey) {
                    console.log('smyhvae ctrl');
                    speed = 20;
                }

                /*
                 * 37 左
                 * 38 上
                 * 39 右
                 * 40 下
                 */
                switch (event.keyCode) {
                    case 37:
                        //alert("向左"); left值减小
                        box1.style.left = box1.offsetLeft - speed + 'px'; // 在初始值的基础之上，减去 speed 大小
                        break;
                    case 39:
                        //alert("向右");
                        box1.style.left = box1.offsetLeft + speed + 'px';
                        break;
                    case 38:
                        //alert("向上");
                        box1.style.top = box1.offsetTop - speed + 'px';
                        break;
                    case 40:
                        //alert("向下");
                        box1.style.top = box1.offsetTop + speed + 'px';
                        break;
                }
            };
        </script>
    </body>
</html>


```

上方代码，待改进的地方：

（1）移动盒子时，如果要加速，需要先按`方向键`，再按`Ctrl键`。

（2）首次移动盒子时，动作较慢。后续如果学习了定时器相关的内容，可以再改进。


