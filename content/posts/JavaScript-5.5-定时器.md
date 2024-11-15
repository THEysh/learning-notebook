---
author: 杨盛晖
data: 2024-11-05T10:01:00+08:00
title: JavaScript-5.5-定时器
featured: true
draft: false
tags: ['javaScript']
categories: ['web']
---




<ArticleTopAd></ArticleTopAd>




## 定时器的常见方法

- setInterval()：循环调用。将一段代码，**每隔一段时间**执行一次。（循环执行）

- setTimeout()：延时调用。将一段代码，等待一段时间之后**再执行**。（只执行一次）

备注：在实际开发中，二者是可以根据需要，互相替代的。

## setInterval() 的使用

`setInterval()`：循环调用。将一段代码，**每隔一段时间**执行一次。（循环执行）

**参数**：

- 参数1：回调函数，该函数会每隔一段时间被调用一次。

- 参数2：每次调用的间隔时间，单位是毫秒。

**返回值**：返回一个Number类型的数据。这个数字用来作为定时器的**唯一标识**，方便用来清除定时器。

### 定义定时器

**方式一**：匿名函数

每间隔一秒，将 数字 加1：

```javascript
    let num = 1;
   setInterval(function () {
       num ++;
       console.log(num);
   }, 1000);
```

**方式二：**

每间隔一秒，将 数字 加1：

```javascript
    setInterval(fn,1000);

    function fn() {
       num ++;
       console.log(num);
    }

```

### 清除定时器

定时器的返回值是作为这个定时器的**唯一标识**，可以用来清除定时器。具体方法是：假设定时器setInterval()的返回值是`参数1`，那么`clearInterval(参数1)`就可以清除定时器。

setTimeout()的道理是一样的。

代码举例：

```html
<script>
    let num = 1;

    const timer = setInterval(function () {
        console.log(num);  //每间隔一秒，打印一次num的值
        num ++;
        if(num === 5) {  //打印四次之后，就清除定时器
            clearInterval(timer);
        }

    }, 1000);
</script>

```

## setTimeout() 的使用

`setTimeout()`：延时调用。将一段代码，等待一段时间之后**再执行**。（只执行一次）

**参数**：

- 参数1：回调函数，该函数会每隔一段时间被调用一次。

- 参数2：每次调用的间隔时间，单位是毫秒。

**返回值**：返回一个Number类型的数据。这个数字用来作为定时器的**唯一标识**，方便用来清除定时器。

### 定义和清除定时器

代码举例：

```javascript
    const timer = setTimeout(function() {
        console.log(1); // 3秒之后，再执行这段代码。
    }, 3000);

    clearTimeout(timer);

```

代码举例：（箭头函数写法）

```javascript
    setTimeout(() => {
        console.log(1); // 3秒之后，再执行这段代码。
    }, 3000);
```


### setTimeout() 举例：5秒后关闭网页两侧的广告栏

假设网页两侧的广告栏为两个img标签，它们的样式为：


```html
<style>
    ...
    ...

</style>

```

5秒后关闭广告栏的js代码为：

```html
    <script>
        window.onload = function () {
            //获取相关元素
            var imgArr = document.getElementsByTagName("img");
            //设置定时器：5秒后关闭两侧的广告栏
            setTimeout(fn,5000);
            function fn(){
                imgArr[0].style.display = "none";
                imgArr[1].style.display = "none";
            }
        }
    </script>
```


