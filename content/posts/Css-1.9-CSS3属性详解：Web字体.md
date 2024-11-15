---
author: 杨盛晖
data: 2024-11-07T09:25:00+08:00
title: Css-1.9-CSS3属性详解：Web字体
featured: true
draft: false
tags: ['css']
categories: ['web']
---



<ArticleTopAd></ArticleTopAd>



## 前言


开发人员可以为自已的网页指定特殊的字体（将指定字体提前下载到站点中），无需考虑用户电脑上是否安装了此特殊字体。从此，把特殊字体处理成图片的方式便成为了过去。

支持程度比较好，甚至 IE 低版本的浏览器也能支持。

## 字体的常见格式

> 不同浏览器所支持的字体格式是不一样的，我们有必要了解一下字体格式的知识。

#### TureTpe格式：(**.ttf**)

.ttf 字体是Windows和Mac的最常见的字体，是一种RAW格式。

支持这种字体的浏览器有IE9+、Firefox3.5+、Chrome4+、Safari3+、Opera10+、iOS Mobile、Safari4.2+。



#### OpenType格式：(**.otf**)

.otf 字体被认为是一种原始的字体格式，其内置在TureType的基础上。

支持这种字体的浏览器有Firefox3.5+、Chrome4.0+、Safari3.1+、Opera10.0+、iOS Mobile、Safari4.2+。


#### Web Open Font Format格式：(**.woff**)

woff字体是Web字体中最佳格式，他是一个开放的TrueType/OpenType的压缩版本，同时也支持元数据包的分离。

支持这种字体的浏览器有IE9+、Firefox3.5+、Chrome6+、Safari3.6+、Opera11.1+。

#### Embedded Open Type格式：(**.eot**)

.eot字体是IE专用字体，可以从TrueType创建此格式字体，支持这种字体的浏览器有IE4+。


#### SVG格式：(**.svg**)

.svg字体是基于SVG字体渲染的一种格式。

支持这种字体的浏览器有Chrome4+、Safari3.1+、Opera10.0+、iOS Mobile Safari3.2+。

**总结：**

了解了上面的知识后，**我们就需要为不同的浏览器准备不同格式的字体**。通常我们会通过字体生成工具帮我们生成各种格式的字体，因此无需过于在意字体格式之间的区别。


下载字体的网站推荐：

- <http://www.zhaozi.cn/>

- <http://www.youziku.com/>


## WebFont 的使用步骤

打开网站<http://iconfont.cn/webfont#!/webfont/index>，如下：

![](http://img.smyhvae.com/20180220_1328.png)

上图中，比如我想要「思源黑体-粗」这个字体，那我就点击红框中的「本地下载」。

下载完成后是一个压缩包，压缩包链接：http://download.csdn.net/download/smyhvae/10253561

解压后如下：

![](http://img.smyhvae.com/20180220_1336.png)

上图中， 我们把箭头处的html文件打开，里面告诉了我们 webfont 的**使用步骤**：

![](http://img.smyhvae.com/20180220_1338.png)

（1）第一步：使用font-face声明字体

```css
@font-face {font-family: 'webfont';
    src: url('webfont.eot'); /* IE9*/
    src: url('webfont.eot?#iefix') format('embedded-opentype'), /* IE6-IE8 */
    url('webfont.woff') format('woff'), /* chrome、firefox */
    url('webfont.ttf') format('truetype'), /* chrome、firefox、opera、Safari, Android, iOS 4.2+*/
    url('webfont.svg#webfont') format('svg'); /* iOS 4.1- */
}
```


（2）第二步：定义使用webfont的样式

```css
.web-font{
    font-family:"webfont" !important;
    font-size:16px;font-style:normal;
    -webkit-font-smoothing: antialiased;
    -webkit-text-stroke-width: 0.2px;
    -moz-osx-font-smoothing: grayscale;}
```


（3）第三步：为文字加上对应的样式

```html
<i class="web-font">这一分钟，你和我在一起，因为你，我会记得那一分钟。从现在开始，我们就是一分钟的朋友。这是事实，你改变不了，因为已经完成了。</i>
```

**举例：**

我们按照上图中的步骤来，引入这个字体。完整版代码如下：

```html
<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">
    <title></title>
    <style>

        p{
            font-size:30px;
        }

        /*  如果要在网页中使用web字体（用户电脑上没有这种字体）*/
        /* 第一步：声明字体*/
        /* 告诉浏览器 去哪找这个字体*/
        @font-face {font-family: 'my-web-font';
            src: url('font/webfont.eot'); /* IE9*/
            src: url('font/webfont.eot?#iefix') format('embedded-opentype'), /* IE6-IE8 */
            url('font/webfont.woff') format('woff'), /* chrome、firefox */
            url('font/webfont.ttf') format('truetype'), /* chrome、firefox、opera、Safari, Android, iOS 4.2+*/
            url('font/webfont.svg#webfont') format('svg'); /* iOS 4.1- */
        }
        /* 第二步：定义一个类名，谁加这类名，就会使用 webfont 字体*/
        .webfont{
            font-family: 'my-web-font';
        }
    </style>
</head>
<body>
    <!-- 第三步：引用 webfont 字体 -->
    <p class="webfont">永不止步</p>
</body>
</html>
```


代码解释：

（1）`my-web-font`这个名字是随便起的，只要保证第一步和第二步中的名字一样就行。

（2）因为我把字体文件单独放在了font文件夹中，所以在src中引用字体资源时，写的路径是 `font/...`

**举例：**
```html
<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">
    <title></title>
    <style>
        @font-face {font-family: 'my-web-font';
            src:
            url('font/albbdfdk/AlimamaDongFangDaKai-Regular.woff') , /* chrome、firefox */
            url('font/albbdfdk/AlimamaDongFangDaKai-Regular.ttf') , /* chrome、firefox、opera、Safari, Android, iOS 4.2+*/
  
        }
        /* 第二步：定义一个类名，谁加这类名，就会使用 webfont 字体*/
        .webfont{
            font-family:'my-web-font';
        }
        div{
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body>
    <!-- 第三步：引用 webfont 字体 -->
     <div>
        <h1 class="webfont">hello world, 你好世界！</h1>
     </div>
</body>
</html>
```

## 字体图标（阿里的 iconfont 网站举例）

我们其实可以把图片制作成字体。常见的做法是：把网页中一些小的图标，借助工具生成一个字体包，然后就可以像使用文字一样使用图标了。这样做的优点是：

- 将所有图标打包成字体库，减少请求；

- 具有矢量性，可保证清晰度；

- 使用灵活，便于维护。

也就是说，我们可以把这些图标当作字体来看待，凡是字体拥有的属性（字体大小、颜色等），均适用于图标。

**使用步骤如下：**（和上一段的使用步骤是一样的）

打开网站<http://iconfont.cn/>，找到想要的图标，加入购物车。然后下载下来：

![](http://img.smyhvae.com/20180220_1750.png)

压缩包下载之后，解压，打开里面的demo.html，里面告诉了我们怎样引用这些图标。
例如：
![](http://img.smyhvae.com/20180220_1755.png)

**举例1**：（图标字体引用）

```html
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Document</title>  
    <style>  
        @font-face {  
            font-family: 'iconfont';  
            src: url('icon/font_dg8dcsef1wu/iconfont.ttf')
        }  
        .iconfont {  
            font-family: 'iconfont';  
            font-size: 16px;  
            font-style: normal;  
        }  
    </style>  
</head>  
<body>  
    <!-- 假设您知道图标对应的 Unicode 码点，这里用 U+E001 作为示例 -->  
    <span class="iconfont">&#xe6eb;</span>  
</body>  
</html>
```

**举例2**：（伪元素的方式使用图标字体）

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="icon/font_dg8dcsef1wu/iconfont.css">
</head>
<body>
    <span class="iconfont icon-boluo">我是菠萝</span>
</body>
</html>
```

**示例3:**

这个JavaScript文件是一个图标字体库，用于动态地加载和渲染图标。&lt;svg&gt;标签内的&lt;use&gt;元素通过```<use xlink:href="#icon-boluo"></use>```引用了这个图标字体库中的一个具体图标（在这个例子中是#icon-boluo）

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="./icon/font_dg8dcsef1wu/iconfont.js"></script>
    <style>
        .icon {
          width: 3em;
          height: 3em;
          vertical-align: -0.15em;
          fill: currentColor;
          overflow: hidden;
        }
    </style>
</head>
<body>
    <svg class="icon" aria-hidden="true">
        <use xlink:href="#icon-boluo"></use>
    </svg>
</body>
</html>
```

## 其他相相关网站介绍

- Font Awesome 使用介绍：<http://fontawesome.dashgame.com/>

定制自已的字体图标库：

- <http://iconfont.cn/>

- <https://icomoon.io/>

SVG素材：

- <https://www.iconfont.cn/>




## 360浏览器网站案例

暂略。

这里涉及到：jQuery fullPage   全屏滚动插件。

- 中文网址:http://www.dowebok.com

- 相关说明:http://www.dowebok.com/77.html











