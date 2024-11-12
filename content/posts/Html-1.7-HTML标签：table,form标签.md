---
author: 杨盛晖
data: 2024-11-06T09:23:00+08:00
title: Html-1.7-HTML标签：table,form标签
featured: true
draft: false
tags: ['html']
categories: ['web']
---



<ArticleTopAd></ArticleTopAd>


## 本文主要内容

 - 列表标签：`<ul>`、`<ol>`、`<dl>`
 - 表格标签：`<table>`
 - 框架标签及内嵌框架`<iframe>`
 - 表单标签：`<form>`
 - 多媒体标签



## 列表标签

列表标签分为三种。

### 1、无序列表`<ul>`，无序列表中的每一项是`<li>`

英文单词解释如下：

- ul：unordered list，“无序列表”的意思。
- li：list item，“列表项”的意思。

例如：

```html
<ul>
	<li>默认1</li>
	<li>默认2</li>
	<li>默认3</li>
</ul>
```

注意：

- li不能单独存在，必须包裹在ul里面；反过来说，ul的“儿子”不能是别的东西，只能有li。
- 我们这里再次强调，ul的作用，并不是给文字增加小圆点的，而是增加无序列表的“语义”的。


**属性：**

 - `type="属性值"`。属性值可以选： `disc`(实心原点，默认)，`square`(实心方点)，`circle`(空心圆)。


  - 不光是`<ul>`标签有`type`属性，`<ul>`里面的`<li>`标签也有`type`属性（虽然说这种写法很少见）。


当然了，列表之间是可以**嵌套**的。我们来举个例子。代码：

```html
  <ul>
	<li><b>北京市</b>
		<ul>
			<li>海淀区</li>
			<li>朝阳区</li>
			<li>东城区</li>

		</ul>
	</li>

	<li><b>广州市</b>
		<ul>
			<li>天河区</li>
			<li>越秀区</li>
		</ul>
	</li>
  </ul>
```

**li里面什么都能放，甚至可以再放一个ul**。

### 2、有序列表`<ol>`，里面的每一项是`<li>`

英文单词：Ordered List。

例如：

```html
<ol >
	<li>呵呵哒1</li>
	<li>呵呵哒2</li>
	<li>呵呵哒3</li>
</ol>
```

**属性：**
 - `type="属性值"`。属性值可以是：1(阿拉伯数字，默认)、a、A、i、I。结合`start`属性表示`从几开始`。

举例：

```html
<ol type="1">
	<li>呵呵</li>
	<li>呵呵</li>
	<li>呵呵</li>
</ol>

<ol type="a">
	<li>嘿嘿</li>
	<li>嘿嘿</li>
	<li>呵呵</li>
</ol>

<ol type="i" start="4">
	<li>哈哈</li>
	<li>哈哈</li>
	<li>哈哈</li>
</ol>

<ol type="I" start="10">
	<li>么么</li>
	<li>么么</li>
	<li>么么</li>
</ol>
```

和无序列表一样，有序列表也是可以嵌套的哦，这里就不举类似的例子了。


ol和ul就是语义不一样，怎么使用都是一样的。
ol里面只能有li，li必须被ol包裹。li是容器级。

ol这个东西用的不多，如果想表达顺序，大家一般也用ul。举例如下：

```html
<ul>
	<li>1. 小苹果</li>
	<li>2. 月亮之上</li>
	<li>3. 最炫民族风</li>
</ul>
```

### 3、定义列表`<dl>`

> 定义列表的作用非常大。

`<dl>`英文单词：definition list，没有属性。dl的子元素只能是dt和dd。

 - `<dt>`：definition title 列表的标题，这个标签是必须的
 - `<dd>`：definition description 列表的列表项，如果不需要它，可以不加

备注：dt、dd只能在dl里面；dl里面只能有dt、dd。

举例：

```html
<dl>
	《中华人民共和国合同法》第一条到第三条内容如下：
	<dt>第一条</dt>
	<dd>为了保护当事人的合法权益，维护社会经济秩序，促进社会主义市场经济的发展，制定本法。</dd>
	<dt>第二条</dt>
	<dd>本法适用于在中华人民共和国境内订立和履行的合同</dd>
	<dt>第三条</dt>
	<dd>当事人平等自愿地订立合同，依法享有选择合同相对人、订立合同的权利，独立自主地决定合同的内容</dd>
</dl>
```
- （1）是一个列表，列出了dd项目
- （2）每一个词儿都有自己的描述项。

备注：dd是描述dt的。


定义列表用法非常灵活，可以一个dt配很多dd：

```html
	<dl>
		<dt>北京</dt>
		<dd>国家首都，政治文化中心</dd>
		<dd>污染很严重，PM2.0天天报表</dd>
		<dt>上海</dt>
		<dd>魔都，有外滩、东方明珠塔、黄浦江</dd>
		<dt>广州</dt>
		<dd>中国南大门，有珠江、小蛮腰</dd>
	</dl>
```

还可以拆开，让每一个dl里面只有一个dt和dd，这样子感觉清晰一些：

```html
	<dl>
		<dt>北京</dt>
		<dd>国家首都，政治文化中心</dd>
		<dd>污染很严重，PM2.0天天报表</dd>
	</dl>

	<dl>
		<dt>上海</dt>
		<dd>魔都，有外滩、东方明珠塔、黄浦江</dd>
	</dl>

	<dl>
		<dt>广州</dt>
		<dd>中国南大门，有珠江、小蛮腰</dd>
	</dl>
```

```html
<dl>
	<dt>购物指南</dt>
	<dd>
		<a href="#">购物流程</a>
		<a href="#">会员介绍</a>
		<a href="#">生活旅行/团购</a>
		<a href="#">常见问题</a>
		<a href="#">大家电</a>
		<a href="#">联系客服</a>
	</dd>
</dl>
<dl>
	<dt>配送方式</dt>
	<dd>
		<a href="#">上门自提</a>
		<a href="#">211限时达</a>
		<a href="#">配送服务查询</a>
		<a href="#">配送费收取标准</a>
		<a href="#">海外配送</a>
	</dd>
</dl>

```

dt、dd都是容器级标签，想放什么都可以。所以，现在就应该更加清晰的知道：用什么标签，不是根据样子来决定，而是语义（语义本质上是结构）。

## 表格标签

表格标签用`<table>`表示。
一个表格`<table>`是由每行`<tr>`组成的，每行是由每个单元格`<td>`组成的。
所以我们要记住，一个表格是由行组成的（行是由列组成的），而不是由行和列组成的。
在以前，要想固定标签的位置，唯一的方法就是表格。现在可以通过CSS定位的功能来实现。但是现在在做页面的时候，表格作用还是有一些的。

例如，一行的单元格：
```html
<table>
	<tr>
		<td></td>
		<td></td>
		<td></td>
		<td></td>
	</tr>
</table>
```
上面的表格中没有加文字，所以在生成的网页中什么都看不到。
例如，3行4列的单元格：
```html
<table>
	<tr>
		<td>xxx</td>
		<td>23</td>
		<td>男</td>
		<td>黄冈</td>
	</tr>

	<tr>
		<td>许嵩</td>
		<td>29</td>
		<td>男</td>
		<td>安徽</td>
	</tr>

	<tr>
		<td>邓紫棋</td>
		<td>23</td>
		<td>女</td>
		<td>香港</td>
	</tr>

</table>
```


上图中的表格好像没看到边框呀，不急，接下来看看`<table>`标签的属性。

**`<table>`的属性：**
 - `border`：边框。像素为单位。
 - `style="border-collapse:collapse;"`：单元格的线和表格的边框线合并（表格的两边框合并为一条）
 - `width`：宽度。像素为单位。
 - `height`：高度。像素为单位。
 - `bordercolor`：表格的边框颜色。
 - `align`：**表格**的水平对齐方式。属性值可以填：left right center。
注意：这里不是设置表格里内容的对齐方式，如果想设置内容的对齐方式，要对单元格标签`<td>`进行设置）
 - `cellpadding`：单元格内容到边的距离，像素为单位。默认情况下，文字是紧挨着左边那条线的，即默认情况下的值为0。
注意不是单元格内容到四条边的距离哈，而是到一条边的距离，默认是与左边那条线的距离。如果设置属性`dir="rtl"`，那就指的是内容到右边那条线的距离。
 - `cellspacing`：单元格和单元格之间的距离（外边距），像素为单位。默认情况下的值为0
 - `bgcolor="#99cc66"`：表格的背景颜色。
 - `background="路径src/..."`：背景图片。
背景图片的优先级大于背景颜色。
 - `bordercolorlight`：表格的上、左边框，以及单元格的右、下边框的颜色
 - `bordercolordark`：表格的右、下边框，以及单元格的上、左的边框的颜色
这两个属性的目的是为了设置3D的效果。
 - `dir`：公有属性，单元格内容的排列方式(direction)。 可以 取值：`ltr`：从左到右（left to right，默认），`rtl`：从右到左（right to left）
既然说`dir`是共有属性，如果把这个属性放在任意标签中，那表明这个标签的位置可能会从右开始排列。

备注：表格中很细表格边线的制作，CSS的写法：


### `<tr>`：行

一个表格就是一行一行组成的。

**属性：**

 - `dir`：公有属性，设置这一行单元格内容的排列方式。可以取值：
 	- `ltr`：从左到右（left to right，默认）
	- `rtl`：从右到左（right to left）
 - `bgcolor`：设置这一行的单元格的背景色。
注：没有background属性，即：无法设置这一行的背景图片，如果非要设置，可以用css实现。
 - `height`：一行的高度
 - `align="center"`：一行的内容水平居中显示，取值：left、center、right
 - `valign="center"`：一行的内容垂直居中，取值：top、middle、bottom

### `<td>`：单元格

**属性：**

 - `align`：内容的横向对齐方式。属性值可以填：left right center。如果想让每个单元格的内容都居中，这个属性太麻烦了，以后用css来解决。
 - `valign`：内容的纵向对齐方式。属性值可以填：top middle bottom
 - `width`：绝对值或者相对值(%)
 - `height`：单元格的高度
 - `bgcolor`：设置这个单元格的背景色。
 - `background`：设置这个单元格的背景图片。

### 单元格的合并

单元格的属性：

- `colspan`：横向合并。例如`colspan="2"`表示当前单元格在水平方向上要占据两个单元格的位置。
- `rowspan`：纵向合并。例如`rowspan="2"`表示当前单元格在垂直方向上要占据两个单元格的位置。
```html

    <table border="1" align="center">
        <tr >
            <td colspan="2" rowspan="2">xxx</td>
            <td>男</td>
            <td>黄冈</td>
        </tr>
        <tr>
            <td>男</td>
            <td>安徽</td>
        </tr>
        <tr>
            <td>23</td>
            <td>23</td>
            <td>女</td>
            <td>香港</td>
        </tr>
    </table>

    <table border="1" align="center">
        <tr >
            <td colspan="2" align="center">000</td>
            <td>001</td>
            <td>002</td>
        </tr>
        <tr>
            <td>000</td>
            <td>001</td>
            <td colspan="2" align="center">002</td>
        </tr>
        <tr>
            <td>000</td>
            <td>001</td>
            <td>002</td>
            <td>003</td>
        </tr>
    </table>

    <table border="1" align="center">
        <tr >
            <td colspan="2" align="center">000</td>
            <td>001</td>
            <td>002</td>
        </tr>
        <tr>
            <td>000</td>
            <td>001</td>
            <td rowspan="2" align="center">002</td>
            <td>003</td>
        </tr>
        <tr>
            <td>000</td>
            <td>001</td>
            <td>002</td>
        </tr>
    </table>
```

### `<th>`：加粗的单元格。相当于`<td>` + `<b>`

- 属性同`<td>`标签。



### `<caption>`：表格的标题。使用时和`tr`标签并列

 - 属性：`align`，表示标题相对于表格的位置。属性取值可以是：left、center、right、top、bottom
效果：

### 表格的`<thead>`标签、`<tbody>`标签、`<tfoot>`标签

这三个标签有与没有的区别：

- 1、如果写了，那么这三个部分的**代码顺序可以任意**，浏览器显示的时候还是按照thead、tbody、tfoot的顺序依次来显示内容。如果不写thead、tbody、tfoot，那么浏览器解析并显示表格内容的时候是从按照代码的从上到下的顺序来显示。
- 2、当表格非常大内容非常多的时候，如果用thead、tbody、tfoot标签的话，那么**数据可以边获取边显示**。如果不写，则必须等表格的内容全部从服务器获取完成才能显示出来。

举例：

```html
 <body>

	<table border="1">

		<tbody>
		<tr>
			<td>xxx</td>
			<td>23</td>
			<td>男</td>
			<td>黄冈</td>
		</tr>
		</tbody>

		<tfoot>
		<tr>
			<td>许嵩</td>
			<td>29</td>
			<td>男</td>
			<td>安徽</td>
		</tr>
		</tfoot>

		<thead>
		<tr>
			<td>邓紫棋</td>
			<td>23</td>
			<td>女</td>
			<td>香港</td>
		</tr>
		</thead>

	</table>

 </body>
```

## 框架标签

## `<iframe>`

（内联框架）标签用于在一个网页中嵌入另一个 HTML 页面。它允许开发者在主页面上显示其他网站或相同网站的内容，而不需要离开当前页面。

## 主要特点

1. **嵌入其他网页**：可以加载其他网页的内容，比如文档、视频、地图等。
2. **独立性**：嵌入的内容与主页面相对独立，因此可以在其中使用独立的 JavaScript 和 CSS。
3. **尺寸控制**：可以通过设置 `width` 和 `height` 属性来控制 `<iframe>` 的大小。

## 常用属性

- **`src`**：指定要嵌入的页面 URL。
- **`width`**：设置 `<iframe>` 的宽度，可以使用像素或百分比。
- **`height`**：设置 `<iframe>` 的高度，可以使用像素或百分比。
- **`frameborder`**：设置边框的显示（在 HTML5 中不推荐使用）。
- **`allowfullscreen`**：允许 `<iframe>` 全屏显示。
- **`title`**：为 `<iframe>` 提供描述性的标题，增强可访问性。

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iframe 示例</title>
    <style>
        body {
            display: flex;
            height: 100vh;
            margin: 0;
        }
    </style>
</head>
<body >
    <iframe src="https://www.taobao.com" width="50%" height="100%" title="左侧内容"></iframe>
    <iframe src="标签.html" width="50%" height="100%" title="右侧内容"></iframe>

</body>
</html>
```
## 表单标签

表单标签用`<form>`表示，用于与服务器的交互。表单就是收集用户信息的，就是让用户填写的、选择的。

**属性：**
 - `name`：表单的名称，用于JS来操作或控制表单时使用；
 - `id`：表单的名称，用于JS来操作或控制表单时使用；
 - `action`：指定表单数据的处理程序，一般是PHP，如：action=“login.php”
 - `method`：表单数据的提交方式，一般取值：get(默认)和post

注意：表单和表格嵌套时，是在`<form>`标记中套`<table>`标记。

form标签里面的action属性和method属性，在后续的 ajax文章上再讲。这里简单说一下：action属性就是表示，表单将提交到哪里。 method属性表示用什么HTTP方法提交，有get、post两种。

**get提交和post提交的区别：**

GET方式：
将表单数据，以"name=value"形式追加到action指定的处理程序的后面，两者间用"?"隔开，每一个表单的"name=value"间用"&"号隔开。
特点：只适合提交少量信息，并且不太安全(不要提交敏感数据)、提交的数据类型只限于ASCII字符。

POST方式：
将表单数据直接发送(隐藏)到action指定的处理程序。POST发送的数据不可见。Action指定的处理程序可以获取到表单数据。
特点：可以提交海量信息，相对来说安全一些，提交的数据格式是多样的(Word、Excel、rar、img)。

**Enctype：**
表单数据的编码方式(加密方式)，取值可以是：application/x-www-form-urlencoded、multipart/form-data。Enctype只能在POST方式下使用。

- Application/x-www-form-urlencoded：**默认**加密方式，除了上传文件之外的数据都可以
- Multipart/form-data：**上传附件时，必须使用这种编码方式**。


### `<input>`：输入标签（文本框）

用于接收用户输入。

```html
<input type="text" />
```

**属性：**

- **`type="属性值"`**：文本类型。属性值可以是：
	- `text`（默认）
	- `password`：密码类型
	- `radio`：单选按钮，名字相同的按钮作为一组进行单选（单选按钮，天生是不能互斥的，如果想互斥，必须要有相同的name属性。name就是“名字”。
	）。非常像以前的收音机，按下去一个按钮，其他的就抬起来了。所以叫做radio。
	- `checkbox`：多选按钮，**name 属性值相同的按钮**作为一组进行选择。
	- `checked`：将单选按钮或多选按钮默认处于选中状态。当`<input>`标签设置为`type="radio"`或者`type=checkbox`时，可以用这个属性。属性值也是checked，可以省略。
	- `hidden`：隐藏框，在表单中包含不希望用户看见的信息
	- `button`：普通按钮，结合js代码进行使用。
	- `submit`：提交按钮，传送当前表单的数据给服务器或其他程序处理。这个按钮不需要写value自动就会有“提交”文字。这个按钮真的有提交功能。点击按钮后，这个表单就会被提交到form标签的action属性中指定的那个页面中去。
	- `reset`：重置按钮，清空当前表单的内容，并设置为最初的默认值
	- `image`：图片按钮，和提交按钮的功能完全一致，只不过图片按钮可以显示图片。
	- `file`：文件选择框。
	提示：如果要限制上传文件的类型，需要配合JS来实现验证。对上传文件的安全检查：一是扩展名的检查，二是文件数据内容的检查。

 - **`value="内容"`**：文本框里的默认内容（已经被填好了的）

 - `size="50"`：表示文本框内可以显示**五十个字符**。一个英文或一个中文都算一个字符。
注意**size属性值的单位不是像素哦**。

 - `readonly`：文本框只读，不能编辑。因为它的属性值也是readonly，所以属性值可以不写。
用了这个属性之后，在google浏览器中，光标点不进去；在IE浏览器中，光标可以点进去，但是文字不能编辑。

 - `disabled`：文本框只读，不能编辑，光标点不进去。属性值可以不写。

> 备注：HTML5中，input的类型又增加了很多（比如date、color，我们会在 html5 中讲到）。

**举例**：

```html
	<form>
		姓名：<input value="呵呵" >显示信息<br>
		昵称：<input value="哈哈" readonly=""><br>
		名字：<input type="text" value="name" disabled=""><br>
		密码：<input type="password" value="pwd" size="50"><br>
		性别：<input type="radio" name="gender" id="radio1" value="male" checked="">男
			  <input type="radio" name="gender" id="radio2" value="female" >女<br>
		爱好：<input type="checkbox" name="love" value="eat">吃饭
			  <input type="checkbox" name="love" value="sleep">睡觉
			  <input type="checkbox" name="love" value="bat">打豆豆
	</form>
```


注意，多个单选框的input标签中，name 的属性值可以相同，但是 **id 的属性值必须是唯一的**。我们知道，html的标签中，id的属性值是唯一的。

**四种按钮的举例**：

```html
	<form>
		<input type="button" value="普通按钮"><br>
		<input type="submit"  value="提交按钮"><br>
		<input type="reset" value="重置按钮"><br>
		<input type="image" src="1.jpg" width="800" value="图片按钮2"><br>
		<input type="file" value="文件选择框">
	</form>
```

**前端开发工程师，重点关心页面的美、样式、板式、交互。至于数据的提供和比较重的业务逻辑，都是后台工程师做的事情。**


### `<select>`：下拉列表标签

`<select>`标签里面的每一项用`<option>`表示。select就是“选择”，option“选项”。

select标签和ul、ol、dl一样，都是组标签。

**`<select>`标签的属性：**

- `multiple`：可以对下拉列表中的选项进行多选。属性值为 multiple，也可以没有属性值。也就是说，既可以写成 `multiple=""`，也可以写成`multiple="multiple"`。
- `size="3"`：如果属性值大于1，则列表为滚动视图。默认属性值为1，即下拉视图。

**`<option>`标签的属性：**

 - `selected`：预选中。没有属性值。

举例：

```html
    <form>
		<select>
			<option>小学</option>
			<option>初中</option>
			<option>高中</option>
			<option>大学</option>
			<option selected="">研究生</option>
		</select>
	</form>
```

### `<label>`标签

我们先来看下面一段代码：

```html
<input type="radio" name="sex" /> 男
<input type="radio" name="sex" /> 女

```

对于上面这样的单选框，我们只有点击那个单选框（小圆圈）才可以选中，点击“男”、“女”这两个文字时是无法选中的；于是，label标签派上了用场。

本质上来讲，“男”、“女”这两个文字和input标签时没有关系的，而label就是解决这个问题的。我们可以通过label把input和汉字包裹起来作为整体。

解决方法如下：

```html
<input type="radio" name="sex" id="nan" /> <label for="nan">男</label>
<input type="radio" name="sex" id="nv"  /> <label for="nv">女</label>
```

上方代码中，让label标签的**for 属性值**，和 input 标签的 **id 属性值相同**，那么这个label和input就有绑定关系了。


当然了，复选框也有label：（任何表单元素都有label）

```html
<input type="checkbox" id="kk" />
<label for="kk">10天内免登陆</label>
```


## 多媒体

在HTML5之前，在网页上播放音频/视频的通用方法是利用Flash来播放。但是大多情况下，并非所有用户的浏览器都安装了Flash插件，由此使得音频、视频播放的处理变得非常复杂；并且移动设备的浏览器并不支持Flash插件。

H5里面提供了视频和音频的标签。

### 音频

HTML5通过`<audio>`标签来解决音频播放的问题。

使用举例：

```html
	<audio src="music/yinyue.mp3" autoplay controls> </audio>
```

效果如下：

![](http://img.smyhvae.com/20180206_1958.png)

我们可以通过附加属性，来更友好地控制音频的播放，如：

- `autoplay` 自动播放。写成`autoplay` 或者 `autoplay = ""`，都可以。

- `controls` 控制条。（建议把这个选项写上，不然都看不到控件在哪里）

- `loop` 循环播放。

- `preload` 预加载 同时设置 autoplay 时，此属性将失效。

**处理兼容性问题：**

由于版权等原因，不同的浏览器可支持播放的格式是不一样的：

![](http://img.smyhvae.com/20180206_1945.png)

为了做到多浏览器支持，可以采取以下兼容性写法：

```html
<!--推荐的兼容写法：-->
<audio controls loop>
    <source src="music/yinyue.mp3"/>
    <source src="music/yinyue.ogg"/>
    <source src="music/yinyue.wav"/>
    抱歉，你的浏览器暂不支持此音频格式
</audio>
```

代码解释：如果识别不出音频格式，就弹出那句“抱歉”。


### 视频


HTML5通过`<video>`标签来解决视频播放的问题。

使用举例：

```html
	<video src="video/movie.mp4" controls autoplay></video>
```


我们可以通过附加属性，来更友好地控制视频的播放，如：

- `autoplay` 自动播放。写成`autoplay` 或者 `autoplay = ""`，都可以。

- `controls` 控制条。（建议把这个选项写上，不然都看不到控件在哪里）

- `loop` 循环播放。

- `preload` 预加载 同时设置 autoplay 时，此属性将失效。

- `width`：设置播放窗口宽度。

- `height`：设置播放窗口的高度。

由于版权等原因，不同的浏览器可支持播放的格式是不一样的：

![](http://img.smyhvae.com/20180206_2025.png)

兼容性写法：

```html
    <!--<video src="video/movie.mp4" controls  autoplay ></video>-->
    <!--  行内块 display:inline-block -->
    <video controls autoplay>
        <source src="video/movie.mp4"/>
        <source src="video/movie.ogg"/>
        <source src="video/movie.webm"/>
        抱歉，不支持此视频
    </video>
```
