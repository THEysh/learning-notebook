---
author: 杨盛晖
data: 2024-11-012T20:24:00+08:00
title: 在GitHub中添加ssh认证
featured: true
draft: false
tags: ['github']
categories: ['other']
---

## 第一步

**在下面路径创建一个.ssh的文件夹，一开始是空目录**
![](https://pic.imgdb.cn/item/67317853d29ded1a8c13f963.png)

## 第二步
**使用命令行创建ssh key**
```bath
ssh-keygen
```
使用命令生成密钥，第一次是确认指纹（表示是不是你自己的操作，选择yes），后续表示让你输入密码，就是说这个提示要求你输入一个 passphrase，用于保护你的私钥文件。如果你不希望使用 passphrase，可以直接按回车键，留空即可。我输入的是回车
![](https://pic.imgdb.cn/item/673178d2d29ded1a8c1483d7.png)

## 第三步
这个时候当前文件夹会有2个文件，打开.pub结尾的文件。
![](https://pic.imgdb.cn/item/673179d9d29ded1a8c16031c.png)

下面打开GitHub登录
![](https://pic.imgdb.cn/item/67317a44d29ded1a8c1672fc.png)
![](https://pic.imgdb.cn/item/67317aafd29ded1a8c16e7c3.png)

**复制里面.pub结尾的文件所有内容到github，添加Key**，标题自己描述一下，最后选择有表示意义的，不如电脑太多，ssh密钥容易记乱。

## 第四步
使用命令激活ssh的key
```bath
ssh -T git@github.com
```
![](https://pic.imgdb.cn/item/67317b10d29ded1a8c175282.png)
最后测试：
![](https://pic.imgdb.cn/item/67317b72d29ded1a8c17c4df.png)

## 第五步

拷贝ssh链接

![](https://pic.imgdb.cn/item/67317c9ad29ded1a8c18e4cc.png)

打开git bath 或者使用vscode使用命令克隆

```bath
git clone git@github.com:THEysh/notebook.git
```

这样就把这个项目拷贝下来了

![](https://pic.imgdb.cn/item/67317d6dd29ded1a8c19bde8.png)
