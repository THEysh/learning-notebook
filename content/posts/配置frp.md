---
author: 杨盛晖
data: 2024-11-012T20:25:00+08:00
title: 配置frp
featured: true
draft: false
tags: ['frp']
categories: ['other']
---

## 条件
- 一个公网ip(服务器)
- 一台电脑

frp官方文档：https://gofrp.org/zh-cn/docs/setup/


## 配置服务器

- 我们使用linux centos系统。从 GitHub 的 [Release](https://github.com/fatedier/frp/releases) 页面中下载最新版本的客户端和服务器二进制文件。所有文件都打包在一个压缩包中，还包含了一份完整的配置参数说明。

- 解压下载的压缩包。我下载的是：
  ![](https://pic.imgdb.cn/item/6739bdf6d29ded1a8c910409.png)

- 将文件解压到服务器的一个目录
    ```bath
    tar -xzvf /home/frp/frp_0.61.0_linux_amd64.tar.gz -C /home/frp
    ```

    ![](https://pic.imgdb.cn/item/6739be3fd29ded1a8c913926.png)

- 解压出来的文件中有 frps 和 frpc 两个文件.
  frps 是 frp 服务器端，frpc 是 frp 客户端。
  ![](https://pic.imgdb.cn/item/6739bf49d29ded1a8c925672.png)
 
- 我的文件目录是：
    ```bash
    /home/frp/frp_0.61.0_linux_amd64
    ```
    我们在服务器端这里只保留 frps 文件，删除 frpc 文件。
    ```bath
    rm -rf /home/frp/frp_0.61.0_linux_amd64/frpc*
    ```
- 下面配置服务器端的配置文件`frps.toml`随后保存。
   配置文件内容示例如下：
    ```toml
    # 绑定地址，监听所有网络接口
    bindAddr = "0.0.0.0"

    # 监听的端口，用于接收客户端连接
    bindPort = 7100

    # KCP 协议绑定的端口，与 bindPort 相同
    kcpBindPort = 7100

    # Web 管理界面的配置
    webServer.addr = "0.0.0.0"  # Web 管理界面绑定的地址
    webServer.port = 7500       # Web 管理界面的端口
    webServer.user = "user"     # Web 管理界面的用户名
    webServer.password = "password"  # Web 管理界面的密码

    # 日志配置
    log.to = "/frpslog/frps.log"  # 日志文件路径
    log.level = "info"            # 日志级别，可选值有 debug, info, warn, error
    log.maxDays = 3               # 日志文件保留天数

    # 认证配置
    auth.method = "token"         # 认证方法，使用 token 进行认证
    auth.token = "tokentoken"     # 认证 token，客户端需要提供相同的 token 才能连接

    # 允许的端口范围
    allowPorts = [
        { start = 6000, end = 7000 },  # 允许客户端使用的端口范围
    ]
    ```
**配置systemd 来管理 frps 服务**
- 在 Linux 系统下使用 systemd 来管理 frps 服务，包括启动、停止、配置后台运行和设置开机自启动。
- 安装 systemd
如果Linux 服务器上尚未安装 systemd，可以使用包管理器如 yum（适用于 CentOS/RHEL）或 apt（适用于 Debian/Ubuntu）来安装它：
    ```bash
    # 使用 yum 安装 systemd（CentOS/RHEL）
    yum install systemd

    # 使用 apt 安装 systemd（Debian/Ubuntu）
    apt install systemd
    ```
- 创建 frps.service 文件
使用文本编辑器 (例如 vim) 在 /etc/systemd/system 目录下创建一个 frps.service 文件，用于配置 frps 服务。
    ```bash
    sudo vim /etc/systemd/system/frps.service
    ```
    写入内容：
    ```service
    [Unit]
    # 服务名称，可自定义(建议不要更改，以后名字就叫frp server)
    Description = frp server
    After = network.target syslog.target
    Wants = network.target

    [Service]
    Type = simple
    # 启动frps的命令，需修改为您的frps的安装路径
    # 这里写的是我的安装路径
    ExecStart = /home/frp/frp_0.61.0_linux_amd64/frps -c /home/frp/frp_0.61.0_linux_amd64/frps.toml

    [Install]
    WantedBy = multi-user.target
    ```
- 设定好后，随后可以使用启动服务。然后查看服务状态。
- 启动服务相关命令如下：
    ```bash
    # 启动frp
    sudo systemctl start frps
    # 停止frp
    sudo systemctl stop frps
    # 重启frp
    sudo systemctl restart frps
    # 查看frp状态
    sudo systemctl status frps
    ```
- 设置开机启动
    ```bash
    sudo systemctl enable frps
    ```
- 这时候如果没有办法访问，你需要设置安全组，让其打开这个端口（一般的云服务器都需要设置）。除此之外还要设置防火墙允许访问。
- 设置后要重启服务器：
```bash
# 允许 FRP 服务器端口 7100
sudo firewall-cmd --zone=public --add-port=7100/tcp --permanent
sudo firewall-cmd --zone=public --add-port=7100/udp --permanent

# 允许 FRP Web 管理界面端口 7500
sudo firewall-cmd --zone=public --add-port=7500/tcp --permanent
sudo firewall-cmd --zone=public --add-port=7500/udp --permanent

# 允许 FRP 客户端使用的端口范围 6000-7000
sudo firewall-cmd --zone=public --add-port=6000-7000/tcp --permanent
sudo firewall-cmd --zone=public --add-port=6000-7000/udp --permanent
# 开放所有端口
sudo firewall-cmd --zone=public --list-all
# 重新加载防火墙规则
sudo firewall-cmd --reload

# 查看所有开放的端口
sudo firewall-cmd --list-ports
```
![](https://pic.imgdb.cn/item/6739d77fd29ded1a8ca64154.png)
- 访问7500端口可以直接输入账户密码登录服务端（这是就是之前配置文件里的账户密码）
![](https://pic.imgdb.cn/item/6739d7c8d29ded1a8ca67e2b.png)

![](https://pic.imgdb.cn/item/6739d833d29ded1a8ca6d38d.png)

## 配置客户端
我们使用Win10安装客户端，从 GitHub 的 [Release](https://github.com/fatedier/frp/releases) 页面中下载最新版本的客户端windows版本：`frp_0.61.0_windows_amd64.zip`。我们删除服务端的代码，然后安装客户端。
![](https://pic.imgdb.cn/item/6739dbdfd29ded1a8caa3c1f.png)

现在我在配置客户端的时候，在客户端5244端口有运行一个服务,下面我们写配置把这个服务放到公网上:
![](https://pic.imgdb.cn/item/6739ff34d29ded1a8cc99efc.png)

**在需要被访问的内网机器上部署 frpc**

 ![](https://pic.imgdb.cn/item/673a018ed29ded1a8ccbbb4c.png)
 `frpc.toml` 文件，假设 frps 所在服务器的公网 IP 地址为 110.x.x.147。以下是示例配置：

 ```toml
 # 指定 frps 服务器的地址，这里是局域网内的 IP 地址
serverAddr = "110.x.x.147"
# 指定 frps 服务器监听的端口，必须与服务器端配置一致
serverPort = 7100
# 登录失败时是否退出程序，默认为 false，设置为 true 表示一旦登录失败则立即退出
loginFailExit = true

# 日志文件的存储位置
log.to = "./frpc.log"
# 日志级别，可选值有 debug, info, warn, error
log.level = "info"
# 日志文件保留天数
log.maxDays = 3

# 认证方式，这里使用 token 进行认证
# 需要在服务器端配置相同
# ---------
auth.method = "token"
# 认证 token，必须与服务器端配置相同
auth.token = "tokentoken"
# ---------

# 定义一个代理，名称为 Factorio
[[proxies]]
name = "Factorio"
# 代理类型，这里使用 udp
type = "udp"
# 本地服务的 IP 地址，通常为 127.0.0.1
localIP = "127.0.0.1"
# 本地服务的端口
localPort = 34197
# 远程映射的端口，即外部通过此端口访问内部服务
remotePort = 6001

# 定义另一个代理，名称为 Minecraft
[[proxies]]
name = "Minecraft"
# 代理类型，这里使用 tcp
type = "tcp"
# 本地服务的 IP 地址，通常为 127.0.0.1
localIP = "127.0.0.1"
# 本地服务的端口
localPort = 25565
# 远程映射的端口，即外部通过此端口访问内部服务
remotePort = 6002
 ```
后续设置好配置后，在当前文件夹打开cmd，运行命令启动服务：
```bash
frpc.exe -c frpc.toml
```
- 服务器端配置好后，客户端就可以通过公网IP访问了。
下面是客户端启动时候的一些状态：
![](https://pic.imgdb.cn/item/673a0873d29ded1a8cd250a2.png)
在web端口可以看见新配置好的服务，我配置的alist项目（服务）：
![](https://pic.imgdb.cn/item/673a095ed29ded1a8cd34003.png)

**frp 配置文件示例**
 可以在官方文档查看更多的示例：https://gofrp.org/zh-cn/docs/examples/
 ![](https://pic.imgdb.cn/item/673a17dbd29ded1a8ce064d1.png)