FROM ubuntu:latest
# 设置环境变量，避免交互式配置
ENV DEBIAN_FRONTEND=noninteractive
# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . .

# 解压Hugo压缩包并移动到系统路径
RUN tar -xzf hugofiles/hugo_extended_0.125.7_Linux-64bit.tar.gz -C hugofiles/ && \
    mv hugofiles/hugo /usr/local/bin/ 

# 暴露Hugo服务器的端口
EXPOSE 1313


# docker build -t hugo_image .
# docker run -it --name my_container hugo_image bash 
