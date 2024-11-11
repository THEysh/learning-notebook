# 使用官方的Hugo镜像作为基础镜像
FROM hugo:latest
# 设置工作目录
WORKDIR /app
# 将当前目录下的所有文件复制到容器的/app目录下
COPY . .
# 构建Hugo站点
EXPOSE 1313

CMD ["hugo", "server", "--bind=0.0.0.0"]

#docker build -t notebook .