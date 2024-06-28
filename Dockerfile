# 指定基础镜像
FROM 10.17.205.187/inforextraction/pytorch:2.2.2-cuda11.8-cudnn8-onnx1.17.1
# 构建镜像
RUN mkdir laying
# 拷贝文件
COPY / /laying
# 设置当前工作目录
WORKDIR /laying
# 指定容器启动时需要运行的命令 ENTRYPOINT命令可以追加命令
# RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ENTRYPOINT [ "python" ]
# 指定容器启动的时候需要执行的命令 只有最后一个命令会生效
CMD [ "server.py" ]