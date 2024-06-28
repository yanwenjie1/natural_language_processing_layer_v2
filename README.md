# 分层模型V2
与分层模型（旧版）相比，优化了项目结构，优化了推理方式，项目功能与精度上并无太多拓展与优化


## 与旧版相比-性能参数比对
默认参数: 

滑窗行数:max_seq_len=128 

每行Token:max_word_len=40

预训练模型:roberta-small
<table>
  <tr>
    <th style="text-align:center">模型版本</th>
    <th style="text-align:center">batch_size</th>
    <th style="text-align:center">训练显存</th>
    <th style="text-align:center">A4000单卡训练速度</th>
    <th style="text-align:center">A6000单卡训练速度</th>
  </tr>
  <tr>
    <td style="text-align:center">V1</td>
    <td style="text-align:center">16</td>
    <td style="text-align:center">-</td>
    <td style="text-align:center">-</td>
    <td style="text-align:center">-</td>
  </tr>
    <tr>
    <td style="text-align:center">V2</td>
    <td style="text-align:center">16</td>
    <td style="text-align:center">13.8G</td>
    <td style="text-align:center">-</td>
    <td style="text-align:center">-</td>
  </tr>
</table>

<table>
  <tr>
    <th style="text-align:center">模型版本</th>
    <th style="text-align:center">batch_size</th>
    <th style="text-align:center">推理显存</th>
    <th style="text-align:center">A4000单卡推理速度</th>
    <th style="text-align:center">A6000单卡推理速度</th>
  </tr>
  <tr>
    <td style="text-align:center">V1</td>
    <td style="text-align:center">8</td>
    <td style="text-align:center">1.3G</td>
    <td style="text-align:center">4000行/S</td>
    <td style="text-align:center">-</td>
  </tr>
    <tr>
    <td style="text-align:center">V2</td>
    <td style="text-align:center">8</td>
    <td style="text-align:center">1.3G</td>
    <td style="text-align:center">-</td>
    <td style="text-align:center">25000行/S</td>
  </tr>
</table>


## 环境依赖
注意：与旧版相比，很多包版本做了升级
```
conda create -n yourname python==3.10.14
conda activate yourname / source activate yourname
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.40.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
pip install onnxruntime-gpu==1.17.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
pip install rich==12.5.1
pip install flask
pip install gevent
pip install bs4
```

## 项目结构
```html
|-- config.py
|-- README.md
|-- server.py
|-- train.py
|-- test.py
|-- data
|   |-- 财务附注定位V2
|       |-- __init__.py
|       |-- dev_data.pkl
|       |-- labels.json
|       |-- preprocess.py
|       |-- train_data.pkl
|       |-- 样本1.json
|-- utils
    |-- adversarial_training.py
    |-- base_model.py
    |-- functions.py
    |-- onnx_convert.py
    |-- server_models.py
    |-- train_class.py
    |-- train_models.py
```

## 快速开始

```html
一、训练
1.制造样本，并导入Label_Studio平台，使用html标注
2.标注后导出标注结果，模仿项目【data/财务附注定位V2】，新建项目，拷贝preprocess.py
3.执行preprocess.py
4.修改config内data_dir参数
5.修改train.py脚本，新增一个if分支，然后执行train.py
二、服务
1.修改server.py内的model_name变量
2.运行server.py即可
三、升级
参考docker升级手册和Dockerfile即可，可在107.46上部署升级
```


## 注意事项

```html
1.本质上仍然是以行为细粒度做的ner任务，暂不支持解决嵌套问题，但实际上已经可以解决嵌套实体识别和关系抽取问题，如有需求，联系管理员
2.这种任务并不复杂，用roberta-small足矣
3.如果想获得更好的外推性，联系管理员，但一般情况下，优化样本即可
4.【财务附注定位V2】完整文件位于【技术中心\0AI\DeepLearning\通用语料】
```

## label_studio平台标签

```html
标签体系范例
<View>
  <Filter toName="ner" minlength="0" name="filter"/>
  <HyperTextLabels name="ner" toName="text">
	<Label value="货币资金" background="#FFA39E"/>
	<Label value="合并财务报表附注" background="#ff9eea"/>
	<Label value="现金流量表" background="#0d87d3"/>
  </HyperTextLabels>

  <View style="border: 1px solid #CCC;                border-radius: 10px;                padding: 5px">
    <HyperText name="text" value="$html" granularity="paragraph"/>
  </View>
</View>
```


 ## 更新日志
 - 2024-01-18：修改了server.py和preprocess.py
 - 2024-01-18：补充提供了样例json文件
 - 2024-03-18：修改了server.py
