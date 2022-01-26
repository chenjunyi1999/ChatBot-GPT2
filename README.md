# ChatBot-Pytorch
一个使用`Pytorch`和`Huggingface Transofrmers` 构建的 `gpt2` 聊天机器人


## 如何使用
1. **安装依赖库**
```
pip install -r requirements.txt
```
2. **下载并处理数据集**
```
python load_data.py
```
3. **训练模型**

  &emsp; 你可以从初始状态训练一个模型
```
python main.py --mode="train"
```

  &emsp; 你也可以从一个保存过后的`checkpoint`处开始训练(例如文件名为`best.ckpt`)
```
python main.py --mode="train" --ckpt_name="best"
```
4. **推理和交互**
```
python main.py --mode="infer" --ckpt_name="best"
```
