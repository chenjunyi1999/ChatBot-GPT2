# ChatBot-GPT2
## 简介
一个使用`Pytorch`和`Huggingface Transofrmers` 构建的 `gpt2` 多轮聊天机器人 <br>

## 项目结构 
`process_data.py` : 处理四个格式不同的数据集的一些方法 <br>
`load_data.py` : 调用process_data.py 将四个不同的数据集合并保存为json形式<br>
`mydataset.py` : 定义数据集以及与数据处理相关的函数<br>
`main.py` : 主函数<br>
`trainer.py` : 定义模型训练与验证方法<br>
`predictor.py` : 定义模型预测与交互方法<br>
`evaluator.py`: 定义评估标准包括(Bleu,Rouge, Distinct) <br>
`settings.py` : 项目配置参数<br>
`utils.py` : 工具类<br>

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
4. **模型评估**
```
python main.py --mode="evaluate" --ckpt_name="best"
```
5. **推理和交互**
```
python main.py --mode="infer" --ckpt_name="best"
```

## 参考
1. [ChatBot-GPT2总结](https://github.com/chenjunyi1999/ML-Tutorial/tree/main/Project_Notes/ChatBot-GPT2%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0)
2. [huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)
3. [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)
4. [devjwsong/gpt2-dialogue-generation-pytorch](https://github.com/devjwsong/gpt2-dialogue-generation-pytorch)


