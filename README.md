# ChatBot-Pytorch
A GPT-2 ChatBot implemented using Pytorch and Huggingface-transformers


## How to use
1. install the required packages
```
pip install -r requirements.txt
```
2. Download and process the data
```
python load_data.py
```
3. train the model

  you can train the model from a initial state
```
python main.py --mode="train"
```

  also you can train a model from  a specific checkpoint (eg. best.ckpt)
```
python main.py --mode="train" --ckpt_name="best"
```
4. infer&interact
```
python main.py --mode="infer" --ckpt_name="best"
```
