from settings import config
from utils import seed_everything
from trainer import Trainer
from predictor import Predictor
from mydataset import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

# 命令行参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str, required=False, help="The running mode: train or inference?")
parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")
args = parser.parse_args()

# 创建模型路径
if not os.path.exists(config['ckpt_dir']):
    os.makedirs(config['ckpt_dir'])

# Tokenizer 和 Vocab 的初始化
print("load the tokenizer and vocab...")
tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
special_tokens = {
    'bos_token': config['bos_token'],
    'additional_special_tokens': [config['sp1_token'], config['sp2_token']],
}
eos_token = tokenizer.eos_token
num_new_tokens = tokenizer.add_special_tokens(special_tokens)

vocab = tokenizer.get_vocab()
vocab_size = len(vocab)

config['bos_id'] = vocab[config['bos_token']]
config['eos_id'] = vocab[eos_token]
config['sp1_id'] = vocab[config['sp1_token']]
config['sp2_id'] = vocab[config['sp2_token']]

# 模型初始化
print("load the model...")
seed_everything(config['seed'])
model = GPT2LMHeadModel.from_pretrained(config['model_name']).to(config['device'])
model.resize_token_embeddings(vocab_size)
config['max_len'] = min(config['max_len'], model.config.n_ctx)

if args.mode == 'train':
    print("this is the train mode")
    # 优化器初始化
    print("load the optimizer...")
    optim = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    # 初始化数据
    print("Loading train & valid data...")
    train_set = CustomDataset(config['train_prefix'], config)
    valid_set = CustomDataset(config['valid_prefix'], config)
    ppd = PadCollate(eos_id=config['eos_id'])
    train_loader = DataLoader(train_set,
                              collate_fn=ppd.pad_collate,
                              shuffle=True,
                              batch_size=config['batch_size'],
                              num_workers=config['num_workers'],
                              pin_memory=True
                              )
    valid_loader = DataLoader(valid_set,
                              collate_fn=ppd.pad_collate,
                              shuffle=True,
                              batch_size=config['batch_size'],
                              num_workers=config['num_workers'],
                              pin_memory=True
                              )

    # 计算总的step数
    num_batches = len(train_loader)
    total_train_steps = num_batches*config['num_epochs']
    warmup_steps = int(config['warmup_ratio'] * total_train_steps)

    # 初始化schedule
    sched = get_polynomial_decay_schedule_with_warmup(optim,
                                                      num_warmup_steps=warmup_steps,
                                                      num_training_steps=total_train_steps,
                                                      power=2
                                                      )
    writer = SummaryWriter()
    trainer = Trainer(model, optim, sched, args, train_loader, valid_loader, writer)
    trainer.train()

elif args.mode == 'infer':
    print("this is the infer mode")
    if args.ckpt_name is not None:
        if os.path.exists(f"{config['ckpt_dir']}/{args.ckpt_name}.ckpt"):
            print("Loading the pre-trained checkpoint...")
            ckpt = torch.load(f"{config['ckpt_dir']}/{args.ckpt_name}.ckpt", map_location=config['device'])
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print("Can't inference!")
            exit()
    predictor = Predictor(tokenizer,model,vocab)
    predictor.infer()
else:
    print("mode error!")