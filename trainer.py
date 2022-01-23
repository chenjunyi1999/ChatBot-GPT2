from settings import config
from utils import seed_everything
from tqdm import tqdm
import numpy as np
import torch
import math
import os
import sys


class Trainer():
    def __init__(self,model,optim,sched,args,train_loader,valid_loader,writer):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.best_loss = sys.float_info.max
        self.last_epoch = 0
        if self.args.ckpt_name is not None:
            if os.path.exists(f"{config['ckpt_dir']}/{self.args.ckpt_name}.ckpt"):
                print("Loading the pre-trained checkpoint...")
                ckpt = torch.load(f"{config['ckpt_dir']}/{self.args.ckpt_name}.ckpt", map_location=config['device'])
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.optim.load_state_dict(ckpt['optim_state_dict'])
                self.sched.load_state_dict(ckpt['sched_state_dict'])
                self.best_loss = ckpt['loss']
                self.last_epoch = ckpt['epoch']
            else:
                print("Can't find the file! Training will start with the initialized model.")

    def train(self):
        seed_everything(config['seed'])
        print("training processing...")
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, start_epoch + config['num_epochs']):
            self.model.train()

            print(f"#" * 50 + f"Epoch: {epoch}" + "#" * 50)
            train_losses = []
            train_ppls = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(config['device']), token_type_ids.to(config['device']), labels.to(config['device'])

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                # 训练三个步骤 0.清空梯度 1.计算loss 2.反向传播backward 3.step(这里有两个optim和sched)
                loss, logits= outputs[0], outputs[1]
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.sched.step()

                # 当前截断
                train_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                train_ppls.append(ppl)

                # if i % 200 == 0:
                #     loss_ = [loss.item() for loss in train_losses]
                #     loss_ = np.mean(loss_)
                #     ppls_ = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls]
                #     ppls_ = np.mean(ppls_)
                #     print(f"batch: {i} || loss: {loss_} || ppls: {ppls_}")

            # 计算一个epoch中loss和ppl的平均值
            train_losses = [loss.item() for loss in train_losses]
            train_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls]
            train_loss = np.mean(train_losses)
            train_ppl = np.mean(train_ppls)
            print(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")

            # writer语法 writer.add_scalar('name', y(value) , x)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("PPL/train", train_ppl, epoch)
            self.last_epoch += 1

            valid_loss, valid_ppl = self.validation()

            # 如果当前的验证集损失比之前所有的都好 跟新best_loss 并保存当前的模型
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'sched_state_dict': self.sched.state_dict(),
                    'loss': self.best_loss,
                    'epoch': self.last_epoch
                }

                torch.save(state_dict,
                           f"{config['ckpt_dir']}/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
                print("*" * 10 + "Current best checkpoint is saved." + "*" * 10)
                print(f"{config['ckpt_dir']}/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")

            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")

            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("PPL/valid", valid_ppl, epoch)

            self.writer.add_scalars("Losses", {
                'train': train_loss,
                'valid': valid_loss,
            }, epoch)
            self.writer.add_scalars("PPLs", {
                'train': train_ppl,
                'valid': valid_ppl,
            }, epoch)

        print("Training finished!")

    def validation(self):
        seed_everything(config['seed'])
        print("Validation processing...")
        self.model.eval()

        valid_losses = []
        valid_ppls = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, labels = batch
                input_ids, token_type_ids, labels = \
                    input_ids.to(config['device']), token_type_ids.to(config['device']), labels.to(config['device'])

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss, logits = outputs[0], outputs[1]

                valid_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                valid_ppls.append(ppl)

            valid_losses = [loss.item() for loss in valid_losses]
            valid_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls]
            valid_loss = np.mean(valid_losses)
            valid_ppl = np.mean(valid_ppls)

            if math.isnan(valid_ppl):
                valid_ppl = 1e+8

        return valid_loss, valid_ppl

