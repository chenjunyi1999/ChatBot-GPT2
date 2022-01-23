import torch
from settings import config
from itertools import chain
from torch.nn import functional as F
from utils import seed_everything


class Predictor():
    def __init__(self,tokenizer,model,vocab):
        self.tokenizer = tokenizer
        self.model = model
        self.bos_id = vocab[config['bos_token']]
        self.eos_id = vocab[tokenizer.eos_token]
        self.sp1_id = vocab[config['sp1_token']]
        self.sp2_id = vocab[config['sp2_token']]

    def infer(self):
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{config['end_command']}\".")

        self.model.eval()
        seed_everything(config['seed'])

        with torch.no_grad():
            input_hists = []

            while True:
                utter = input("You: ")
                if utter == config['end_command']:
                    print("Bot: Good bye.")
                    break

                input_ids = [self.sp1_id] + self.tokenizer.encode(utter)
                input_hists.append(input_ids)

                if len(input_hists) >= config['max_turns']:
                    num_exceeded = len(input_hists) - config['max_turns']
                    input_hists = input_hists[num_exceeded:]

                input_ids = [self.bos_id] + list(chain.from_iterable(input_hists)) + [self.sp2_id]
                start_sp_id = input_hists[0][0]
                next_sp_id = self.sp1_id if start_sp_id == self.sp2_id else self.sp2_id
                token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in
                                  enumerate(input_hists)]
                assert len(token_type_ids) == len(input_hists)
                token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.sp2_id]
                assert len(input_ids) == len(token_type_ids)
                input_len = len(input_ids)

                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(config['device'])
                token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(config['device'])

                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)

                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                print(f"Bot: {res}")
                input_hists.append([self.sp2_id] + self.tokenizer.encode(res))

    # top-k核采样 （beam search的改进版本）
    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, config['max_len']):
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos - 1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)

            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > config['top_p']
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)

            probs = torch.zeros(output.shape, device=config['device']).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)

            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)

            if idx_item == self.eos_id:
                break

            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.sp2_id]]).to(config['device'])
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape

        return output_ids