import re
import torch
from nltk.translate.bleu_score import sentence_bleu
from numpy import mean
from sumeval.metrics.rouge import RougeCalculator
from torch.nn import functional as F
from settings import config
from utils import seed_everything
import numpy as np

punctuation = '.%!,;:?"\、，；'


def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
    return text.strip()


# 以ngram统计词频
def get_dict(word_list, ngram):
    word_dict = {}
    length = len(word_list)
    for i in range(0, length - ngram + 1):
        ngram_token = "".join(word_list[i:i + ngram])
        if word_dict.get(ngram_token) is not None:
            word_dict[ngram_token] += 1
        else:
            word_dict[ngram_token] = 1
    return word_dict


def distinct(raw_sentence, ngram):
    import string
    raw_word_list = list(raw_sentence)
    word_list = []
    for i in range(len(raw_word_list)):
        if raw_word_list[i] in string.punctuation:
            continue
        word_list.append(raw_word_list[i])
    sentence = ''.join(word_list)
    word_list = sentence.split(' ')
    word_dict = get_dict(word_list, ngram)
    total = 0.0
    distict = 0.0
    for key, value in word_dict.items():
        total += value
        distict += 1
    if total == 0.0:
        print("divide 0 error")
        return -1
    else:
        return distict / total


class Evaluator():
    def __init__(self, tokenizer, model, vocab, valid_loader):
        self.tokenizer = tokenizer
        self.model = model
        self.bos_id = vocab[config['bos_token']]
        self.eos_id = vocab[tokenizer.eos_token]
        self.sp1_id = vocab[config['sp1_token']]
        self.sp2_id = vocab[config['sp2_token']]
        self.valid_loader = valid_loader

    def evaluate(self):

        self.model.eval()
        seed_everything(config['seed'])

        rouge = RougeCalculator(stopwords=True, lang="en")

        bleu1_list = []
        bleu2_list = []
        bleu4_list = []
        rouge1_list = []
        rouge2_list = []
        rougel_list = []
        distinct1_list = []
        distinct2_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.valid_loader):
                input_ids, token_type_ids, labels = batch
                for single_input_ids, single_type_ids, single_label in zip(input_ids, token_type_ids, labels):

                    # ---------- delete the last sentence ----------
                    b = np.argwhere(single_input_ids.numpy() == 50259)[-1,-1]
                    single_input_ids[b + 1:] = 50256

                    single_input_ids = torch.LongTensor(single_input_ids).unsqueeze(0).to(config['device'])
                    single_type_ids = torch.LongTensor(single_type_ids).unsqueeze(0).to(config['device'])
                    output_ids = self.nucleus_sampling(single_input_ids, single_type_ids, len(single_input_ids))
                    candidate = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    single_label = single_label.numpy().tolist()
                    single_label_clean = []
                    for i in single_label:
                        if i != -100:
                            single_label_clean.append(i)
                    reference = self.tokenizer.decode(single_label_clean, skip_special_tokens=True)

                    # 把句子转成字符串列表 输入 bleu
                    reference_l = [removePunctuation(reference).strip().split()]
                    candidate_l = removePunctuation(candidate).strip().split()
                    # print(reference_l)
                    # print(candidate_l)
                    bleu1_list.append(sentence_bleu(reference_l, candidate_l, weights=(1, 0, 0, 0)))
                    bleu2_list.append(sentence_bleu(reference_l, candidate_l, weights=(0.5, 0.5, 0, 0)))
                    bleu4_list.append(sentence_bleu(reference_l, candidate_l, weights=(0.25, 0.25, 0.25, 0.25)))
                    rouge1_list.append(rouge.rouge_n(reference, candidate, n=1))
                    rouge2_list.append(rouge.rouge_n(reference, candidate, n=2))
                    rougel_list.append(rouge.rouge_l(reference, candidate))
                    distinct1_list.append(distinct(candidate, 1))
                    distinct2_list.append(distinct(candidate, 1))
                    # print(bleu1_list[-1])
                    # print(bleu2_list[-1])
                    # print(bleu4_list[-1])
                    # print(rouge1_list[-1])
                    # print(rouge2_list[-1])
                    # print(rougel_list[-1])
                    # print(distinct1_list[-1])
                    # print(distinct2_list[-1])
        print("final evaluation results (contains:bleu/rouge/distinct...) ")
        print(mean(bleu1_list))
        print(mean(bleu2_list))
        print(mean(bleu4_list))
        print(mean(rouge1_list))
        print(mean(rouge2_list))
        print(mean(rougel_list))
        print(mean(distinct1_list))
        print(mean(distinct2_list))

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
