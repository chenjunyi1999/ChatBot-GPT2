from nltk.translate.bleu_score import sentence_bleu
from sumeval.metrics.rouge import RougeCalculator


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


if __name__ == '__main__':
    # blue
    reference = [['this', 'is', 'small']]
    candidate = ['this', 'is', 'a', 'small']
    belu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    belu_2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    belu_4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))

    # rouge 1/2/l
    rouge = RougeCalculator(stopwords=True, lang="en")
    reference = "I went to the Mars from my living town."
    candidate = "I went to Mars and living there"
    rouge_1 = rouge.rouge_n(reference, candidate, n=1)
    rouge_2 = rouge.rouge_n(reference, candidate, n=2)
    rouge_l = rouge.rouge_l(reference, candidate)
    # distinct 1/2
    print(distinct("i have a qq", 1))
    print(distinct("i have a qq", 2))
    print(distinct("i have a qq a", 1))
    print(distinct("i have a qq a", 2))
    print(distinct("i have qq a qq a", 1))
    print(distinct("i have qq a qq a", 2))
