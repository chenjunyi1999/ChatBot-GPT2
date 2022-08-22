import json
import os

from tqdm import tqdm
from transformers import GPT2Tokenizer

from process_data import load_daily, load_empathetic, load_persona, load_blended
from settings import config

dataset_list = ['daily_dialog', 'empathetic_dialogues', 'persona_chat', 'blended_skill_talk']


# 合并四个数据集
def merge_data(tokenizer, config):
    train_dialogues = []
    valid_dialogues = []
    num_train = 0
    num_valid = 0
    # 加载四个数据集
    for data_name in dataset_list:
        print(f"Processing {data_name}...")
        if data_name == 'daily_dialog':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_daily(tokenizer, config['train_frac'])
        elif data_name == 'empathetic_dialogues':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_empathetic(tokenizer, config['train_frac'])
        elif data_name == 'persona_chat':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_persona(tokenizer, config['train_frac'])
        elif data_name == 'blended_skill_talk':
            part_train_dialogues, part_valid_dialogues, part_num_train, part_num_valid = load_blended(tokenizer, config['train_frac'])

        train_dialogues += part_train_dialogues
        valid_dialogues += part_valid_dialogues

        print("#" * 50 + f"Analysis on {data_name}" + "#" * 50)
        print(f"The number of train dialogues: {len(part_train_dialogues)}")
        print(f"The number of valid dialogues: {len(part_valid_dialogues)}")
        print(f"The number of train utterances: {part_num_train}")
        print(f"The number of valid utterances: {part_num_valid}")

        num_train += part_num_train
        num_valid += part_num_valid
    return train_dialogues, valid_dialogues, num_train, num_valid


# 保存合并的结果
def save_data(prefix, data_dir, dialogues, tokenizer):
    # text
    print(f"Saving {prefix} text file...")
    with open(f"{data_dir}/{prefix}_utters.json", 'w') as f:
        json.dump(dialogues, f)

    # idx
    print(f"Saving {prefix} idx file...")
    ids = []
    for dialogue in tqdm(dialogues):
        dialogue_ids = []
        for utter in dialogue:
            tokens = tokenizer.tokenize(utter)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            dialogue_ids.append(token_ids)
        ids.append(dialogue_ids)

    with open(f"{data_dir}/{prefix}_ids.json", 'w') as f:
        json.dump(ids, f)


if __name__ == '__main__':

    data_dir = f"{config['data_dir']}"
    if os.path.exists(data_dir):
        print("the data has already existed!")
    else:
        os.makedirs(data_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])

        print("Loading & Merging all datasets...")
        train_dialogues, valid_dialogues, num_train, num_valid = merge_data(tokenizer, config)

        print("Saving train data...")
        save_data(config['train_prefix'], data_dir, train_dialogues, tokenizer)
        print("Saving validation data...")
        save_data(config['valid_prefix'], data_dir, valid_dialogues, tokenizer)

        print("#" * 50 + "Analysis on total data" + "#" * 50)
        print(f"The number of train dialogues: {len(train_dialogues)}")
        print(f"The number of valid dialogues: {len(valid_dialogues)}")
        print(f"The number of train utterances: {num_train}")
        print(f"The number of valid utterances: {num_valid}")
