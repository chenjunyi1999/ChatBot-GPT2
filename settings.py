from torch import cuda


config = {
    'ckpt_dir': 'saved_models',
    'seed': 42,
    'device': 'cuda' if cuda.is_available() else 'cpu',
    'train_frac': 0.85,
    'data_dir': 'data',
    'model_name': 'gpt2',
    'train_prefix': 'train',
    'valid_prefix': 'valid',
    'max_turns': 5,
    'max_len': 1024,
    'lr': 2e-5,
    'batch_size': 8,
    'num_workers': 0,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'bos_token': '<bos>',
    'sp1_token': '<sp1>',
    'sp2_token': '<sp2>',
    'end_command': 'Abort!',
    'top_p':0.9
}