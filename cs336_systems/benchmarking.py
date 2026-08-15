from cs336_basics.model import BasicsTransformerLM
import timeit
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

class ComputeBenchmark:
    """
    Scaling Law
    观测区间通常在 5B 到 500B 个 Token
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int):
        self.vocab_size = vocab_size
        self.context_length = 512
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

    def get_model_size(self, epoch):
        model = BasicsTransformerLM(self.vocab_size, self.context_length, self.d_model, self.num_layers, self.num_heads, self.d_ff)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params}, 可训练参数量: {trainable_params}")


    def time_profile(self, epoch):
        # get device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Mac M1/M2/M3 chips
        else:
            device = torch.device("cpu")

        print(f"Device Type: {device.type}")

        # data_set = load_dataset("roneneldan/TinyStories")
        # train_set = data_set["train"]
        # validate_set = data_set["validation"]
        # train_dataset = DecoderDataset(train_set, char_to_idx, args.max_length)
        # validate_dataset = DecoderDataset(validate_set, char_to_idx, args.max_length)

        model = BasicsTransformerLM(self.vocab_size, self.context_length, self.d_model, self.num_layers, self.num_heads, self.d_ff).to(device)
        # model.train()
        # start_timestamp = timeit.default_timer()
        # for t in range(epoch):
        #     for batch_idx, (input_token, output_token) in enumerate(train_loader):
        #         input_token, output_token = input_token.to(device), output_token.to(device)

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--vocab_size', type=str, default=37, help='vocab size')
    parser.add_argument('--d_model', type=str, default=768, help='d_model')
    parser.add_argument('--num_layers', type=str, default=12, help='d_ff')
    parser.add_argument('--num_heads', type=str, default=12, help='d_heads')
    parser.add_argument('--d_ff', type=str, default=3072, help='d_ff')
    parser.add_argument('--train', type=str, default='data/lettercounting-train.txt', help='path to train examples')
    parser.add_argument('--dev', type=str, default='data/lettercounting-dev.txt', help='path to dev examples')
    parser.add_argument('--output_bundle_path', type=str, default='classifier-output.json', help='path to write the results json to (you should not need to modify)')
    parser.add_argument('--max-length', type=int, default=5462, help='maximum number of characters per example')
    parser.add_argument('--batch-size', type=int, default=16, help='training and evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker processes')
    args = parser.parse_args()
    return args

if __name__ == 'main' :
    args = _parse_args()
    benchmark = ComputeBenchmark(args.vocab_size, args.d_model, args.num_layers, args.num_heads, args.dff)
    print(benchmark.get_model_size)