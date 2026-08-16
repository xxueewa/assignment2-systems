from cs336_basics.model import BasicsTransformerLM
import timeit
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from custom_dataset import DecoderDataset
from utils import *
from torch import optim, nn
import matplotlib.pyplot as plt
import numpy as np

class ComputeBenchmark:
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int):
        self.vocab_size = vocab_size
        self.context_length = 512
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

    def get_model_size(self):
        calc_params = self.num_layers * (4 * self.d_model ** 2 + 3 * self.d_model * self.d_ff + 2 * self.d_model) + 2 * self.vocab_size * self.d_model + self.d_model
        model = BasicsTransformerLM(self.vocab_size, self.context_length, self.d_model, self.num_layers, self.num_heads, self.d_ff)
        print(model.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result = (calc_params == total_params)
        # Compute Profiling: 500 ~ 2,000 条
        # Total training token estimate
        token_requires = (total_params * 200 / 500) * 2 / 1024 / 1024
        print(f"计算总参数量: {calc_params} = 总参数量: {total_params} ? {result}, 可训练参数量: {trainable_params}, 所需训练集大小: {token_requires} GB")


    def read_from(self, file):
        """
        :param file:
        :return: A list of the lines in the file, each exactly 20 characters long
        """
        all_lines = []
        for line in open(file):
            all_lines.append(line[:-1]) # eat the \n
        print("%i lines read in" % len(all_lines))
        return all_lines

    def time_profile(self, num_epochs):
        # get device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Mac M1/M2/M3 chips
        else:
            device = torch.device("cpu")

        print(f"Device Type: {device.type}")

        # Constructs the vocabulary: lowercase letters a to z and space
        number = [chr(ord('0') + i) for i in range(0, 10)]
        vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' '] + number
        vocab_index = Indexer()
        for char in vocab:
            vocab_index.add_and_get_index(char)
        print(repr(vocab_index))

        char_to_idx = {
            char: vocab_index.index_of(char)
            for char in vocab
        }

        dataset = self.read_from("cs336_systems/evaluation_train.txt")
        dataset = DecoderDataset(dataset, char_to_idx, args.max_length)

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0
        )

        model = BasicsTransformerLM(self.vocab_size, self.context_length, self.d_model, self.num_layers, self.num_heads, self.d_ff).to(device)
        model.zero_grad()
        model.train()
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr)

        loss_fcn = nn.CrossEntropyLoss() 
        # 5 warm_up + 10 evaluation epochs
        forward_metrics = []
        backward_metrics = []
        optimizer_metrics = []
        for t in range(num_epochs):
            forward_latency = []
            backward_latency = []
            optimizer_latency = []
            for batch_idx, (input_token, target_token) in enumerate(train_loader):
                
                input_token, target_token = input_token.to(device), target_token.to(device)

                start_timestamp = timeit.default_timer()
                torch.cuda.nvtx.range_push("forward")
                prob = model(input_token)
                loss = loss_fcn(prob.reshape(-1, prob.size(-1)), target_token.reshape(-1))
                torch.cuda.nvtx.range_pop()
                model.zero_grad()
                forward_timestamp = timeit.default_timer()

                torch.cuda.nvtx.range_push("backward")
                loss.backward()
                torch.cuda.nvtx.range_pop()
                backward_timestamp = timeit.default_timer()

                torch.cuda.nvtx.range_push("optimizer")
                optimizer.step()
                torch.cuda.nvtx.range_pop()
                optimizer_timestamp = timeit.default_timer()

                forward_latency.append(forward_timestamp - start_timestamp)
                backward_latency.append(backward_timestamp - forward_timestamp)
                optimizer_latency.append(optimizer_timestamp - backward_timestamp)

            avg_forward, std_forward = np.mean(forward_latency), np.std(forward_latency)
            forward_metrics.append([avg_forward, std_forward])
            avg_backward, std_backward = np.mean(backward_latency), np.std(backward_latency)
            backward_metrics.append([avg_forward, std_forward])
            avg_optimizer, std_optimizer = np.mean(optimizer_latency), np.std(optimizer_latency)
            optimizer_metrics.append([avg_forward, std_forward])

        plt.plot(range(5, num_epochs + 1), np.array(forward_metrics)[:, 0])
        plt.plot(range(5, num_epochs + 1), np.array(forward_metrics)[:, 1])
        plt.plot(range(5, num_epochs + 1), np.array(backward_metrics)[:, 0])
        plt.plot(range(5, num_epochs + 1), np.array(backward_metrics)[:, 1])
        plt.plot(range(5, num_epochs + 1), np.array(optimizer_metrics)[:, 0])
        plt.plot(range(5, num_epochs + 1), np.array(optimizer_metrics)[:, 1])
        plt.legend(['forward', 'backward', 'optimizer'])
        plt.xlabel('Epoch')
        plt.ylabel('Latency')
        plt.title("End-to-End Benchmarking")
        plt.savefig("End_to_End_Benchmarking.png")
        plt.show()

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--vocab_size', type=int, default=37, help='vocab size')
    parser.add_argument('--d_model', type=int, default=768, help='d_model')
    parser.add_argument('--num_layers', type=int, default=12, help='d_ff')
    parser.add_argument('--num_heads', type=int, default=12, help='d_heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='d_ff')
    parser.add_argument('--train', type=str, default='data/lettercounting-train.txt', help='path to train examples')
    parser.add_argument('--dev', type=str, default='data/lettercounting-dev.txt', help='path to dev examples')
    parser.add_argument('--output_bundle_path', type=str, default='classifier-output.json', help='path to write the results json to (you should not need to modify)')
    parser.add_argument('--max-length', type=int, default=512, help='maximum number of characters per example')
    parser.add_argument('--batch-size', type=int, default=16, help='training and evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker processes')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    # based on the assignment, the model sizes to evaluate have five tiers
    model_size = {
        "d_model":[768, 1024, 1280, 2560, 4608],
        "num_layers":[12, 24, 36, 32, 50],
        "num_heads":[12, 16, 20, 32, 36],
        "d_ff":[3072, 4096, 5120, 10240, 12288],
    }

    for i in range(0, 5):
        benchmark = ComputeBenchmark(args.vocab_size, model_size["d_model"][i],  model_size["num_layers"][i], model_size["num_heads"][i], model_size["d_ff"][i])
        benchmark.get_model_size()
        benchmark.time_profile(15)