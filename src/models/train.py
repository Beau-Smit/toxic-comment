import argparse
import gzip
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py

class LSTM(nn.Module):

    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)
        print('text_emb', text_emb.shape)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        print('packed input', packed_input[0].shape, packed_input[1].shape)
        packed_output, _ = self.lstm(packed_input)
        print('packed output', packed_output[0].shape, packed_output[1].shape)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        print('output', output.shape)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        print('out_forward dim', out_forward.shape)
        out_reverse = output[:, 0, self.dimension:]
        print('out_reverse dim', out_reverse.shape)

        out_reduced = torch.cat((out_forward, out_reverse), 1)
        print('out_reduced', out_reduced.shape)
        text_fea = self.drop(out_reduced)
        print('text_fea', text_fea.shape)

        text_fea = self.fc(text_fea)
        print('text_fea2', text_fea.shape)
        text_fea = torch.squeeze(text_fea, 1)
        print('text_fea2', text_fea.shape)
        text_out = torch.sigmoid(text_fea)
        print('text_out', text_out.shape)

        return text_out

# Decode binary data from SM_CHANNEL_TRAINING
# Decode and preprocess data
# Create map dataset


def get_data():
    device = None
    
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
    fields = [('label', label_field), ('clean_text', text_field)]

    train, valid, test = TabularDataset.splits(path = './',
                                    train='train.csv', 
                                    validation='valid.csv',
                                    test='test.csv',
                                    format='CSV',
                                    fields=fields, 
                                    skip_header=True)
    # Iterators
    train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.clean_text),
                                device=device, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.clean_text),
                                device=device, sort=True, sort_within_batch=True)
    test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.clean_text),
                                device=device, sort=True, sort_within_batch=True)
    return train_iter, valid_iter, test_iter


def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda > 0 else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_iter, valid_iter, test_iter = get_data()

    model = LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    logger.info("Start training ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (labels, (text, text_len)) in enumerate(train_iter, 1):
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(text),
                        len(train_iter.sampler),
                        100.0 * batch_idx / len(train_iter),
                        loss.item(),
                    )
                )

        # test the model
        # test(net, test_loader, device)

    # save model checkpoint
    save_model(model, args.model_dir)
    return


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            test_loss += F.cross_entropy(output, labels, reduction="sum").item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{}, {})\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    return


def save_model(save_path, model, optimizer, valid_loss):
    
    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.9, metavar="BETA1", help="beta1 (default: 0.9)"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999, metavar="BETA2", help="beta2 (default: 0.999)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="WD",
        help="L2 weight decay (default: 1e-4)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)