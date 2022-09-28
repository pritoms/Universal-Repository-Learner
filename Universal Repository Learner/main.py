import argparse
import os
import sys
sys.path.append('.')
from utils import str2bool
from data_preprocess import DataPreprocess
import json

parser = argparse.ArgumentParser(description='Universal repository learner')
parser.add_argument('--train', default='False', type=str2bool, help='train the model')
parser.add_argument('--test', default='False', type=str2bool, help='test the model')
parser.add_argument('--data_dir', default='../data', type=str, help='data directory')
parser.add_argument('--save_path', default='./checkpoint', type=str, help='model save path')
parser.add_argument('--config_path', default='./config.json', type=str, help='configuration file path')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
parser.add_argument('--log_interval', type=int, default=200, help='log interval')
parser.add_argument('--cuda', default='True', type=str2bool, help='use cuda')
parser.add_argument('--seed', type=int, default=20190408, help='random seed')

class Config(object):
    def __init__(self, config_path, **kwargs):
        with open(config_path, 'r') as handle:
            self.__dict__ = json.loads(handle.read())
            self.__dict__.update(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, config_path):
        with open(config_path, 'w') as handle:
            json.dump(self.__dict__, handle, indent=4, sort_keys=False)

    def update(self, config_path):
        with open(config_path, 'r') as handle:
            params = json.loads(handle.read())
            self.__dict__.update(params)

        for name, param in params.items():
            setattr(self, name, param)

def main():
    args = parser.parse_args()

    config = Config(args.config_path, **args.__dict__)

    train_loader = data_loader(config, 'train')
    eval_loader = data_loader(config, 'eval')
    test_loader = data_loader(config, 'test')

    data_preprocess = DataPreprocess(config, train_loader, eval_loader, test_loader)
    if args.train:
        data_preprocess.train()
    elif args.test:
        data_preprocess.test()
    else:
        raise NotImplementedError('No action specified!')

if __name__ == "__main__":
    main()
