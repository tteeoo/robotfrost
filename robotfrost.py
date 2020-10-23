#!/usr/bin/env python3
import argparse

# Define args
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e',
    '--epochs',
    type = int,
    help = 'the number of epochs to go through when training',
    default = 20
)
parser.add_argument(
    '-b',
    '--batch-size',
    type = int,
    help = 'the batch size to be used when training',
    default = 128
)
parser.add_argument(
    '-s',
    '--sequence-length',
    type = int,
    help = 'the sequence length to be used when training',
    default = 16
)
parser.add_argument(
    '-c',
    '--cuda',
    type = bool,
    help = 'wether or not to use the gpu when training',
    default = False
)
parser.add_argument(
    '-g',
    '--generate',
    type = bool,
    help = 'set to True to generate a poem instead of train',
    default = False
)
parser.add_argument(
    '-t',
    '--starting-text',
    help = 'the starting text to be used when generating',
    type = str,
    default = ''
)
args = parser.parse_args()

# Import here so not to have a delay for --help
from torch import load, save
from train import train
from poem import predict
from model import RobotFrost
from dataset import GutenbergDataset

# Instantiate model and dataset
dataset = GutenbergDataset(args.sequence_length)
model = RobotFrost(dataset)

# Try to load model
try:
    model.load_state_dict(load('./state.dict.pth'))
except Exception as e:
    print('exception,', e, '... using new model')

# Predict
if args.generate:
    print(' '.join(predict(model, dataset, args.starting_text)))
    exit(0)

# Train
try:
    if args.cuda: model = model.to('cuda')
    train(model, dataset, args.epochs, args.batch_size, args.sequence_length, args.cuda)
except KeyboardInterrupt:
    pass

# Save model
save(model.state_dict(), './state.dict.pth')
