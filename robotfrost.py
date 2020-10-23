#!/usr/bin/env python3
import argparse
from torch import load, save
from train import train
from poem import predict
from model import RobotFrost
from dataset import GutenbergDataset

# Define args
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--sequence-length', type=int, default=6)
parser.add_argument('--starting-text', type=str, default='')
parser.add_argument('--generate', type=bool, default=False)
parser.add_argument('--cuda', type=bool, default=False)
args = parser.parse_args()

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
