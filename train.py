#
# Training script for autoencoder
#
# Author: Markus Schmitt
# Date: Feb 2021
#

import torch
from torch import optim
import numpy as np
import json
import time

import sys

from model import BottleneckNet, NoLatentNet

# Read input file
if len(sys.argv)<2:
    raise Exception("Input file missing. Run program with 'python train.py your_input.txt'")
    exit()
inFile=sys.argv[1]
with open(inFile,'r') as f:
    params = json.load(f)

# Set numpy rng seed
np.random.seed(0)

# Load training data and shuffle it
data=np.loadtxt(params['data']['input_file'])
data=data[np.random.permutation(len(data))]

# Get output locations
outDir=params['data']['output_dir']
outputFile=outDir+params['data']['output_file_name']

# Get model parameters
numLatent=int(params['model']['num_latent'])
encoderWidth=int(params['model']['encoder_width'])
decoderWidth=int(params['model']['decoder_width'])
inputDim=len(data[0])

# Get optimization parameters
numEpochs=int(params['training']['number_of_epochs'])
learningRate=float(params['training']['learning_rate'])

# Set Pytorch rng seed
seed=float(params['training']['seed'])
torch.manual_seed(seed)

# Divide data into training and test data sets 
trainFrac=float(params['training']['training_fraction'])
trainFracIdx=int(trainFrac*len(data))

train_data=torch.as_tensor(data[:trainFracIdx], dtype=torch.float32)
test_data=torch.as_tensor(data[trainFracIdx:], dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

test_dataset = torch.utils.data.TensorDataset(test_data, test_data)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

# Initialize network
net = None
if numLatent>0:
    net = BottleneckNet(numLatent, inputDim, encoder_dim=[encoderWidth,encoderWidth], decoder_dim=[decoderWidth,decoderWidth])
else:
    net = NoLatentNet(inputDim)

# Initialize optimizer
opt = optim.Adam(net.parameters(), lr=learningRate)
lossFunc = torch.nn.functional.smooth_l1_loss # Smooth L1 loss is equivalent to L2 in our case, because our data is bounded

# Timer
startTime=time.perf_counter()

# Write header line of output
with open(outputFile, 'a') as outFile:
    outFile.write("# Epoch   training loss   validation loss\n")

# Training loop
for epoch in range(numEpochs+1):

    # Output every 1000th epoch
    if epoch%1000 == 0:
        net.eval()
        with torch.no_grad():
            # Compute current loss
            validLoss = 0.
            for batch, labels in test_loader:
                pred, latent = net(batch)
                validLoss += lossFunc(pred, labels) * labels.shape[0]
            trainLoss = sum(lossFunc(net(batch)[0], labels) * labels.shape[0] for batch, labels in train_loader)
            with open(outputFile, 'a') as outFile:
                outFile.write('{} {} {}\n'.format(epoch, trainLoss/(trainFrac*len(data)), validLoss/((1-trainFrac)*len(data))))

            if validLoss < minValidLoss:
                # Save net at minimal test loss
                torch.save(net, outputFile+"_min_loss.net.pkl")

                # Save values of latent variables at minimal test loss
                openType = 'w'
                for batch, labels in train_loader:
                    _, latent = net(batch)
                    with open(outputFileLatent, openType) as f:
                        np.savetxt(f, latent)
                    openType = 'a'
                openType = 'a'
                for batch, labels in test_loader:
                    _, latent = net(batch)
                    with open(outputFileLatent, openType) as f:
                        np.savetxt(f, latent)

            # Print timing
            print('# Time per epoch: {}'.format((time.perf_counter()-startTime) / 1000))
            startTime = time.perf_counter()

    net.train()
    for batch, labels in train_loader:
        # Training step
        batch = batch.view(len(batch), inputDim)

        pred, latent = net(batch)
        loss = lossFunc(pred, labels)

        loss.backward()
        opt.step()
        opt.zero_grad()

# Final output
net.eval()
torch.save(net, outputFile+"_final.net.pkl")
with torch.no_grad():
    validLoss = sum(lossFunc(net(batch)[0], labels) * labels.shape[0] for batch, labels in test_loader)
    trainLoss = sum(lossFunc(net(batch)[0], labels) * labels.shape[0] for batch, labels in train_loader)
    with open(outputFile, 'a') as outFile:
        outFile.write('{} {} {}\n'.format(epoch, trainLoss/(trainFrac*len(data)), validLoss/((1-trainFrac)*len(data))))
    print("# Final loss ( training   validation )")
    print('{} {} {}'.format(numEpochs-1, trainLoss/(trainFrac*len(data)), validLoss/((1-trainFrac)*len(data))))
