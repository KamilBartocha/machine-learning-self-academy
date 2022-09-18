import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os.path
from datetime import datetime
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

from cnn_utils import qilv_measure
from cnn_model import DiffusionSRCNN
from cnn_dataload import DiffusionDataset

plt.rcParams.update({'font.size': 16})
plt.rc('axes', titlesize=16, labelsize=16)
plt.rc('font', family='Arial')

# %% Basic configuration


# check if CUDA is available; then use it if it is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# set the seed for reproducible results
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# set network configuration
configuration = {'channels': 1, 'n1': 128, 'n2': 64,
                 'kernel_size1': 9, 'padding1': 4,
                 'kernel_size2': 1, 'padding2': 0,
                 'kernel_size3': 5, 'padding3': 2,
                 'loss_function_mssim': False}

# network configuration
epochs = 50
batch_size = 1

# training models to be saved
trained_general_path = './trained_networks_testowanie/'

if not os.path.exists(trained_general_path):
    os.mkdir(trained_general_path)

# create network directory
trained_network_path = trained_general_path + datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(trained_network_path)

# training data sets
training_data_path_in = './training_iso/data2d_interp_iso_training/'  # input data
training_data_path_out = './training_iso/data2d_iso_training/'  # output data (reference)

testing_data_path_in = './testing_iso/data2d_interp_iso_testing1/'  # input data
testing_data_path_out = './testing_iso/data2d_iso_testing1/'  # output data (reference)

# create instances of the DiffusionDataset classes
trainset = DiffusionDataset(training_data_path_in, training_data_path_out)
testset = DiffusionDataset(training_data_path_in, training_data_path_out)

# data loaders
training_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testing_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

# save network configuration to a file
n1 = configuration['n1']
n2 = configuration['n2']
kernel_size1 = configuration['kernel_size1']
kernel_size2 = configuration['kernel_size2']
kernel_size3 = configuration['kernel_size3']
loss_function_mssim = configuration['loss_function_mssim']
loss_function_mssim_txt = ''

if configuration['loss_function_mssim']:
    loss_function_mssim_txt = '+MSSIM'

ff = open(f'{trained_network_path}/configuration.txt', 'w')
ff.write(f'Epochs: {epochs}\n')
ff.write(f'Batch size: {batch_size}\n')
ff.write(f'Loss function: MSE{loss_function_mssim_txt}\n')
ff.write('\n')
ff.write(f'n1: {n1}\n')
ff.write(f'n2: {n2}\n')
ff.write(f'kernel_size1: {kernel_size1}\n')
ff.write(f'kernel_size2: {kernel_size2}\n')
ff.write(f'kernel_size3: {kernel_size3}\n')
ff.close()

# %% DiffusionSRCNN network configuration

# create an instance of the network
network = DiffusionSRCNN(configuration).to(device)
network = network.double()

# loss function
# loss_function = nn.MSELoss()
loss_function = nn.L1Loss()

optimizer = optim.Adam(
    [
        {"params": network.conv1.parameters(), "lr": 0.0001},
        {"params": network.conv2.parameters(), "lr": 0.0001},
        {"params": network.conv3.parameters(), "lr": 0.00001},
    ], lr=0.00001,
)

# %% Training procedure

# training/validation log
log_training_loss = np.zeros(epochs)
log_testing_psnr = np.zeros(epochs)
log_testing_mse = np.zeros(epochs)
log_testing_mssim = np.zeros(epochs)
log_testing_qilv = np.zeros(epochs)

f = open(f'{trained_network_path}/log.txt', 'w')

# testing loss for the best model
testing_loss_best = np.Inf
log_best = False
log_best_iter = 0

# iterate over training epochs
for ii in range(0, epochs):

    # 1. Training phase
    training_loss = 0

    # iterate over all data sets in the training dataset
    for iter, data in enumerate(training_loader):

        # retrieve the data
        input_data = data[0].to(device, dtype=torch.double)
        output_data = data[1].to(device, dtype=torch.double)

        # set gradients to zero
        optimizer.zero_grad()

        # forward pass
        output_data_network = network(input_data)

        # loss function
        if configuration['loss_function_mssim']:
            loss = loss_function(output_data_network, output_data) * (
                        1 - ssim(output_data_network[0, 0, :, :].cpu().detach().numpy(),
                                 output_data[0, 0, :, :].cpu().detach().numpy()))
        else:
            loss = loss_function(output_data_network, output_data)

        loss.backward()
        training_loss += loss.item()

        # update the parameters
        optimizer.step()

    # ---------------------------------------------------------------------
    # 2. Testing phase
    testing_psnr = 0
    testing_qilv = 0
    testing_mssim = 0
    testing_mse = 0

    with torch.no_grad():
        for data in testing_loader:
            # retrieve the data
            input_data = data[0].to(device, dtype=torch.double)
            output_data = data[1].to(device, dtype=torch.double)

            # forward pass
            output_data_network = network(input_data)

            # calculate loss function
            loss = loss_function(output_data_network, output_data)

            # calculate PSNR
            testing_psnr += 10 * np.log10(input_data.cpu().detach().numpy().max() / loss.item())
            # testing_psnr += 10*np.log10(1/loss.item())

            # calculate MSSIM
            testing_mssim += ssim(data[0][0, 0, :, :].cpu().detach().numpy(),
                                  output_data_network[0, 0, :, :].cpu().detach().numpy())

            # calculate MSE
            testing_mse += mean_squared_error(data[0][0, 0, :, :].cpu().detach().numpy(),
                                              output_data_network[0, 0, :, :].cpu().detach().numpy())

            # calculate QILV
            testing_qilv += qilv_measure(data[0][0, 0, :, :].cpu().detach().numpy(),
                                         output_data_network[0, 0, :, :].cpu().detach().numpy())

            # 3. model saving

    # save current model
    torch.save(network, f'{trained_network_path}/model_epoch_{ii}.pth')

    # save best model
    if loss.item() < testing_loss_best:
        torch.save(network, f'{trained_network_path}/model_best.pth')

        # update loss for the best model
        testing_loss_best = np.copy(loss.item())

        # logging the best model
        log_best_model_txt = f'-- Best model at epoch={ii}'
        log_best = True
        log_best_iter = np.copy(ii)

    else:
        log_best = False

    # 4. log statistics
    log_training_loss[ii] = training_loss / len(training_loader)
    log_testing_qilv[ii] = testing_qilv / len(testing_loader)
    log_testing_psnr[ii] = testing_psnr / len(testing_loader)
    log_testing_mssim[ii] = testing_mssim / len(testing_loader)
    log_testing_mse[ii] = testing_mse / len(testing_loader)

    log_training_txt = f'\nEpoch={ii}, training loss={log_training_loss[ii]}'
    log_testing_txt = f'PSNR={log_testing_psnr[ii]}\nMSSIM={log_testing_mssim[ii]}\nQILV={log_testing_qilv[ii]}\nMSE={log_testing_mse[ii]}'

    # print to console
    print(log_training_txt)
    print(log_testing_txt)

    if log_best == True:
        print(log_best_model_txt)

        # print to a file
    f.write(log_training_txt)
    f.write('\n')
    f.write(log_testing_txt)
    f.write('\n')

    if log_best == True:
        f.write(log_best_model_txt)

    f.write('\n')
# close log file
f.close()

# save testing loss, PSNR, etc..
training_stat = {'log_training_loss': log_training_loss,
                 'log_testing_psnr': log_testing_psnr,
                 'log_testing_qilv': log_testing_qilv,
                 'log_testing_mssim': log_testing_mssim,
                 'log_testing_qilv': log_testing_qilv,
                 'epochs': epochs,
                 'testing_loss_best': testing_loss_best,
                 'log_best_iter': log_best_iter}

np.save(f'{trained_network_path}/training_stat.npy', training_stat)

# %% Plotting

plt.figure(1)
plt.subplot(221)
plt.plot(range(1, epochs + 1), log_training_loss)
plt.xlabel('Epoch number')
plt.ylabel('Loss function')
plt.grid(True, linestyle=':', color='k')
# plt.xticks(range(0, epochs, 2))

plt.subplot(222)
plt.plot(range(1, epochs + 1), log_testing_mse)
plt.xlabel('Epoch number')
plt.ylabel('MSE (testing)')
plt.grid(True, linestyle=':', color='k')
# plt.xticks(range(0, epochs, 2))

plt.subplot(223)
plt.plot(range(1, epochs + 1), log_testing_psnr)
plt.xlabel('Epoch number')
plt.ylabel('PSNR (testing)')
plt.grid(True, linestyle=':', color='k')
# plt.xticks(range(0, epochs, 2))


plt.subplot(224)
plt.plot(range(1, epochs + 1), log_testing_mssim)
plt.xlabel('Epoch number')
plt.ylabel('MSSIM (testing)')
plt.grid(True, linestyle=':', color='k')
# plt.xticks(range(0, epochs, 2))


