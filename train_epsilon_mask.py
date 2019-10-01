import torch
from torchvision import datasets as vdatasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from glow.glow_mask import Glow
import numpy as np
import skimage.io as sio
import matplotlib.pyplot as plt
import os
import json
import argparse
import torch.optim as optim
import cv2
import torchvision
import datasets as datasets


parser = argparse.ArgumentParser(description='train glow network')
parser.add_argument('-dataset',type=str,help='the dataset to train the model on', default='celeba')
parser.add_argument('-K',type=int,help='no. of steps of flow',default=48)
parser.add_argument('-L',type=int,help='no. of time squeezing is performed',default=4)
parser.add_argument('-coupling',type=str,help='type of coupling layer to use',default='affine')
parser.add_argument('-last_zeros',type=bool,help='whether to initialize last layer ot NN with zeros',default=True)
parser.add_argument('-batchsize',type=int,help='batch size for training',default=64)
parser.add_argument('-size',type=int,help='images will be resized to this dimension',default=64)
parser.add_argument('-lr',type=float,help='learning rate for training',default=1e-4)
parser.add_argument('-n_bits_x',type=int,help='requantization of training images',default=5)
parser.add_argument('-epochs',type=int,help='epochs to train for',default=1000)
parser.add_argument('-warmup_iter',type=int,help='no. of warmup iterations',default=10000)
parser.add_argument('-sample_freq',type=int,help='sample after every save_freq',default=50)
parser.add_argument('-save_freq',type=int,help='save after every save_freq',default=1000)
parser.add_argument('-device',type=str,help='whether to use',default="cuda")
args = parser.parse_args()


save_path   = "/home/data1/meng/ip/%s/glow"%args.dataset
training_folder = "./data/%s_preprocessed/train"%args.dataset
# setting up configs as json
config_path = save_path+"/configs.json"
configs     = {"K":args.K,
               "L":args.L,
               "coupling":args.coupling,
               "last_zeros":args.last_zeros,
               "batchsize":args.batchsize,
               "size":args.size,
               "lr": args.lr,
               "n_bits_x":args.n_bits_x,
               "warmup_iter":args.warmup_iter}


print("loading previous model and saved configs to resume training ...")
with open(config_path, 'r') as f:
    configs = json.load(f)

glow = Glow((3,configs["size"],configs["size"]),
            K=configs["K"],L=configs["L"],
            coupling=configs["coupling"],
            n_bits_x=configs["n_bits_x"],
            nn_init_last_zeros=configs["last_zeros"],
            device=args.device)
glow.load_state_dict(torch.load(save_path+"/glowmodel.pt"), strict=False)
print("pre-trained model and configs loaded successfully")
glow.set_actnorm_init()
print("actnorm initialization flag set to True to avoid data dependant re-initialization")
glow = glow.cuda()


def destroy(pic, x, y):
    size = 16
    # B G R
    pic[:, 0, x:x + 1, y:y + size] = -0.5
    pic[:, 1, x:x + 1, y:y + size] = -0.5
    pic[:, 2, x:x + 1, y:y + size] = 0.5

    pic[:, 0, x:x + size, y:y + 1] = -0.5
    pic[:, 1, x:x + size, y:y + 1] = -0.5
    pic[:, 2, x:x + size, y:y + 1] = 0.5

    pic[:, 0, x + size:x + size + 1, y:y + size] = -0.5
    pic[:, 1, x + size:x + size + 1, y:y + size] = -0.5
    pic[:, 2, x + size:x + size + 1, y:y + size] = 0.5

    pic[:, 0, x:x + size, y + size:y + size + 1] = -0.5
    pic[:, 1, x:x + size, y + size:y + size + 1] = -0.5
    pic[:, 2, x:x + size, y + size:y + size + 1] = 0.5
    # pic[:, 0, x:x + size, y:y + size] = 0
    # pic[:, 1, x:x + size, y:y + size] = 0
    # pic[:, 2, x:x + size, y:y + size] = 1
    return pic


def show_mask(x):
    ret = torch.zeros(x.size(0), x.size(1), 3)
    ret_min = x.min()
    ret_max = x.max()
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            if x[i, j, :] > 0:
                ret[i, j, 0], ret[i, j, 1], ret[i, j, 2] = 0, 0, x[i, j, 0].detach().cpu() / ret_max
            elif x[i, j, :] < 0:
                ret[i, j, 0], ret[i, j, 1], ret[i, j, 2] = x[i, j, 0].detach().cpu() / ret_min, 0, 0
            else:
                ret[i, j, 0], ret[i, j, 1], ret[i, j, 2] = 0, 0, 0
    return ret


def plot_z_stats(data, filename):
    fig, ax = plt.subplots()
    ax.hist(data, density=True, bins=40, alpha=0.4, linestyle="-", linewidth=1.5,
            edgecolor="black", label="untrained")
    ax.legend()
    plt.xlabel("z\'s dimemsion")
    plt.ylabel("value")
    plt.savefig(filename)
    plt.close()


def reduce_bits(x):
    nbits = 8
    x = x * 255
    x = torch.floor(x / 2**(8 - nbits))
    x = x / 2**nbits - 0.5
    return x
# setting up dataloader
print("setting up dataloader for the training data")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize,
#                                             drop_last=True, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CelebA5bit(train=False, transform=transforms.Compose([reduce_bits]) ),
    batch_size=args.batchsize, shuffle=False, num_workers=2)


opt          = torch.optim.Adam(glow.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",
                                                          factor=0.5,
                                                          patience=1000,
                                                          verbose=True,
                                                          min_lr=1e-8)



def update_lr(optimizer, itr, lr_1, lr_2):
    # print(optimizer.param_groups)
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:
            param_group["lr"] = lr_1
        elif i == 1:
            param_group["lr"] = lr_2


def train_epsilon(model):
    for i, (inputs, targets) in enumerate(test_loader):
        if i == 2:
            x = inputs[1].unsqueeze(dim=0)
        elif i > 2:
            break

    x = x.cuda()
    x_orig = x.clone()

    model.eval()
    nll, logdet, logpz, z_mu, z_std = glow.nll_loss(x_orig)
    logpx = -(logpz + logdet)
    print("undestroyed logpx", logpx)

    x = destroy(x, 16, 32)
    x = x.cuda()
    model.eval()
    nll, logdet, logpz, z_mu, z_std = glow.nll_loss(x)
    logpx = -(logpz + logdet)
    print("untrained logpx", logpx)

    model.train()
    iters = 100
    alpha_ = 10
    lambda_ = 1
    Losses = []
    lr = 1e-5


    optimizer = optim.SGD([
        {'params': model.epsilon, 'lr': lr},
        {'params': model.mask, 'lr': 0.15}
    ], lr=0)

    for i in range(iters):
        lr_1 = 5 + i / iters
        lr_1 = 10 ** (-1 * lr_1)
        lr_2 = 1.5 + i / iters
        lr_2 = 10 ** (-1 * lr_2)
        update_lr(optimizer, i, lr_1, lr_2)

        nll, logdet, logpz, z_mu, z_std = glow.nll_loss(x.clone(), epsilon=True)

        optimizer.zero_grad()

        logpx = -(logpz + logdet)
        tanh = torch.nn.Tanh()
        loss_1 = (tanh(model.epsilon) * model.mask).abs().sum() * alpha_
        loss_2 = -1 * logpx  # * logpx * lambda_
        Loss = loss_1 + loss_2
        Loss.backward()

        optimizer.step()
        print(i + 1,
              Loss.cpu().detach(),
              loss_1.cpu().detach(),
              loss_2)

        Losses.append(Loss.item())

    model.eval()

    nll, logdet, logpz, z_mu, z_std = glow.nll_loss(x.clone(), epsilon=True)
    logpx = -(logpz + logdet)
    print("trained logpx", logpx)

    epsilon_trained = model.epsilon
    mask_1, mask_2, mask_3 = epsilon_trained[:, 0].permute(1, 2, 0), \
                             epsilon_trained[:, 1].permute(1, 2, 0), \
                             epsilon_trained[:, 2].permute(1, 2, 0)
    mask_1 = show_mask(mask_1).detach().numpy()
    mask_2 = show_mask(mask_2).detach().numpy()
    mask_3 = show_mask(mask_3).detach().numpy()

    mask = np.concatenate((mask_1, mask_2, mask_3), axis=1)
    cv2.imwrite("./mask_imgs/epsilon.jpg", np.uint8(mask * 255))

    MASK = model.mask.clone().cpu().detach()
    MASK = MASK[0, 0]
    MASK = (MASK - MASK.min()) / (MASK.max() - MASK.min() + 1e-5)
    heatmap = cv2.applyColorMap(np.uint8(255 * MASK), cv2.COLORMAP_JET)
    cv2.imwrite("./mask_imgs/MASK.jpg", heatmap)

    x_trained = x.clone() + tanh(model.epsilon) * model.mask
    x_trained = x_trained.clamp(-0.5, 0.5)
    x_comp = torch.cat((x[0], x_trained[0]), dim=2)
    x_comp = glow.postprocess(x_comp)
    x_comp = x_comp.data.cpu().detach()
    unloader = torchvision.transforms.ToPILImage()
    x_comp = unloader(x_comp)
    x_comp = np.array(x_comp)
    cv2.imwrite("./mask_imgs/img_trained.jpg", x_comp)

    ax = plt.subplot()
    ax.plot(Losses, label="Loss")
    plt.xlabel("train iterations")
    plt.ylabel("Loss")
    plt.title("Losses per iteration")
    plt.savefig("./mask_imgs/Losses.jpg")
    plt.close()

    # return nll, -logdet.mean().item(),-logpz.mean().item(), z_.mean().item(), z_.std().item()


train_epsilon(glow)