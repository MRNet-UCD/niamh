import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader import MRDataset
import model
import pandas as pd
from sklearn import metrics


def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=100):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []
    a = np.zeros([1, 1])
    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float()).squeeze(0)



        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, valid_loader, epoch, num_epochs, writer, current_lr, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []
    a = np.zeros([1, 1])
    for i, (image, label, weight) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float()).squeeze(0)

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if ((i % log_every == 0) & (i > 0)) | (i == 281):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    vle=val_loss_epoch
    vae = val_auc_epoch

    y_trues = []
    y_preds = []
    losses = []
    a = np.zeros([1, 1])
    for i, (image, label, weight) in enumerate(valid_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float()).squeeze(0)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5


        if ((i % log_every == 0) & (i > 0)) | (i == 119):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | val auc : {4} | lr : {5}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(valid_loader),
                      np.round(auc, 4),
                      current_lr
                  )
                  )


    return vle, vae, auc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run(args):

    indexes = list(range(0,1130))
    random.seed(26)
    random.shuffle(indexes)

    for fold in range(0,8):

        if fold ==0:
            train_ind = indexes[0:141]+ indexes[282:]
            valid_ind = indexes[141:282]
        elif fold ==1:
            train_ind = indexes[0:282]+indexes[423:]
            valid_ind = indexes[282:423]
        elif fold ==2:
            train_ind = indexes[0:564]+indexes[705:]
            valid_ind = indexes[564:705]
        elif fold ==3:
            train_ind = indexes[:705]+indexes[846:]
            valid_ind = indexes[705:846]
        elif fold ==4:
            train_ind = indexes[:846]+ indexes[987:]
            valid_ind = indexes[846:987]
        elif fold ==5:
            train_ind = indexes[:987]
            valid_ind = indexes[987:]
        elif fold ==6:
            train_ind = indexes[141:]
            valid_ind = indexes[0:141]
        elif fold ==7:
            train_ind = indexes[0:423]+indexes[568:]
            valid_ind = indexes[423:568]



        log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
        if args.flush_history == 1:
            objects = os.listdir(log_root_folder)
            for f in objects:
                if os.path.isdir(log_root_folder + f):
                    shutil.rmtree(log_root_folder + f)

        now = datetime.now()
        logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(logdir)

        writer = SummaryWriter(logdir)

        augmentor = Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            RandomRotate(25),
            RandomTranslate([0.11, 0.11]),
            RandomFlip(),
          #  transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
        ])
        mrnet = model.MRNet()

        if torch.cuda.is_available():
            mrnet = mrnet.cuda()

        optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0.1)

        if args.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=2, factor=.3, threshold=1e-4, verbose=True)
        elif args.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=args.gamma)

        best_val_loss = float('inf')
        best_val_auc = float(0)

        num_epochs = args.epochs
        iteration_change_loss = 0
        patience = args.patience
        log_every = args.log_every

        t_start_training = time.time()
        train_dataset = MRDataset(train_ind, '/content/data/', args.task,
                              args.plane, valid= False, transform=augmentor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

        validation_dataset = MRDataset(valid_ind,
            '/content/data/', args.task, args.plane, valid = False, transform = None)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)

        valid_dataset = MRDataset([0],
            '/content/data/', args.task, args.plane, valid = True, transform = None)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)


        for epoch in range(num_epochs):
            current_lr = get_lr(optimizer)

            t_start = time.time()

            train_loss, train_auc = train_model(
                mrnet, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every)
            val_loss, val_auc, test_auc = evaluate_model(
                mrnet, validation_loader,valid_loader, epoch, num_epochs, writer, current_lr)

            if args.lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            elif args.lr_scheduler == 'step':
                scheduler.step()

            t_end = time.time()
            delta = t_end - t_start

            print("fold : {0} | train loss : {1} | train auc {2} | val loss {3} | val auc {4} | elapsed time {5} s".format(
                fold, train_loss, train_auc, val_loss, val_auc, delta))

            iteration_change_loss += 1
            print('-' * 30)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                if bool(args.save_model):
                    file_name = f'model_fold{fold}_{args.prefix_name}_{args.task}_{args.plane}_test_auc_{test_auc:0.4f}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
                    for f in os.listdir('./models/'):
                        if (args.task in f) and (args.prefix_name in f) and ('fold'+str(fold) in f) :
                            os.remove(f'./models/{f}')
                    torch.save(mrnet, f'./models/{file_name}')



            if val_loss < best_val_loss:
                best_val_loss = val_loss
                iteration_change_loss = 0

            if iteration_change_loss == patience:
                print('Early stopping after {0} iterations without the decrease of the val loss'.
                      format(iteration_change_loss))
                break

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
