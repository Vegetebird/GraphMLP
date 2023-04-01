import os
import glob
import torch
import random
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from common.utils import *
from common.camera import *
import common.eval_cal as eval_cal
from common.arguments import parse_args

from common.load_data_hm36 import Fusion
from common.load_data_3dhp import Fusion_3dhp
from common.h36m_dataset import Human36mDataset
from common.mpi_inf_3dhp_dataset import Mpi_inf_3dhp_Dataset

from model.block.refine import post_refine, refine_model
from model.graphmlp import Model

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def train(dataloader, model, model_refine, optimizer, epoch):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('train', [input_2D, input_2D_GT, gt_3D, batch_cam])

        output_3D = model(input_2D)

        out_target = gt_3D.clone()
        out_target[:, :, args.root_joint] = 0
        out_target = out_target[:, args.pad].unsqueeze(1)

        if args.refine:
            model_refine.train()
            output_3D = refine_model(model_refine, output_3D, input_2D, gt_3D, batch_cam, args.pad, args.root_joint)
            loss = eval_cal.mpjpe(output_3D, out_target)
        else:
            loss = eval_cal.mpjpe(output_3D, out_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

    return loss_all['loss'].avg


def test(actions, dataloader, model, model_refine):
    model.eval()

    action_error = define_error_list(actions)

    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, gt_3D, batch_cam])

        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, args.joints_left + args.joints_right, :] = output_3D_flip[:, :, args.joints_right + args.joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        out_target = gt_3D.clone()
        out_target = out_target[:, args.pad].unsqueeze(1)

        if args.refine:
            model_refine.eval()
            output_3D = refine_model(model_refine, output_3D, input_2D[:, 0], gt_3D, batch_cam, args.pad, args.root_joint)

        output_3D[:, :, args.root_joint] = 0
        out_target[:, :, args.root_joint] = 0

        action_error = eval_cal.test_calculation(output_3D, out_target, action, action_error, args.dataset, subject)
    
    p1, p2, pck, auc = print_error(args.dataset, action_error, args.train)

    return p1, p2, pck, auc

if __name__ == '__main__':
    seed = 1

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'h36m':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Human36mDataset(dataset_path, args)
        actions = define_actions(args.actions)

        if args.train:
            train_data = Fusion(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)
    elif args.dataset == '3dhp':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Mpi_inf_3dhp_Dataset(dataset_path, args)
        actions = define_actions_3dhp(args.actions, 0)

        if args.train:
            train_data = Fusion_3dhp(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion_3dhp(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)

    model = Model(args).cuda()
    model_refine = post_refine(args).cuda()

    if args.previous_dir != '':
        Load_model(args, model, model_refine)

    lr = args.lr
    all_param = []
    all_param += list(model.parameters())

    if args.refine:
        all_param += list(model_refine.parameters())

    optimizer = optim.Adam(all_param, lr=lr, amsgrad=True)
    
    ##--------------------------------epoch-------------------------------- ##
    best_epoch = 0
    loss_epochs = []
    mpjpes = []

    for epoch in range(1, args.nepoch + 1):
        ## train
        if args.train: 
            loss = train(train_dataloader, model, model_refine, optimizer, epoch)
            loss_epochs.append(loss * 1000)

        ## test
        with torch.no_grad():
            p1, p2, pck, auc = test(actions, test_dataloader, model, model_refine)
            mpjpes.append(p1)

        ## save the best model
        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')

            if args.refine:
                args.previous_refine_name = save_model(args, epoch, p1, model_refine, 'refine')

            args.previous_best = p1

        ## print
        if args.train:
            logging.info('epoch: %d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('%d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            
            ## adjust lr
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay 
        else:
            if args.dataset == 'h36m':
                print('p1: %.2f, p2: %.2f' % (p1, p2))
            elif args.dataset == '3dhp':
                print('pck: %.2f, auc: %.2f, p1: %.2f, p2: %.2f' % (pck, auc, p1, p2))
            break

        ## training curves
        if epoch == 1:
            start_epoch = 3
                
        if args.train and epoch > start_epoch:
            plt.figure()
            epoch_x = np.arange(start_epoch+1, len(loss_epochs)+1)
            plt.plot(epoch_x, loss_epochs[start_epoch:], '.-', color='C0')
            plt.plot(epoch_x, mpjpes[start_epoch:], '.-', color='C1')
            plt.legend(['Loss', 'Test'])
            plt.ylabel('MPJPE')
            plt.xlabel('Epoch')
            plt.xlim((start_epoch+1, len(loss_epochs)+1))
            plt.savefig(os.path.join(args.checkpoint, 'loss.png'))
            plt.close()



