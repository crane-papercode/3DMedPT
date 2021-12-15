import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from intra3d import Intra3D
from model_cls import Model, transformer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
from visualdl import LogWriter
from util import IOStream, cal_loss


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)

    if args.mode == 'train':
        if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
            os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

    os.system('cp model_cls.py checkpoints' + '/' + args.exp_name + '/' + 'model.py')
    os.system('cp intra3d.py checkpoints' + '/' + args.exp_name + '/' + 'data.py')
    os.system('cp main_intra.py checkpoints' + '/' + args.exp_name + '/' + 'main.py')


def train(args, io, split, num_class=2):
    train_loader = DataLoader(
        Intra3D(train_mode='train', cls_state=True, npoints=args.num_points, data_aug=True, choice=split),
        num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        Intra3D(train_mode='test', cls_state=True, npoints=args.num_points, data_aug=False, choice=split),
        num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    model = Model(args, transformer, num_class).to(device)
    # if use multiple GPUs
    if args.use_gpus:
        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    best_test_acc = 0.0
    best_test_bal_acc = 0.0
    best_V_acc = 0.0
    best_A_acc = 0.0
    best_f1_value = 0.0

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    io.cprint("The number of trainable parameters: %.6f(M)" % (count_parameters(model) / (1024 ** 2)))

    with LogWriter(logdir='checkpoints/%s/log/train' % args.exp_name) as writer:
        for epoch in range(args.epochs):
            train_loss = 0.0
            count = 0.0
            model.train()
            train_pred = []
            train_true = []
            for batch_data in tqdm(train_loader, total=len(train_loader)):
                data, label = batch_data
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.shape[0]
                # start training the model
                opt.zero_grad()
                logits = model(data)

                loss = cal_loss(logits, label)
                loss.backward()
                opt.step()

                preds = logits.max(dim=1)[1]
                train_true.append(label.detach().cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
                count += batch_size
                train_loss += loss.item() * batch_size

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            epoch_loss = train_loss * 1.0 / count
            train_acc = accuracy_score(train_true, train_pred)
            train_bal_acc = balanced_accuracy_score(train_true, train_pred)
            io.cprint('[Train %d, loss: %.6f, train acc: %.6f, balanced train acc: %.6f]' % (epoch,
                                                                                             epoch_loss,
                                                                                             train_acc,
                                                                                             train_bal_acc))
            if args.scheduler == 'cos':
                scheduler.step()
            elif args.scheduler == 'step':
                if opt.param_groups[0]['lr'] > 1e-5:
                    scheduler.step()
                if opt.param_groups[0]['lr'] < 1e-5:
                    for param_group in opt.param_groups:
                        param_group['lr'] = 1e-5

            # add to logger
            writer.add_scalar(tag='train_loss', step=epoch, value=epoch_loss)
            writer.add_scalar(tag='train_acc', step=epoch, value=train_acc)
            writer.add_scalar(tag='train_bal_acc', step=epoch, value=train_bal_acc)

            # validation
            best_V_acc, best_A_acc, best_f1_value, best_test_acc, best_test_bal_acc = val(test_loader, num_class,
                                                                                          model, device, epoch,
                                                                                          best_V_acc, best_A_acc,
                                                                                          best_f1_value, best_test_acc,
                                                                                          best_test_bal_acc, writer)

    # once the epochs are completed
    io.cprint('Split %d ------>> best f1 score: %.6f, best_V_acc: %.6f, best_A_acc: %.6f' % (split,
                                                                                             best_f1_value,
                                                                                             best_V_acc,
                                                                                             best_A_acc))


def val(test_loader, num_class, model, device, epoch, best_V_acc,
        best_A_acc, best_f1_value, best_test_acc, best_test_bal_acc, logger):
    test_pred = []
    test_true = []
    per_class_acc = np.zeros((num_class, 2))

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(test_loader, total=len(test_loader)):
            data, label = batch_data
            data, label = data.to(device), label.to(device).squeeze()

            logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            # per-class accuracy
            for cat in np.unique(label.cpu().numpy()):
                acc_num = preds[label == cat].eq(label[label == cat]).sum()
                cls_total_num = label[label == cat].size()[0]
                per_class_acc[cat, 0] += acc_num.item() / cls_total_num
                per_class_acc[cat, 1] += 1

    [V_acc, A_acc] = per_class_acc[:, 0] / per_class_acc[:, 1]
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    f1_value = f1_score(test_true, test_pred)
    test_acc = accuracy_score(test_true, test_pred)
    avg_per_class_acc = balanced_accuracy_score(test_true, test_pred)
    if V_acc > best_V_acc:
        best_V_acc = V_acc
    if A_acc > best_A_acc:
        best_A_acc = A_acc
    if f1_value >= best_f1_value:
        best_f1_value = f1_value
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_split_%d_f1.pth' % (args.exp_name, split))
    if test_acc >= best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_split_%d.pth' % (args.exp_name, split))
        if avg_per_class_acc >= best_test_bal_acc:
            best_test_bal_acc = avg_per_class_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_split_%d.pth' % (args.exp_name, split))

    io.cprint('[Test %d, test acc: %.6f, best V_acc: %.6f, best A_acc: %.6f, best F1: %.6f]' % (
        epoch, test_acc, best_V_acc, best_A_acc, best_f1_value))
    logger.add_scalar(tag='V_acc', step=epoch, value=V_acc)
    logger.add_scalar(tag='A_acc', step=epoch, value=A_acc)
    logger.add_scalar(tag='f1', step=epoch, value=f1_value)

    return best_V_acc, best_A_acc, best_f1_value, best_test_acc, best_test_bal_acc


def test(args, io, split, num_class=2):
    test_loader = DataLoader(
        Intra3D(train_mode='test', cls_state=True, npoints=args.num_points, data_aug=True, choice=split),
        num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    # load feature model
    model = Model(args, transformer, num_class).to(device)
    if args.use_gpus:
        model = nn.DataParallel(model)

    path = 'checkpoints/%s_train/models/model_split_%d_f1.pth' % (args.exp_name[:-5], split)
    model.load_state_dict(torch.load(path))

    model.eval()
    with torch.no_grad():
        # Initialize parameters
        test_pred = []
        test_true = []
        per_class_acc = np.zeros((2, 2))

        # counting throughput
        repetitions = 0
        total_time = 0
        for batch_data in tqdm(test_loader, total=len(test_loader)):
            data, label = batch_data
            data, label = data.to(device), label.to(device).squeeze()

            # initializing the timer
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            logits = model(data)

            # record the end time
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
            repetitions += 1

            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

            # per-class accuracy
            for cat in np.unique(label.cpu().numpy()):
                acc_num = preds[label == cat].eq(label[label == cat]).sum()
                cls_total_num = label[label == cat].size()[0]
                per_class_acc[cat, 0] += acc_num.item() / cls_total_num
                per_class_acc[cat, 1] += 1

        # calculate the throughput
        throughput = (repetitions * args.test_batch_size) / total_time
        io.cprint('Final throughput: %f' % throughput)

        [V_acc, A_acc] = per_class_acc[:, 0] / per_class_acc[:, 1]
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        f1_value = f1_score(test_true, test_pred)
        test_acc = accuracy_score(test_true, test_pred)
        avg_per_class_acc = balanced_accuracy_score(test_true, test_pred)
        io.cprint('[Test: test acc: %.6f, balanced test acc: %.6f]' % (test_acc, avg_per_class_acc))
        io.cprint('Split %d ------>> Test: f1 score: %.6f, V_acc: %.6f, A_acc: %.6f' % (split, f1_value, V_acc, A_acc))
    return f1_value, V_acc, A_acc


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='intra3d_cls', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train', metavar='N', choices=['train', 'test'],
                        help='model mode')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--use_gpus', type=bool, default=True,
                        help='evaluate the model')
    parser.add_argument('--use_norm', type=bool, default=True,
                        help='Whether to use norm')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--num_K', nargs='+', type=int,
                        help='list of num of neighbors')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--head', type=int, default=8, metavar='N',
                        help='Dimension of heads')
    parser.add_argument('--dim_k', type=int, default=32, metavar='N',
                        help='Dimension of key/query tensors')
    args = parser.parse_args()

    _init_()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = True

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.use_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        io.cprint('Using GPU ' + str(torch.cuda.current_device()))
    else:
        io.cprint('Using CPU')

    for split in range(5):
        if args.mode == 'train':
            train(args, io, split)
        elif args.mode == 'test':
            best_F1_list = []
            best_V_list = []
            best_A_list = []
            for _ in range(50):
                best_f1_value, best_V_acc, best_A_acc = test(args, io, split)
                best_F1_list.append(best_f1_value)
                best_V_list.append(best_V_acc)
                best_A_list.append(best_A_acc)

            io.cprint("Split %d --->>> Best f1 score: %.5f | Best V acc: %.5f | Best A acc: %.5f" % (split,
                                                                                                     max(best_F1_list),
                                                                                                     max(best_V_list),
                                                                                                     max(best_A_list)))
        else:
            print("Error")
            exit(-1)
    # print("The average of V is %.4f and A is %.4f" % (np.mean(best_V), np.mean(best_A)))
