import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_cls import Model, transformer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from visualdl import LogWriter
from util import IOStream, cal_loss
from modelnet40_dataset import ModelNet40


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)

    if args.mode == 'train':
        if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
            os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

    os.system('cp model_cls.py checkpoints' + '/' + args.exp_name + '/' + 'model.py')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py')
    os.system('cp modelnet40_dataset.py checkpoints' + '/' + args.exp_name + '/' + 'data.py')
    os.system('cp main_m40.py checkpoints' + '/' + args.exp_name + '/' + 'main.py')


def train(args, io, num_class=40):
    train_loader = DataLoader(ModelNet40(args.num_points, partition='train'),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(args.num_points, partition='test'),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    # Try to load models
    model = Model(args, transformer, num_class).to(device)

    # if use multiple GPUs
    if args.use_gpus:
        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    best_test_acc = 0.0
    best_test_bal_acc = 0.0

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
            io.cprint('[Train %d, loss: %.6f, train acc: %.3f, balanced train acc: %.3f]' % (epoch,
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

            test_pred = []
            test_true = []
            model.eval()
            with torch.no_grad():
                for batch_data in tqdm(test_loader, total=len(test_loader)):
                    data, label = batch_data
                    data, label = data.to(device), label.to(device).squeeze()

                    logits = model(data)
                    preds = logits.max(dim=1)[1]
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)

            test_acc = accuracy_score(test_true, test_pred)
            avg_per_class_acc = balanced_accuracy_score(test_true, test_pred)

            if avg_per_class_acc >= best_test_bal_acc:
                best_test_bal_acc = avg_per_class_acc
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)

            io.cprint('[Test %d, test acc: %.3f, test avg acc: %.3f]' % (epoch, test_acc, avg_per_class_acc))

        # once the epochs are completed
        io.cprint('Best test acc:: %.3f | Best test avg acc:: %.3f' % (best_test_acc, best_test_bal_acc))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='m40_cls', metavar='N',
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
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'],
                        help='Scheduler to use: [cos, step]')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--use_gpus', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--use_norm', type=bool, default=False,
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
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    train(args, io)
