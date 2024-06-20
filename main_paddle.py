import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np

# import warnings
warnings.filterwarnings('ignore')
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.optimizer.lr import MultiStepDecay
paddle.device.cuda.synchronize()

from sklearn import metrics
from cgcnn.data_paddle import CIFData
from cgcnn.data_paddle import collate_pool, get_train_val_test_loader
from cgcnn.model_paddle import CrystalGraphConvNet


parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr_milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.01, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

# args = parser.parse_args(['/home/data_cy/cgcnn-master/root_dir'])
args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and paddle.device.is_compiled_with_cuda()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error

    # load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(paddle.zeros([2]))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=True if args.task ==
                                                       'classification' else False)
    if args.cuda:
        model.to(device=paddle.CUDAPlace(0))

    # Learning rate scheduler
    scheduler = optim.lr.MultiStepDecay(learning_rate=args.lr, milestones=args.lr_milestones, gamma=0.1)
    
    # Define loss function and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.Momentum(parameters=model.parameters(), learning_rate=scheduler,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')


    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = paddle.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.set_state_dict(checkpoint['state_dict'])
            optimizer.set_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = paddle.load('model_best.pth.tar')
    model.set_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

  
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader()):
       
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (
                paddle.to_tensor(input[0], place=paddle.CUDAPlace(0), dtype='float32'),
                paddle.to_tensor(input[1], place=paddle.CUDAPlace(0), dtype='float32'),
                paddle.to_tensor(input[2], place=paddle.CUDAPlace(0)),
                [paddle.to_tensor(crys_idx, place=paddle.CUDAPlace(0)) for crys_idx in input[3]]
            )
        else:
            input_var = (
                paddle.to_tensor(input[0], dtype='float32'),
                paddle.to_tensor(input[1], dtype='float32'),
                paddle.to_tensor(input[2]),
                input[3]
            )

       
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = paddle.to_tensor(target.reshape([-1]), dtype='int64')
        if args.cuda:
            target_var = paddle.to_tensor(target_normed, place=paddle.CUDAPlace(0))
        else:
            target_var = paddle.to_tensor(target_normed)

        
        output = model(*input_var)
        loss = criterion(output, target_var)

        
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.cpu()), target)
            losses.update(loss.cpu().numpy(), target.shape[0])
            mae_errors.update(mae_error, target.shape[0])
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(output.cpu(), target)
            losses.update(loss.cpu().numpy().item(), target.shape[0])
            accuracies.update(accuracy, target.shape[0])
            precisions.update(precision, target.shape[0])
            recalls.update(recall, target.shape[0])
            fscores.update(fscore, target.shape[0])
            auc_scores.update(auc_score, target.shape[0])

        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                      'MAE {mae_val:.3f} ({mae_avg:.3f})'.format(
                          epoch, i, len(train_loader), 
                          batch_time=batch_time, data_time=data_time, 
                          loss_val=losses.val.item(), loss_avg=losses.avg.item(),
                          mae_val=mae_errors.val.item(), mae_avg=mae_errors.avg.item()))

            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                          epoch, i, len(train_loader), 
                          batch_time=batch_time, data_time=data_time, 
                          loss=losses, accu=accuracies, 
                          prec=precisions, recall=recalls, 
                          f1=fscores, auc=auc_scores))

def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with paddle.no_grad():
                input_var = (
                            paddle.to_tensor(input[0], place=paddle.CUDAPlace(0), dtype='float32'),
                            paddle.to_tensor(input[1], place=paddle.CUDAPlace(0), dtype='float32'),
                            paddle.to_tensor(input[2], place=paddle.CUDAPlace(0)),
                            [paddle.to_tensor(crys_idx, place=paddle.CUDAPlace(0)) for crys_idx in input[3]])
        else:
            with paddle.no_grad():
                input_var = (paddle.to_tensor(input[0], dtype='float32'),
                             paddle.to_tensor(input[1], dtype='float32'),
                             input[2],
                             input[3])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = paddle.to_tensor(target.view(-1),dtype='int64')
        if args.cuda:
            with paddle.no_grad():
                target_var = paddle.to_tensor(target_normed, place=paddle.CUDAPlace(0))
        else:
            with paddle.no_grad():
                target_var = paddle.to_tensor(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.shape[0])
            mae_errors.update(mae_error, target.shape[0])
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds +=  test_pred.reshape([-1]).numpy().tolist()
                test_targets += test_target.reshape([-1]).numpy().tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.shape[0])
            accuracies.update(accuracy, target.shape[0])
            precisions.update(precision, target.shape[0])
            recalls.update(recall, target.shape[0])
            fscores.update(fscore, target.shape[0])
            auc_scores.update(auc_score, target.shape[0])
            if test:
                test_pred = paddle.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.reshape([-1]).numpy().tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'.format(i, len(val_loader)),
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time),
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses),
                      'MAE {mae_val:.3f} ({mae_avg:.3f})'.format(mae_val=mae_errors.val.item(), mae_avg=mae_errors.avg.item()))
                   
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors_avg:.3f}'.format(star=star_label,
                                                        mae_errors_avg=mae_errors.avg.item()))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = paddle.mean(tensor)
        self.std = paddle.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return paddle.mean(paddle.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    paddle.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
