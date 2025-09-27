import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
from losses import *
from models.preactresnet import *
from models.densenet import *

parser = argparse.ArgumentParser("Demo")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use-gpu', type=str, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--val-interval', type=int, default=2)

parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--model', type=str, default='preactresnet18', help='model architecture',
                    choices=['preactresnet18', 'densenet121'])
parser.add_argument('--train_bs', type=int, default=128, help='batch size for trainloader')
parser.add_argument('--test_bs', type=int, default=256, help='batch size for testloader')
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=0.1, help='base learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
parser.add_argument('--milestones', nargs='+', type=int, default=[50, 100])
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay")
parser.add_argument('--weight-decay', type=float, default=1e-4, help="l2 regularization")

parser.add_argument('--w_ce', type=float, default=1.0)
parser.add_argument('--w_occe', type=float, default=1.0,
                    help="gamma parameter from the paper. We propose a full grid search in range (0.01, 1).")


def get_model(num_classes):
    if args.model == 'preactresnet18':
        model = preactresnet18(num_classes=num_classes)
    elif args.model == 'densenet121':
        model = densenet121(num_classes=num_classes)
    else:
        raise ValueError("Model architecture not available.")

    if args.use_gpu:
        model.cuda(args.gpu)

    return model

def main():
    dataset = ClosedSetDataset('data', train=True, args=args)
    trainloader = dataset.get_loader()

    test_dataset = ClosedSetDataset('data', train=False, args=args)
    testloader = test_dataset.get_loader()

    exp_name = f'{args.model}_ce={args.w_ce}_occe={args.w_occe}_e={args.n_epochs}_s={args.seed}'
    out_dir = f'logs/{args.dataset}/{args.model}/' + exp_name
    writer = SummaryWriter(out_dir)

    occe_criterion = OCCELoss()
    ce_criterion = nn.CrossEntropyLoss()

    model = get_model(dataset.classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train(model=model, ce_criterion=ce_criterion, occe_criterion=occe_criterion, optimizer=optimizer,
          scheduler=scheduler, trainloader=trainloader, testloader=testloader, writer=writer,
          save=f'{out_dir}/checkpoint.pth')


def train(model, ce_criterion, occe_criterion, optimizer, scheduler, trainloader, testloader, writer, save):
    best_val_acc = float('-inf')
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        ce_losses, occe_losses, total_losses = AverageMeter(), AverageMeter(), AverageMeter()
        accuracy = AverageMeter()
        for data, labels in trainloader:
            if args.use_gpu:
                data, labels = data.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = model(data)

            loss_ce = ce_criterion(outputs, labels)
            loss_occe = occe_criterion(outputs, labels)
            total_loss = args.w_ce * loss_ce + args.w_occe * loss_occe

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch - 1) % args.val_interval == 0:
                with torch.no_grad():
                    _, pred = torch.max(outputs, 1)
                    accuracy.update((pred == labels).sum() / labels.size(0), labels.size(0))
                    ce_losses.update(loss_ce, labels.size(0))
                    occe_losses.update(loss_occe, labels.size(0))
                    total_losses.update(total_loss, labels.size(0))

        scheduler.step()
        if (epoch - 1) % args.val_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Train/Classifier_LR', current_lr, epoch)
            writer.add_scalar(f'Train/Accuracy', accuracy.avg, epoch)
            writer.add_scalar(f'Train/CE Loss', ce_losses.avg, epoch)
            writer.add_scalar(f'Train/OCCE Loss', occe_losses.avg, epoch)
            writer.add_scalar(f'Train/Total Loss', total_losses.avg, epoch)

            val_acc = evaluate(model, ce_criterion, occe_criterion, testloader, eval_idx=epoch, writer=writer)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save), exist_ok=True)
                torch.save(model.state_dict(), save)


def evaluate(model, ce_criterion, occe_criterion, testloader, eval_idx, writer, saved_model=None):
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
        if args.use_gpu:
            model.cuda(args.gpu)
    model.eval()
    total_losses, ce_losses, occe_losses = AverageMeter(), AverageMeter(), AverageMeter()
    accuracy = AverageMeter()
    with torch.no_grad():
        for data, labels in testloader:
            if args.use_gpu:
                data, labels = data.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = model(data)

            loss_ce = ce_criterion(outputs, labels)
            loss_occe = occe_criterion(outputs, labels)
            total_loss = args.w_ce * loss_ce + args.w_occe * loss_occe

            _, pred = torch.max(outputs, 1)
            accuracy.update((pred == labels).sum() / labels.size(0), labels.size(0))
            ce_losses.update(loss_ce, labels.size(0))
            occe_losses.update(loss_occe, labels.size(0))
            total_losses.update(total_loss, labels.size(0))

    writer.add_scalar(f'Val/Accuracy', accuracy.avg, eval_idx)
    writer.add_scalar(f'Val/CE Loss', ce_losses.avg, eval_idx)
    writer.add_scalar(f'Val/OCCE Loss', occe_losses.avg, eval_idx)
    writer.add_scalar(f'Val/Total Loss', total_losses.avg, eval_idx)

    return accuracy.avg


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    if args.use_gpu:
        args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
    else:
        print("Currently using CPU")

    main()
