import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import Models.AnyNet.utils.logger as logger
import torch.backends.cudnn as cudnn
import numpy as np
import Models.AnyNet.models.anynet as anynet
import tqdm

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--datatype', default='2015', help='data type')
parser.add_argument('--data_path', default=None, help='datapath')
parser.add_argument('--pretrained_path', type=str, default='checkpoints/checkpoint.tar', help='pretrained model path')
parser.add_argument('--resume_path', type=str, default=None, help='resume path')
parser.add_argument('--save_path', type=str, default='results/train_anynet', help='the path of saving checkpoints and log')

parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--train_file', type=str, default=None)
parser.add_argument('--validation_file', type=str, default=None)
parser.add_argument('--load_npy', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')

parser.add_argument('--epochs', type=int, default=110, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')   
parser.add_argument('--train_bsize', type=int, default=8, help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8, help='batch size for testing (default: 8)')

parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1, 1])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=5)

args = parser.parse_args()

if args.datatype == '2015':
    from Models.AnyNet.dataloader import KITTIloader2015 as ls
    from Models.AnyNet.dataloader import KITTILoader as DA
elif args.datatype == '2012':
    from Models.AnyNet.dataloader import KITTIloader2012 as ls
    from Models.AnyNet.dataloader import KITTILoader as DA
elif args.datatype == 'other':
    from Models.AnyNet.dataloader import KITTI_dataset as ls
    from Models.AnyNet.dataloader import KITTILoader3D as DA
    
def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    if args.datatype == 'other':
        train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
            args.data_path, args.train_file, args.validation_file, load_npy=args.load_npy)
    else:
        train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
            args.data_path, log, args.split_file)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, load_npy=args.load_npy),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, load_npy=args.load_npy),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0
    cudnn.benchmark = True
    if args.resume_path:
        if os.path.isfile(args.resume_path):
            log.info("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume_path, checkpoint['epoch']))
            # test(TestImgLoader, model, log, checkpoint['epoch'])
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume_path))
            log.info("=> Will start from scratch.")
    elif args.pretrained_path:
        if os.path.isfile(args.pretrained_path):
            checkpoint = torch.load(args.pretrained_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'".format(args.pretrained_path))
        else:
            log.info("=> no pretrained model found at '{}'".format(
                args.pretrained_path))
            log.info("=> Will start from scratch.")
    else:
        log.info("=> Will start from scratch.")

    start_full_time = time.time()
    if args.evaluate:
        test(TestImgLoader, model, log)
        return
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        test(TestImgLoader, model, log, epoch)

    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    model.train()

    for batch_idx, (imgL, imgR, disp_L) in tqdm.tqdm(enumerate(dataloader), ascii=True, desc=("training epoch " + str(epoch)), total=(length_loader), unit='iteration'):
        imgL = imgL.cuda().float()
        imgR = imgR.cuda().float()
        disp_L = disp_L.cuda().float()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]

        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], reduction='mean')
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())
        torch.cuda.empty_cache()

    info_str = '\t'.join(['Stage {} = {:.4f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss at {}: '.format(epoch) + info_str)


def test(dataloader, model, log, epoch=-1):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    Error = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
    }
    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in tqdm.tqdm(enumerate(dataloader), ascii=True, desc="Testing", total=(length_loader), unit='iteration'):
        imgL = imgL.cuda().float()
        imgR = imgR.cuda().float()
        disp_L = disp_L.cuda().float()

        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())
                Error[str(x)].append(D1s[x].val)
        torch.cuda.empty_cache()

    info_str = ', '.join(['Stage {}={:.3f}%'.format(x, D1s[x].avg * 100) for x in range(stages)])
    Error3 = np.asarray(Error["3"], dtype=np.float32)
    # log.info("Max Error is {}, while Min Error is {}".format(np.max(Error3), np.min(Error3)))
    if epoch > -1:
        log.info('Average test 3-Pixel Error at Epoch {}: '.format(epoch) + info_str)
    else:
        log.info('Average test 3-Pixel Error: ' + info_str)
    torch.cuda.empty_cache()


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 32:
        lr = args.lr
    elif epoch <= 55:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

if __name__ == '__main__':
    main()
