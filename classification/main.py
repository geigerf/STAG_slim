import sys
# sys.path.insert(0, '/home/msc20f10/Python_Code/pytorch_stag/STAG_slim')
import numpy as np
import os
import time
import shutil
import random
import argparse

filePath = os.path.abspath(__file__)
fileDir = os.path.dirname(filePath)
parentDir = os.path.dirname(fileDir)
sys.path.insert(0, parentDir)

localtime = time.localtime(time.time())
localtime = time.asctime(localtime)
default_experiment = localtime.replace(' ', '_')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Touch-Classification.')
parser.add_argument('--dataset', default=('/scratch1/msc20f10/data/'
                                          + 'classification/metadata.mat'),
                    help="Path to metadata.mat file.")
parser.add_argument('--nframes', type=int,
                    help='Number of input frames [1--8]', default=1)
parser.add_argument('--reset', type=str2bool, nargs='?', const=True,
                    default=False, help="Start from scratch (do not load weights).")
parser.add_argument('--test', type=str2bool, nargs='?', const=True,
                    default=False, help="Just run test and quit.")
parser.add_argument('--snapshotDir',
                    default='/scratch1/msc20f10/stag/training_checkpoints',
                    help="Where to store checkpoints during training.")
parser.add_argument('--gpu', type=int, default=None,
                    help=("ID number of the GPU to use [0--4]."
                          + "If left unspecified all visible CPUs."))
parser.add_argument('--experiment', default=default_experiment,
                    help="Name of the current experiment.")
parser.add_argument('--nfilters', type=int, default=16,
                    help="Number of filters for the first convolution.")
parser.add_argument('--dropout', type=float, default=0.4,
                    help="Dropout between the two ResNet blocks.")
parser.add_argument('--dropoutFC', type=float, default=0,
                    help="Dropout before the fully connected layer.")
parser.add_argument('--dropoutMax', type=float, default=0,
                    help="Dropout before the first max pooling layer.")
parser.add_argument('--addConv', type=str2bool, default=False,
                    help="Convolution after the concatenation.")
parser.add_argument('--epochs', type=int, default=60,
                    help="Number of epochs to train.")
parser.add_argument('--stride', type=int, default=1,
                    help="Stride to use for the first convolution.")
parser.add_argument('--dilation', type=int, default=1,
                    help="Dilation to use for the first convolution.")
parser.add_argument('--kernel', type=int, default=3,
                    help="Kernel size to use for the first convolution.")
parser.add_argument('--customData', type=str2bool, default=False,
                    help="Use custom data loader.")
args = parser.parse_args()

# This line makes only the chosen GPU visible.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#________________________________________________________________________________________#

import torch
import torch.backends.cudnn as cudnn

from ObjectClusterDataset import ObjectClusterDataset
from custom_dataloader import load_data, DataLoader


nClasses = 27
nFrames = args.nframes
epochs = args.epochs
batch_size = 32
workers = 0
experiment = 'nf' + str(nFrames) + '_' + args.experiment

metaFile = args.dataset
doFilter = True


class Trainer(object):
    def __init__(self):
        self.init()
        super(Trainer, self).__init__()


    def init(self):
        # Init model
        if args.customData:
            self.data = load_data(filename=metaFile, augment=False, kfold=3,
                                  seed=123)
        
        self.val_loader = self.loadDatasets('test', False, False)

        self.initModel()


    def loadDatasets(self, split='train', shuffle=True,
                     useClusterSampling=False):
        if args.customData:
            idx = 2*int(split!='train')
            set_size = len(self.data[idx+1])
            return torch.utils.data.DataLoader(
                DataLoader(self.data[idx].reshape((set_size,32,32)),
                           self.data[idx+1], augment=False,
                           nframes=nFrames, use_clusters=False,
                           nclasses=nClasses),
            batch_size=batch_size, shuffle=shuffle, num_workers=workers)
        else:
            return torch.utils.data.DataLoader(
                ObjectClusterDataset(split=split,
                                     doAugment=(split=='train'),
                                     doFilter=doFilter,
                                     sequenceLength=nFrames,
                                     metaFile=metaFile,
                                     useClusters=useClusterSampling),
                batch_size=batch_size, shuffle=shuffle, num_workers=workers)


    def run(self):
        trainloss_history = []
        trainprec_history = []
        testloss_history = []
        testprec_history = []
        print('[Trainer] Starting...')

        self.counters = {
            'train': 0,
            'test': 0,
        }

        if args.test:
            self.doSink()
            return

        val_loader = self.val_loader
        train_loader = self.loadDatasets('train', True, False)

        for epoch in range(epochs):
            print('Epoch %d/%d....' % (epoch, epochs))
            self.model.updateLearningRate(epoch)

            trainprec1, _, trainloss, cm_train = self.step(train_loader,
                                                           self.model.epoch,
                                                           isTrain = True)
            prec1, _, testloss, _ = self.step(val_loader,
                                           self.model.epoch,
                                           isTrain = False,
                                           sinkName=None)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.model.bestPrec
            if is_best:
                self.model.bestPrec = prec1
            self.model.epoch = epoch + 1
            if not args.snapshotDir is None:
                self.saveCheckpoint(self.model.exportState(), is_best)
                
            trainloss_history.append(trainloss)
            trainprec_history.append(trainprec1)
            testloss_history.append(testloss)
            testprec_history.append(prec1)

        # Final results
        res, cm_test, cm_test_cl = self.doSink()
        
        savedir = '/home/msc20f10/Python_Code/results/stag/'
        if args.experiment == 'slim16':
            savedir = savedir + 'slim16/'
        else:
            savedir = savedir + 'np' + str(self.model.nParams) + '_'
        np.save(savedir + experiment + '_history.npy',
                np.array([trainloss_history, trainprec_history,
                          testloss_history, testprec_history, res,
                          cm_train.numpy(), cm_test.numpy(), cm_test_cl.numpy(),
                          self.model.ntParams, self.model.nParams]))
        
        print('DONE')


    def doSink(self):
        res = {}

        print('Running test...')
        res['test-top1'], res['test-top3'],\
            _, conf_mat = self.step(self.val_loader, self.model.epoch,
                                    isTrain=False, sinkName='test')

        print('Running test with clustering...')
        val_loader_cluster = self.loadDatasets('test', False, True)
        res['test_cluster-top1'], res['test_cluster-top3'],\
            _, conf_mat_cluster = self.step(val_loader_cluster,
                                            self.model.epoch,
                                            isTrain=False,
                                            sinkName='test_cluster')

        print('--------------\nResults:')
        for k,v in res.items():
            print('\t%s: %.3f %%' % (k,v))
            
        return res, conf_mat, conf_mat_cluster

    
    def initModel(self):
        cudnn.benchmark = True

        from ClassificationModel import ClassificationModel as Model # the main model

        initShapshot = os.path.join(args.snapshotDir, experiment,
                                    'model_best.pth.tar')
        if args.reset:
            initShapshot = None

        self.model = Model(numClasses=nClasses,
                           sequenceLength=nFrames, inplanes=args.nfilters,
                           dropout=args.dropout, dropoutFC=args.dropoutFC,
                           stride=args.stride, dilation=args.dilation,
                           kernel=args.kernel, addConv=args.addConv,
                           dropoutMax=args.dropoutMax)
        self.model.epoch = 0
        self.model.bestPrec = -1e20
        
        if not initShapshot is None:
            state = torch.load(initShapshot)
            assert not state is None, 'Warning: Could not read checkpoint %s!' % initShapshot
            print('Loading checkpoint %s...' % (initShapshot))
            self.model.importState(state)


    def step(self, data_loader, epoch, isTrain=True, sinkName=None):
        if isTrain:
            data_loader.dataset.refresh()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        results = {
            'batch': [],
            'rec': [],
            'frame': [],
        }
        catRes = lambda res,key: res[key].cpu().numpy() if not key in results else np.concatenate((results[key], res[key].cpu().numpy()), axis=0)
    
        end = time.time()
        conf_matrix = torch.zeros(nClasses, nClasses).cpu()
        for i, (inputs) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputsDict = {
                'image': inputs[1],
                'pressure': inputs[2],
                'objectId': inputs[3],
                }

            res, loss = self.model.step(inputsDict, isTrain,
                                        params = {'debug': True})

            losses.update(loss['Loss'], inputs[0].size(0))
            top1.update(loss['Top1'], inputs[0].size(0))
            top3.update(loss['Top3'], inputs[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if isTrain:
                self.counters['train'] = self.counters['train'] + 1

            print('{phase}: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'
                  .format(epoch, i, len(data_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top3=top3,
                          phase=('Train' if isTrain else 'Test')))
            
            for t, p in zip(inputs[3].view(-1), res['pred'].view(-1)):
                conf_matrix[t.long(), p.long()] += 1
        
        self.counters['test'] = self.counters['test'] + 1

        return top1.avg, top3.avg, losses.avg, conf_matrix


    def saveCheckpoint(self, state, is_best):
        snapshotDir = os.path.join(args.snapshotDir, experiment)
        if not os.path.isdir(snapshotDir):
            os.makedirs(snapshotDir, 0o777)
        chckFile = os.path.join(snapshotDir, 'checkpoint.pth.tar')
        print('Writing checkpoint to %s...' % chckFile)
        torch.save(state, chckFile)
        if is_best:
            bestFile = os.path.join(snapshotDir, 'model_best.pth.tar')
            shutil.copyfile(chckFile, bestFile)
        print('\t...Done.')


    @staticmethod
    def make():
       
        random.seed(454878 + time.time() + os.getpid())
        np.random.seed(int(12683 + time.time() + os.getpid()))
        torch.manual_seed(23142 + time.time() + os.getpid())

        ex = Trainer()
        ex.run()


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

        
if __name__ == "__main__":    
    Trainer.make()
      
    