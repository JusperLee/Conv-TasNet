import torch
import time
import os
from utils import get_logger
from Conv_TasNet import check_parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm_
from SI_SNR import si_snr_loss


def to_device(dicts, device):
    '''
       load dict data to cuda
    '''
    def to_cuda(datas):
        if isinstance(datas, torch.Tensor):
            return datas.to(device)
        else:
            raise RuntimeError('datas is not torch.Tensor type')

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    else:
        raise RuntimeError('input egs\'s type is not dict')


class Trainer():
    '''
       Trainer of Conv-Tasnet
       input:
             net: load the Conv-Tasnet model
             checkpoint: save model path
             optimizer: name of opetimizer
             gpu_ids: (int/tuple) id of gpus
             optimizer_kwargs: the kwargs of optimizer
             clip_norm: maximum of clip norm, default: None
             min_lr: minimun of learning rate
             patience: Number of epochs with no improvement after which learning rate will be reduced
             factor: Factor by which the learning rate will be reduced. new_lr = lr * factor
             logging_period: How long to print
             resume: the kwargs of resume, including path of model, Whether to restart
             stop: Stop training cause no improvement
    '''

    def __init__(self,
                 net,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 stop=6):
        # if the cuda is available and if the gpus' type is tuple
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid

        # mkdir the file of Experiment path
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint

        # build the logger object
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # current epoch
        self.stop = stop

        # Whether to resume the model
        if resume['resume_state']:
            if not os.path.exists(resume['path']):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = torch.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume['path'], self.cur_epoch))
            # load nnet
            net.load_state_dict(cpt["model_state_dict"])
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optimizer_state_dict"])
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

        # check model parameters
        self.param = check_parameters(self.net)

        # Reduce lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=patience, verbose=True, min_lr=min_lr)

        # logging
        self.logger.info("Starting preparing model ............")
        self.logger.info("Model summary:\n{}".format(net))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.param))
        self.clip_norm = clip_norm
        # clip norm
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def create_optimizer(self, optimizer, kwargs, state=None):
        '''
           create optimizer
           optimizer: (str) name of optimizer
           kwargs: the kwargs of optimizer
           state: the load model optimizer state
        '''
        supported_optimizer = {
            "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": torch.optim.Adam,  # weight_decay, lr
            "adadelta": torch.optim.Adadelta,  # weight_decay, lr
            "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": torch.optim.Adamax  # lr, weight_decay
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.net.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def save_checkpoint(self, best=True):
        '''
            save model
            best: the best model
        '''
        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict()
            },
            os.path.join(self.checkpoint,
                         "{0}.pt".format("best" if best else "last")))

    def train(self, train_dataloader):
        '''
           training model
        '''
        self.logger.info('Training model ......')
        losses = []
        start = time.time()
        current_step = 0
        for egs in train_dataloader:
            current_step += 1
            egs = to_device(egs, self.device)
            self.optimizer.zero_grad()
            ests = data_parallel(self.net, egs['mix'], device_ids=self.gpuid)
            loss = si_snr_loss(ests, egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()
            losses.append(loss.item())
            if len(losses) == self.logging_period:
                avg_loss = sum(
                    losses[-self.logging_period:])/self.logging_period
                self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                    self.cur_epoch, current_step, self.optimizer[0].param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses)/len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
                    self.cur_epoch, self.optimizer[0].param_groups[0]['lr'], total_loss_avg,(end-start)/60))
        return total_loss_avg
    
    def val(self,val_dataloader):
        '''
           validation model
        '''
        self.logger.info('Validation model ......')
