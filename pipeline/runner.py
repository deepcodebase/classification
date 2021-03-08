import shutil
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm, trange
from hpman.m import _
import horovod.torch as hvd
from rich.console import Console
from rich.progress import (
    _TrackThread, Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeRemainingColumn, TimeElapsedColumn)
from tensorboardX import SummaryWriter

from dataset import get_dataset, Prefetcher
from model import get_model
from utils.misc import get_date, get_time, time2str, model_size
from evaluation import Metric, compute_acc, get_loss


class BaseRunner:

    def __init__(self):
        self.init_framework()
        self.init_path()
        self.init_logging()
        self.init_tensorboard()
        self.init_data()
        self.init_model()
        self.init_optimizer()
        self.load_checkpoint()
        self.sync()
        self.c.log(f'[green]rank #{hvd.rank()} is ready.')

    def is_root(self):
        return hvd.rank() == 0

    def init_framework(self):
        hvd.init()
        torch.manual_seed(_('seed', 2021))
        if _('gpu'):
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(_('seed'))

        self.epoch = 0
        self.step = 0
        self.lr = _("lr", 0.01)
        self.loss_func = get_loss(_("loss", "nll_loss"))
        self.ckpt_fmt = 'epoch_{epoch:04d}.pt'

    def init_logging(self):
        self.c = Console(width=80)
        # self.p = Progress(
        #     SpinnerColumn(),
        #     TextColumn("[progress.description]{task.description}"),
        #     BarColumn(),
        #     TextColumn("{task.completed}/{task.total}"),
        #     TimeRemainingColumn(),
        #     TimeElapsedColumn(),
        #     console=self.c,
        #     transient=True,
        #     disable=not self.is_root(),
        #     auto_refresh=True
        # )
        # self.p.start()
    
    def track(self,
            sequence,
            description="Working...",
            total: int = None,
            completed: int = 0,
            update_period: float = 0.1):
        if self.is_root():

            if total is None:
                try:
                    task_total = len(sequence)
                except:
                    raise ValueError(
                        f"unable to get size of {sequence!r}, please specify 'total'"
                    )
            else:
                task_total = total

            task_id = self.p.add_task(
                description, total=task_total, completed=completed)

            if self.p.auto_refresh:
                with _TrackThread(self.p, task_id, update_period) as track_thread:
                    for value in sequence:
                        yield value
                        track_thread.completed += 1
            else:
                for value in sequence:
                    yield value
                    self.p.advance(task_id, 1)
                    self.p.refresh()
            task = self.p._tasks[task_id]
            self.c.log(
                f'[yellow]time used for {task.description}:'
                f' {time2str(task.elapsed)}')
            self.p.remove_task(task_id)
        else:
            for value in sequence:
                yield value

    def init_path(self):
        self.output = Path('output') / _('dataset', 'mnist') \
            / f"{_('label', 'base')}_{_('date', get_date())}"
        self.log_dir = self.output / 'log'
        self.ckpt_dir = self.output / 'ckpt'
        self.tb_dir = self.output / 'tb'
        self.result_dir = self.output / 'result'
        self.result_path = self.result_dir / 'result.json'

        for directory in [
                self.log_dir, self.ckpt_dir, self.tb_dir, self.result_dir]:
            if _('restart', False):
                shutil.rmtree(directory, True)
            directory.mkdir(parents=True, exist_ok=True)

    def init_tensorboard(self):
        self.tb = SummaryWriter(logdir=str(self.tb_dir))

    def init_data(self):
        self.train_loader, self.train_sampler = self._get_data('train')
        self.val_loader, self.val_sampler = self._get_data('val')

    def _get_data(self, split='train'):
        dataset = get_dataset(
            _('dataset'), _('data_location', '/data/mnist'), split)
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank())
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=_('batch_size', 64), sampler=sampler,
            num_workers=_('num_workers', 2))
        return dataloader, sampler

    def init_model(self):
        self.model = get_model(_('model', 'basenet'))
        if self.is_root():
            self.c.log(f'[green]model size: {model_size(self.model)}')
        if _('gpu'):
            self.model.cuda()

    def init_optimizer(self):
        lr_scaler = hvd.size() if not _("use_adasum", False) else 1
        if _('gpu') and _("use_adasum") and hvd.nccl_build():
            lr_scaler = hvd.local_size()
        optimizer = optim.SGD(
            self.model.parameters(), lr=_("lr") * lr_scaler,
            momentum=_("momentum", 0.5), weight_decay=_("weight_decay", 0))
        compression = hvd.Compression.fp16 if _("fp16_allreduce", False) \
            else hvd.Compression.none
        self.optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=self.model.named_parameters(),
            compression=compression,
            backward_passes_per_step=_("batches_per_allreduce", 1),
            op=hvd.Adasum if _("use_adasum") else hvd.Average,
            gradient_predivide_factor=_("gradient_predivide_factor", 1.))

    def load_checkpoint(self, ckpt_path=None):
        if self.is_root():
            if ckpt_path is None:
                ckpt_path = self.get_latest_checkpoint()
            if ckpt_path is not None:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.epoch = checkpoint['epoch'] + 1
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def get_latest_checkpoint(self):
        ckpt_list = list(sorted(self.ckpt_dir.glob('*.pt')))
        if len(ckpt_list) > 0:
            return ckpt_list[-1]
        else:
            return None

    def sync(self):
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def train(self):
        start_epoch = self.epoch
        if self.is_root():
            self.c.log(f'[green]training starts from epoch #{start_epoch}.')
        for self.epoch in trange(
                start_epoch, _("epochs", 10), desc='training',
                total=_("epochs"), initial=start_epoch, ncols=80):
        # for self.epoch in self.track(
        #         range(start_epoch, _("epochs", 10)), description='training',
        #         total=_("epochs"), completed=start_epoch):
            self.train_epoch()
            self.save_checkpoint()
            self.val_epoch()

    def train_epoch(self):
        self.model.train()
        self.train_sampler.set_epoch(self.epoch)
        for self.step, (data, target) in enumerate(tqdm(
                self.train_loader, desc=f'train #{self.epoch}', ncols=80)):
        # for self.step, (data, target) in enumerate(self.track(
        #         self.train_loader, description=f'train #{self.epoch}')):
            self.adjust_lr()
            if _("gpu"):
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            self.logging(loss=loss.item(), lr=self.lr / hvd.size())

    def train_epoch_prefetch(self):
        self.model.train()
        self.train_sampler.set_epoch(self.epoch)
        prefetcher = Prefetcher(self.train_loader)
        data, target = prefetcher.next()
        self.step = 0
        task_id = self.p.add_task(
            f'train #{self.epoch}', total=len(self.train_loader))
        while data is not None:
            self.adjust_lr()
            if _("gpu"):
                data, target = data.cuda(), target.cuda()
            if _("gpu"):
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            self.logging(loss=loss.item(), lr=self.lr / hvd.size())
            data, target = prefetcher.next()
            self.step += 1
            self.p.advance(task_id)
        self.p.remove_task(task_id)

    def adjust_lr(self):
        # if self.epoch < _("warmup_epochs", 5):
        #     epoch = self.epoch + float(self.step + 1) / len(self.train_loader)
        #     lr_adj = 1. / hvd.size() * (
        #         epoch * (hvd.size() - 1) / _("warmup_epochs") + 1)
        # elif self.epoch < 30:
        #     lr_adj = 1.
        # elif self.epoch < 60:
        #     lr_adj = 1e-1
        # elif self.epoch < 80:
        #     lr_adj = 1e-2
        # else:
        #     self.lr_adj = 1e-3
        if self.epoch < 30:
            lr_adj = 1.
        elif self.epoch < 60:
            lr_adj = 1e-1
        else:
            lr_adj = 1e-2
        self.lr = _("lr") * hvd.size() * _("batches_per_allreduce") * lr_adj
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def logging(self, **kwargs):
        if self.is_root():
            if self.model.training and self.step % _("log_interval", 10) != 0:
                return
            total_step = self.epoch * len(self.train_loader) + self.step \
                if self.model.training else self.epoch
            split = 'train' if self.model.training else 'val'
            for key, value in kwargs.items():
                self.tb.add_scalar(f'{split}/{key}', value, total_step)
            log_str = ''
            for key, value in kwargs.items():
                log_str += f'{key}: {value:.4f} '
            color = 'white'  if self.model.training else 'green'
            # self.c.log(f'[{color}]{split} - {log_str}')
            with self.c.capture() as capture:
                self.c.log(f'[{color}]{split} - {log_str}')
            str_output = capture.get()
            tqdm.write(str_output.strip())
            # tqdm.write(f'[{color}]{split} - {log_str}')

    def val_epoch(self):
        val_loss = Metric('val_loss')
        val_acc = Metric('val_acc')
        self.model.eval()
        with torch.no_grad():
            for data, target in tqdm(
                    self.val_loader, desc=f'val #{self.epoch}', ncols=80):
            # for data, target in self.track(
            #         self.val_loader, description=f'val #{self.epoch}'):
                if _("gpu"):
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                loss = self.loss_func(output, target)
                val_acc.update(compute_acc(output, target))
                val_loss.update(loss)
        self.logging(loss=val_loss.avg, acc=val_acc.avg)

    def test(self):
        test_loss = Metric('test_loss')
        test_acc = Metric('test_acc')
        self.model.eval()
        self.train_loader = self.train_sampler = None
        test_loader, test_sampler = self._get_data('test')
        for data, target in test_loader:
            if _("gpu"):
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.loss_func(output, target)
            test_acc.update(compute_acc(output, target))
            test_loss.update(loss)

        if hvd.rank() == 0:
            self.c.log(
                '[cyan]Test set: Average loss: {:.4f},'
                'Accuracy: {:.2f}%\n'.format(
                test_loss.avg, 100. * test_acc.avg))

    def save_checkpoint(self):
        if self.is_root():
            filepath = self.ckpt_dir / self.ckpt_fmt.format(
                epoch=self.epoch + 1)
            state = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, filepath)