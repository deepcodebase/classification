import torch
import torch.optim as optim
from hpman.m import _
import horovod.torch as hvd
import rich

from dataset import get_dataset
from model import get_model


class HvdTrainer:

    def __init__(self):
        self.init_framework()
        self.init_data()
        self.init_model()
        self.init_optimizer()
        self.sync()

    def init_framework(self):
        hvd.init()
        torch.manual_seed(_('seed', 2021))
        if _('gpu', True):
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(_('seed'))

    def init_data(self):
        self.train_loader, self.train_sampler = self._get_data('train')

    def _get_data(self, split='train'):
        dataset = get_dataset(_('dataset', 'mnist'), split)
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank())
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=_('batch_size', 64), sampler=sampler,
            num_workers=_('num_workers', 2))
        return dataloader, sampler

    def init_model(self):
        self.model = get_model(_('model', 'basenet'))
        if _('gpu'):
            self.model.cuda()

    def init_optimizer(self):
        lr_scaler = hvd.size() if not _("use_adasum", False) else 1
        if _('gpu') and _("use_adasum") and hvd.nccl_build():
            lr_scaler = hvd.local_size()
        optimizer = optim.SGD(
            self.model.parameters(), lr=_("lr", 0.01) * lr_scaler,
            momentum=_("momentum", 0.5))
        compression = hvd.Compression.fp16 if _("fp16_allreduce", False) \
            else hvd.Compression.none
        self.optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=self.model.named_parameters(),
            compression=compression,
            backward_passes_per_step=_("batches_per_allreduce", 1),
            op=hvd.Adasum if _("use_adasum") else hvd.Average,
            gradient_predivide_factor=_("gradient_predivide_factor", 1.))

    def sync(self):
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def run(self):
        for self.epoch in range(_("epochs", 10)):
            self.train_epoch()
        self.test()

    def train_epoch(self):
        self.model.train()
        # Horovod: set epoch to sampler for shuffling.
        self.train_sampler.set_epoch(self.epoch)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if _("gpu"):
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            loss, output = self.model.compute_loss(data, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % _("log_interval", 10) == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_sampler),
                    100. * batch_idx / len(self.train_loader), loss.item()))
    
    def test(self):
        self.model.eval()
        self.train_loader = self.test_sampler = None
        test_loader, test_sampler = self._get_data('test')
        test_loss = 0.
        test_accuracy = 0.
        for data, target in test_loader:
            if _("gpu"):
                data, target = data.cuda(), target.cuda()
            test_loss, output = self.model.compute_loss(data, target)
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(
                target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        test_loss = self.metric_average(test_loss, 'avg_loss')
        test_accuracy = self.metric_average(test_accuracy, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))

    def metric_average(self, val, name):
        tensor = val.detach()
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()