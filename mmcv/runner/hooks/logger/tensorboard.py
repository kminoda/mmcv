# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
from distutils.version import LooseVersion

from mmcv.utils import TORCH_VERSION
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

        self.image_logged_epochs = []

    @master_only
    def before_run(self, runner):
        super(TensorboardLoggerHook, self).before_run(runner)
        if (LooseVersion(TORCH_VERSION) < LooseVersion('1.1')
                or TORCH_VERSION == 'parrots'):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True, allow_image=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            elif 'image' in tag:
                epoch = self.get_epoch(runner)
                if epoch in self.image_logged_epochs:
                    continue
                self.writer.add_image(tag, val, epoch)
                self.image_logged_epochs.append(epoch)
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()



# @HOOKS.register_module()
# class TensorboardImageLoggerHook(LoggerHook):

#     def __init__(self,
#                  log_dir=None,
#                  interval=10,
#                  ignore_last=True,
#                  reset_flag=False,
#                  by_epoch=True):
#         super(TensorboardImageLoggerHook, self).__init__(interval, ignore_last,
#                                                     reset_flag, by_epoch)
#         self.log_dir = log_dir

#     @master_only
#     def before_run(self, runner):
#         super(TensorboardImageLoggerHook, self).before_run(runner)
#         if (LooseVersion(TORCH_VERSION) < LooseVersion('1.1')
#                 or TORCH_VERSION == 'parrots'):
#             try:
#                 from tensorboardX import SummaryWriter
#             except ImportError:
#                 raise ImportError('Please install tensorboardX to use '
#                                   'TensorboardImageLoggerHook.')
#         else:
#             try:
#                 from torch.utils.tensorboard import SummaryWriter
#             except ImportError:
#                 raise ImportError(
#                     'Please run "pip install future tensorboard" to install '
#                     'the dependencies to use torch.utils.tensorboard '
#                     '(applicable to PyTorch 1.1 or higher)')

#         if self.log_dir is None:
#             self.log_dir = osp.join(runner.work_dir, 'tf_logs')
#         self.writer = SummaryWriter(self.log_dir)

#     @master_only
#     def log(self, runner):
#         tags = self.get_image_tags(runner)
#         print('TAGS!!!!!!!!!!!!!!!!!!!!!!!!', tags)
#         for tag, val in tags.items():
#             self.writer.add_scalar(tag, val, self.get_iter(runner))
    
#     def get_image_tags(self, runner):
#         tags = {}
#         for var, val in runner.log_buffer.output.items():
#             if 'image' in var:
#                 tags[var] = val
#         return tags

#     @master_only
#     def after_run(self, runner):
#         self.writer.close()
