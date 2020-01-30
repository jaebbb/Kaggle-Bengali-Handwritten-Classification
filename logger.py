import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import tqdm


class CustomLogger():
    def __init__(self, out):
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        # add the learning rate :)
        self.log_headers=['epoch','iteration','train/loss','train/graph_acc','train/vowel_acc','train/conso_acc','valid/loss','valid/graph_acc','valid/vowel_acc','valid/conso_acc','elapsed_time', 'lr']

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

    def write(self, is_train, epoch, iteration, loss, acc1, acc2, acc3, lr=None):

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo'))-self.timestamp_start).total_seconds()
            log = None
            if is_train:
                log = [epoch, iteration] + [loss] + [acc1] +[acc2] +[acc3] + [''] * 4 + [elapsed_time] + [lr]
            else:
                log = [epoch, iteration] + [''] * 4 + [loss] + [acc1] +[acc2] +[acc3]  + [elapsed_time]

            log = map(str, log)
            f.write(','.join(log) + '\n')


