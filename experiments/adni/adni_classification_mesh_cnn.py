import os
import shutil
import sys
import time
from glob import glob

import numpy as np
from sklearn import model_selection as ms

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer

def runTrain():
    # train data
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)

    # test and validation data
    opt_ = TestOptions().parse()
    opt_.serial_batches = True  # no shuffle
    dataset_test = DataLoader(opt_)
    opt_.phase = 'val'
    dataset_val = DataLoader(opt_)

    print(f'#meshes (train/val/test) = {dataset_size}/{len(dataset_val)}/{len(dataset_test)}')

    model = create_model(opt)
    num_model_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        def eval_acc(data, phase):
            writer.reset_counter()
            for i, d in enumerate(data):
                model.set_input(d)
                writer.update_counter(*model.test())
            writer.print_acc(epoch, writer.acc, phase=phase)

        if epoch % opt.run_test_freq == 0:
            eval_acc(dataset_val, 'VAL')
            eval_acc(dataset_test, 'TEST')
            eval_acc(dataset, 'TRAIN')

    writer.close()

if __name__ == '__main__':
    sys.argv = ['train.py', '--dataroot', '../data/ADNI_MeshCNN', \
                '--name', 'hippocampus', '--ncf', '64', '128', '256', '256', \
                '--ninput_edges', '6834', '--pool_res', '5000', '4000', ' 3000', '2000', \
                '--batch_size', '8', '--norm', 'group', '--resblocks', '1', \
                '--flip_edges', '0.2', '--slide_verts', '0.2', '--num_aug', '20', '--niter_decay', '0', '--niter', '100', '--lr', '0.00001']

    sys.argv += ['--save_epoch_freq', '10000', '--no_vis']

    # find hippocampi
    files = sorted(glob('../../data/adni/**/*obj', recursive=True))
    y = np.array([int(f.split('/')[-2] == 'AD') for f in files])
    assert len(np.unique(y)) == 2


    # monte carlo cross validation
    for seed in range(100):

        phase = ['train', 'val', 'test']
        # split data in training, validation and testing sets
        idx_train, _ = ms.train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        # hold back validation set from the training data
        idx_train, idx_val = ms.train_test_split(idx_train, test_size=0.25, random_state=seed, stratify=y[idx_train])
        set_id = np.full(len(y), 2)
        set_id[idx_val] = 1
        set_id[idx_train] = 0

        # prepare file structure for MeshCNN DataLoader
        shutil.rmtree('../../data/adni_mesh_cnn', ignore_errors=True)
        for i, f in enumerate(files):
            trgt = os.path.dirname(f).replace('adni', 'adni_mesh_cnn')
            trgt = os.path.join(trgt, phase[set_id[i]], os.path.basename(f))
            os.makedirs(os.path.dirname(trgt), exist_ok=True)
            os.symlink(os.path.abspath(f), trgt)

        # run training
        runTrain()