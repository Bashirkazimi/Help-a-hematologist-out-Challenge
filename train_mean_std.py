"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time

import matplotlib.pyplot as plt
import torch

from options.train_options import TrainOptions
from data import create_dataset, get_source_and_target, get_source_and_target_mean_std
from models import create_model
# from util.visualizer import Visualizer
import os
from tifffile import imsave


def plot_results(realA, realB, fakeA, fakeB, recA, recB, save_path, idx):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    realA = (realA - realA.min()) / (realA.max() - realA.min())
    realB = (realB - realB.min()) / (realB.max() - realB.min())
    fakeA = (fakeA - fakeA.min()) / (fakeA.max() - fakeA.min())
    fakeB = (fakeB - fakeB.min()) / (fakeB.max() - fakeB.min())
    recA = (recA - recA.min()) / (recA.max() - recA.min())
    recB = (recB - recB.min()) / (recB.max() - recB.min())
    ax[0, 0].imshow(realA.transpose((1, 2, 0)))
    ax[0, 0].set_title('Real Mat_19/Ace_20', fontsize=18)

    ax[0, 1].imshow(fakeB.transpose((1, 2, 0)))
    ax[0, 1].set_title('Mat_19/Ace_20 as WBC', fontsize=18)

    # ax[0, 1].imshow(fakeA.transpose((1, 2, 0)))
    # ax[0, 1].set_title('Fake Source', fontsize=18)

    ax[0, 2].imshow(recA.transpose((1, 2, 0)))
    ax[0, 2].set_title('Reconstructed Mat_19/Ace_20', fontsize=18)

    ax[1, 0].imshow(realB.transpose((1, 2, 0)))
    ax[1, 0].set_title('Real WBC', fontsize=18)

    ax[1, 1].imshow(fakeA.transpose((1, 2, 0)))
    ax[1, 1].set_title('WBC as Mat_19/Ace_20', fontsize=18)

    # ax[1, 1].imshow(fakeB.transpose((1, 2, 0)))
    # ax[1, 1].set_title('Fake Target', fontsize=18)

    ax[1, 2].imshow(recB.transpose((1, 2, 0)))
    ax[1, 2].set_title('Reconstructed WBC', fontsize=18)

    plt.savefig(os.path.join(save_path, '{}.png'.format(idx)))

    plt.close()


def save_it(visuals, save_path):
    real_As = visuals['real_A'].cpu().numpy()
    real_Bs = visuals['real_B'].cpu().numpy()
    fake_As = visuals['fake_A'].cpu().numpy()
    fake_Bs = visuals['fake_B'].cpu().numpy()
    rec_As = visuals['rec_A'].cpu().numpy()
    rec_Bs = visuals['rec_B'].cpu().numpy()
    for i in range(real_As.shape[0]):
        plot_results(real_As[i], real_Bs[i], fake_As[i], fake_Bs[i], rec_As[i], rec_Bs[i], save_path, i)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)    # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)

    dataset, val = get_source_and_target_mean_std('metadata.csv', batch_size=opt.batch_size, num_workers=opt.num_threads)
    dataset_size = len(dataset.dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     # model.compute_visuals()

                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)

                print(message)  # print the message
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)  # save the message

                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        b = next(iter(val))
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        os.makedirs('figures_mean_std/epoch_{}'.format(epoch), exist_ok=True)
        save_it(visuals, 'figures_mean_std/epoch_{}'.format(epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

