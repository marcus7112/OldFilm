# test a video flow diffusion model based on RegionMM for MUG dataset

import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
from misc import grid2fig, conf2fig
from datasets_mug import MUG_test
import random
from DM.modules.video_flow_diffusion_model import FlowDiffusion


start = timeit.default_timer()
BATCH_SIZE = 10
root_dir = '/data/hfn5052/text2motion/videoflowdiff_mug'
data_dir = "/data/hfn5052/text2motion/dataset/MUG"
GPU = "5"
postfix = "-j-sl-vr-of-tr-rmm"
# default value
ddim_sampling_eta = 1.0
timesteps = 1000
if "ddim" in postfix:
    sampling_step = 10
    ddim_sampling_eta = 1.0
    postfix = postfix + "%04d_%.2f" % (sampling_step, ddim_sampling_eta)
else:
    sampling_step = 1000
INPUT_SIZE = 128
N_FRAMES = 40
RANDOM_SEED = 1234
NUM_VIDEOS = 10
NUM_ITER = NUM_VIDEOS // BATCH_SIZE
MEAN = (0.0, 0.0, 0.0)
cond_scale = 1.0
# the path to trained DM
RESTORE_FROM = "/data/hfn5052/text2motion/videoflowdiff_mug/snapshots-j-sl-vr-of-tr-rmm/flowdiff_0005_S111600.pth"
CKPT_DIR = os.path.join(root_dir, "ckpt"+postfix)
os.makedirs(CKPT_DIR, exist_ok=True)
IMG_DIR = os.path.join(root_dir, "ckpt_img"+postfix)
os.makedirs(IMG_DIR, exist_ok=True)
print(root_dir)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("cond scale:", cond_scale)
print("sampling step:", sampling_step)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()


args = get_arguments()


def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowDiffusion(is_train=True,
                          sampling_timesteps=sampling_step,
                          ddim_sampling_eta=ddim_sampling_eta,
                          timesteps=timesteps,
                          config_pth="/workspace/code/demo-dgx2/RegionMM/mug128.yaml",
                          pretrained_pth="/data/hfn5052/text2motion/RegionMM/log-mug/mug128/"
                                         "snapshots/RegionMM_0100_S046500.pth")
    model.cuda()

    if args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            model.diffusion.load_state_dict(checkpoint['diffusion'])
            print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
            exit(-1)
    else:
        print("NO checkpoint found!")
        exit(-1)

    model.eval()

    setup_seed(args.random_seed)
    testloader = data.DataLoader(MUG_test(data_dir=data_dir,
                                          image_size=args.input_size,
                                          num_frames=N_FRAMES,
                                          mean=MEAN),
                                 batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()
    cnt = 0

    for i_iter, batch in enumerate(testloader):
        if i_iter >= NUM_ITER:
            break

        data_time.update(timeit.default_timer() - iter_end)

        real_vids, ref_texts, real_names = batch
        # use first frame of each video as reference frame
        ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
        bs = real_vids.size(0)

        batch_time.update(timeit.default_timer() - iter_end)

        with torch.no_grad():
            model.set_train_input(ref_img=ref_imgs, real_vid=real_vids, ref_text=ref_texts)
            model.forward()

        model.set_sample_input(sample_img=ref_imgs, sample_text=ref_texts)
        model.sample_one_video(cond_scale=cond_scale)

        for batch_idx in range(bs):
            # model.set_sample_input(sample_img=ref_imgs[batch_idx].unsqueeze(dim=0), sample_text=[ref_texts[batch_idx]])
            # model.sample_one_video(cond_scale=cond_scale)
            # save one video
            msk_size = ref_imgs.shape[-1]
            save_src_img = sample_img(ref_imgs, batch_idx)
            nf = real_vids.size(2)
            new_im_list = []

            img_dir_name = "%04d_%s_%.2f" % (cnt, real_names[batch_idx], cond_scale)
            cur_img_dir = os.path.join(IMG_DIR, img_dir_name)
            os.makedirs(cur_img_dir, exist_ok=True)

            for frame_idx in range(nf):
                save_tar_img = sample_img(real_vids[:, :, frame_idx], batch_idx)
                save_real_out_img = sample_img(model.real_out_vid[:, :, frame_idx], batch_idx)
                save_real_warp_img = sample_img(model.real_warped_vid[:, :, frame_idx], batch_idx)
                save_sample_out_img = sample_img(model.sample_out_vid[:, :, frame_idx], batch_idx)
                save_sample_warp_img = sample_img(model.sample_warped_vid[:, :, frame_idx], batch_idx)
                save_real_grid = grid2fig(model.real_vid_grid[batch_idx, :, frame_idx].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_fake_grid = grid2fig(model.sample_vid_grid[batch_idx, :, frame_idx].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_real_conf = conf2fig(model.real_vid_conf[batch_idx, :, frame_idx])
                save_fake_conf = conf2fig(model.sample_vid_conf[batch_idx, :, frame_idx])
                new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                new_im.paste(Image.fromarray(save_sample_out_img, 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(save_sample_warp_img, 'RGB'), (msk_size * 2, msk_size))
                new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                new_im_arr = np.array(new_im)
                # save frame
                new_im_name = "%03d_%04d_%s_%.2f.png" % (frame_idx, cnt, real_names[batch_idx], cond_scale)
                imageio.imsave(os.path.join(cur_img_dir, new_im_name), new_im_arr)
                new_im_list.append(new_im_arr)
            video_name = "%04d_%s_%.2f.gif" % (cnt, real_names[batch_idx], cond_scale)
            print(video_name)
            imageio.mimsave(os.path.join(CKPT_DIR, video_name), new_im_list)
            cnt += 1

        iter_end = timeit.default_timer()

        if i_iter % args.print_freq == 0:
            print('Test:[{0}/{1}]\t'
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                  .format(i_iter, NUM_ITER, batch_time=batch_time))

    end = timeit.default_timer()
    print(end - start, 'seconds')
    print(CKPT_DIR)
    print(IMG_DIR)


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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()

