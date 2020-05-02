import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
from torchvision.transforms import Normalize
from PIL import Image

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=60, type=int, help='training images crop size')
parser.add_argument('--interpolation', default='bilinear', type=str, help='Interpolation the model was learned for')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--generator_weights', default='./epochs/netG_epoch_norm_lanczos_4_30.pth', type=str,
                    help='path for generator net to test')


if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    INTERPOLATION = opt.interpolation

    inter_dict = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS
    }
    
    normalize = Normalize(mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225])
    unnormalize = Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

    netG = Generator(16,UPSCALE_FACTOR)

    if opt.generator_weights != '':
        netG.load_state_dict(torch.load(opt.generator_weights))
        print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    results = {'base':[], 'interpolation':[], 'psnr': [], 'ssim': []}

    for trans in inter_dict:

        val_set = ValDatasetFromFolder('../testing/small/testing', upscale_factor=UPSCALE_FACTOR, interpolation=trans)
        val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
        
        generator_criterion = GeneratorLoss()
        
        if torch.cuda.is_available():
            netG.cuda()
        #    generator_criterion.cuda()
            
        netG.eval()
        out_path = 'valing_results/SRF_' + str(UPSCALE_FACTOR) + INTERPOLATION + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            n_batch=0
            for val_lr, val_hr_restore, val_hr in val_bar:
                n_batch+=1
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr 
                for k, v in enumerate(lr):
                    lr[k] = normalize(v)
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
                for k, v in enumerate(sr):
                    sr[k] = unnormalize(v)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
                if n_batch < 20:
                    for k, v in enumerate(hr):
                        hr[k] = normalize(v)
                    if torch.cuda.is_available():
                        hr = hr.cuda()
                        sr_hr = netG(hr)
                    for k, v in enumerate(sr_hr):
                        sr_hr[k] = unnormalize(v)
                        hr[k] = unnormalize(hr[k])
                    val_images.extend(
                        #[display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                        #display_transform()(sr.data.cpu().squeeze(0))])
                        [val_hr_restore.squeeze(0), hr.data.cpu().squeeze(0),
                        sr.data.cpu().squeeze(0)])
                    utils.save_image(sr_hr, out_path + '%s_unscalled.png' % (INTERPOLATION), padding=5)

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + '%s_base_%s_.png' % (INTERPOLATION), padding=5)
            del val_images, val_bar, val_save_bar

        # save loss\scores\psnr\ssim
        results['base'].append(INTERPOLATION)
        results['interpolation'].append(trans)
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'base': results['base'], 'interpolation': results['interpolation'], 
                'PSNR': results['psnr'], 'SSIM': results['ssim']})
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + INTERPOLATION + '_val_results.csv')

    
