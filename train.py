import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
from torchvision.transforms import Normalize

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=120, type=int, help='training images crop size')
parser.add_argument('--interpolation', default='bilinear', type=str)
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')

normalize = Normalize(mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225])
unnormalize = Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    BATCH_SIZE = opt.batch_size
    INTERPOLATION = opt.interpolation

    train_set = TrainDatasetFromFolder('../data/small/jpgs', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, interpolation=INTERPOLATION)
    val_set = ValDatasetFromFolder('../testing/nowe', upscale_factor=UPSCALE_FACTOR, interpolation=INTERPOLATION)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(16,UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    mse_loss = nn.MSELoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        mse_loss = mse_loss.cuda()
    
    adversarial_criterion = nn.BCELoss()

    optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    
    for epoch in range(3):
        train_bar = tqdm(train_loader)
        netG.train()
        saved = False
        mean_mse = 0
        for data, target in train_bar:
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            #     ###
            #     target_real = Variable(torch.rand(batch_size)*0.5 + 0.7).cuda()
            #     target_fake = Variable(torch.rand(batch_size)*0.3).cuda()
            #     ###
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)
            netG.zero_grad()
            if torch.cuda.is_available():
                fake_img = fake_img.cuda()
                target = target.cuda()
            loss = mse_loss(fake_img, target)
            loss.backward()
            mean_mse += loss.item()
            optimizerG.step()
            # if epoch == 2 and saved == False:
            #     utils.save_image(real_img, './output/preHR.jpg')
            #     utils.save_image(fake_img, './output/preSR.jpg')
            #     utils.save_image(z, './output/preLR.jpg')
            #     for i in range(len(fake_img.data)):
            #         fake_img.data[i] = unnormalize(fake_img.data[i])
            #         z.data[i] = unnormalize(z.data[i])
            #         real_img.data[i] = unnormalize(target.data[i])
            #     utils.save_image(real_img, './output/unpreHR.jpg')
            #     utils.save_image(fake_img, './output/unpreSR.jpg')
            #     utils.save_image(z, './output/unpreLR.jpg')
            #     saved = True
            #     #print(loss)
        print(mean_mse / 159)

    optimizerG = optim.Adam(netG.parameters(), lr=0.00005)
    optimizerD = optim.Adam(netD.parameters(), lr=0.00001)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            #     ###
            #     target_real = Variable(torch.rand(batch_size)*0.5 + 0.7).cuda()
            #     target_fake = Variable(torch.rand(batch_size)*0.3).cuda()
            #     ###
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)
    
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            ###
            #real_out = netD(real_img)
            #fake_out = netD(fake_img)
            #d_loss = adversarial_criterion(real_out, target_real) + \
            #                 adversarial_criterion(fake_out, target_fake)
            #real_out = real_out.mean()
            #fake_out = fake_out.mean()
            ###
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss 
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            
            #fake_img = netG(z)
            #fake_out = netD(fake_img).mean()
            
            
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            #running_results['d_score'] += real_out.mean() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            #running_results['g_score'] += fake_out.mean() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
            del real_out, fake_out
    
        netG.eval()
        out_path = 'training_results/SRF_' + 'norm' + str(UPSCALE_FACTOR) + '/'
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
                    val_images.extend(
                        #[display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                        #display_transform()(sr.data.cpu().squeeze(0))])
                        [val_hr_restore.squeeze(0), hr.data.cpu().squeeze(0),
                        sr.data.cpu().squeeze(0)])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + '%s_epoch_%d_index_%d_.png' % (INTERPOLATION, epoch, index), padding=5)
                index += 1
            del val_images, val_bar, val_save_bar
        # save model parameters
        #print(epoch)
        torch.save(netG.state_dict(), 'epochs/netG_epoch_norm_%s_%d_%d.pth' % (INTERPOLATION, UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_norm_%s_%d_%d.pth' % (INTERPOLATION, UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + INTERPOLATION + 'norm' + '_train_results.csv', index_label='Epoch')
