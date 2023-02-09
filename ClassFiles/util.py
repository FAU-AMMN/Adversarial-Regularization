import os
#import tensorflow as tf
import torch 
import numpy as np
import fnmatch
from skimage.metrics import structural_similarity as ssim_np
from torchmetrics import StructuralSimilarityIndexMeasure

def quality(truth, recon):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for fixed images truth and reconstruction, evaluates average l2 value and ssim score
    recon = cut_image(recon)
    l2 = torch.mean(torch.sqrt(torch.sum(torch.square(truth - recon),axis=(1, 2, 3))))
    psnr = - 10 * torch.log10(torch.mean(torch.square(truth - recon)))
    amount_images = truth.shape[0]
    #ssi = 0
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ssi = ssim(recon, truth)
    
    """for k in range(amount_images):
        ssi = ssi + ssim(truth[k,...,0], cut_image(recon[k,...,0]))"""
    #ssi = ssi/amount_images
    return [l2, psnr, ssi]
    """# for fixed images truth and reconstruction, evaluates average l2 value and ssim score
    truth = truth.cpu().detach().numpy()
    recon = recon.cpu().detach().numpy()
    recon = cut_image(recon)
    l2 = np.average(np.sqrt(np.sum(np.square(truth - recon), axis = (1,2,3))))
    psnr = - 10 * np.log10(np.average(np.square(truth - recon)))
    amount_images = truth.shape[0]
    ssi = 0
    for k in range(amount_images):
        ssi = ssi + ssim(truth[k,...,0], cut_image(recon[k,...,0]))
    ssi = ssi/amount_images
    return [l2, psnr, ssi]"""

def cut_image(pic):
    # hard cut image to [0,1]
    pic = torch.maximum(pic, torch.zeros(pic.shape).cuda())
    pic = torch.minimum(pic, torch.ones(pic.shape).cuda())
    return pic
    # hard cut image to [0,1]
    """pic = np.maximum(pic, 0.0)
    pic = np.minimum(pic, 1.0)
    return pic"""

def normalize_image(pic):
    # normalizes image to average 0 and variance 1
    av = np.average(pic)
    pic = pic - av
    sigma = np.sqrt(np.average(np.square(pic)))
    pic = pic/(sigma + 1e-8)
    return pic

def scale_to_unit_intervall(pic):
    # scales image to unit interval
    min = np.amin(pic)
    pic = pic - min
    max = np.amax(pic)
    pic = pic/(max+ 1e-8)
    return pic

def create_single_folder(folder):
    # creates folder and catches error if it exists already
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass

def find(pattern, path):
    # finds all files with defined pattern in path and all of its subfolders
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name).replace("\\", "/"))
    return result

def lrelu(x):
    relu_fn = torch.nn.ReLU()
    return (relu_fn(x) - 0.1*relu_fn(-x))

def l2_norm(tensor):
    # l2 norm for a tensor in (batch, x, y, channel) format
    return np.mean(np.sqrt(np.sum(np.square(tensor), axis=(1,2,3))))

def image_l1(inputs):
    # contracts an image tensor of shape [batch, size, size, channels] to its l1 values along the size dimensions
    return torch.mean(torch.abs(inputs), dim=1)

