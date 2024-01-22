import os, tqdm, argparse, json, functools, random, sys, cv2, h5py, pickle, natsort
import fnmatch2 as fnmatch
import torch, torchvision
from PIL import Image
from imageio import imwrite
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

import torch.nn as nn
import torch.nn.functional as F

#import model
sys.path.append("./sg2im")
#image retransformation
from sg2im.data.utils import imagenet_deprocess_batch, imagenet_preprocess, Resize
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag

# visualise sg for inference phase of controlnet
import sg2im.vis as vis

VG_DIR = os.path.expanduser('/scratch/s194258/datasets/vg')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])
parser.add_argument('--checkpoint_control', default="lllyasviel/control_v11p_sd15_seg")

parser.add_argument('--checkpoint', default="runwayml/stable-diffusion-v1-5")
#parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

# experiment setup
parser.add_argument('--batch_size', default=3, type=int) # default of Justin for testing
parser.add_argument('--train_loader', default=False, type=bool_flag)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple) # must have same size as the pretrained models input
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_vis_samples', default=6, type=int)
parser.add_argument('--shuffle_val', default=False, type=bool_flag) # I changed
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# inputs
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'ADE/train.h5'))
#parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'ADE/test.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab_ade.json'))
parser.add_argument('--ade_color_json', default=os.path.join(VG_DIR, 'ade_colors.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)
parser.add_argument('--images_json',
    default=os.path.join(VG_DIR, 'image_data.json'))
parser.add_argument('--vocab_to_model_pred_json',
    default=os.path.join(VG_DIR, 'vocab_to_model_pred.json'))
parser.add_argument('--vocab_to_model_obj_json',
    default=os.path.join(VG_DIR, 'vocab_to_model_obj.json'))

#data options
parser.add_argument('--prompts', default= os.path.join(VG_DIR, 'promptsv5.txt'))
parser.add_argument('--prompts_control', default= os.path.join(VG_DIR, 'prompts_controlv5.txt'))
parser.add_argument('--ids', default= os.path.join(VG_DIR, 'idsv5.npy'))

parser.add_argument('--segment_dir', default= os.path.join(VG_DIR, 'source_v5'))
parser.add_argument('--sg2im_dir', default= os.path.join(VG_DIR, 'sg2im_out_v5'))
parser.add_argument('--control_dir', default= os.path.join(VG_DIR, 'control_out_v5'))
parser.add_argument('--sd_dir', default= os.path.join(VG_DIR, 'sd_out_v5'))

# output
parser.add_argument('--results_dir', default= "results")

def load_prompts(args):
    ids = np.load(args.ids, allow_pickle=True)
    print(len(ids))
    # sd
    f = open(args.prompts, "r") 
    # reading the file 
    sd_file = f.readlines() 
    # controlnet
    f = open(args.prompts_control, "r") 
    # reading the file 
    control_file = f.readlines() 
    # make the dics of paths and prompts connected to the id number
    prompts_controlnet = {}
    prompts_sd = {}
    
    for i in range(len(ids)):
        prompts_controlnet[str(ids[i])+".png"] = control_file[i][:-2]
        prompts_sd[str(ids[i])+".png"] = sd_file[i][:-2]
    return prompts_controlnet, prompts_sd, ids

# get paths of the original images
def get_image_paths(args, prompt_dic):
    list_keys = list(prompt_dic.keys())
    with open(args.images_json, 'r') as f:
        images = json.load(f)
    image_id_to_image = {i['image_id']: i for i in images}
    paths = {}
    for key in list_keys:
        image_id = key[:-4]
        image = image_id_to_image[int(image_id)]
        base, filename = os.path.split(image['url'])
        path = os.path.join(os.path.basename(base), filename)
        paths[key] = path
    return paths

def load_imgs_vis(args):
    ids = np.load(args.ids, allow_pickle=True)

    ## load seg maps
    seg_imgs = {}
    im_folder = args.segment_dir
    for filename in os.listdir(im_folder):
        if filename == '.DS_Store':
            continue
        f = os.path.join(im_folder, filename)
        #index = filename[3:-4]
        #img = Image.open(f).convert('RGB')
        #plt.imshow(img)
        #plt.title(filename)
        #plt.show()
        seg_imgs[filename] = f
    
    ## load control outputs 
    sg2im_imgs = {}
    im_folder = args.sg2im_dir
    for filename in os.listdir(im_folder):
        if filename == '.DS_Store':
            continue
        f = os.path.join(im_folder, filename)
        #index = filename[3:-4]
        #img = Image.open(f).convert('RGB')
        #plt.imshow(img)
        #plt.title(filename)
        #plt.show()
        sg2im_imgs[filename] = f

    ## load seg maps
    control_imgs = {}
    im_folder = args.control_dir
    for filename in os.listdir(im_folder):
        if filename == '.DS_Store':
            continue
        f = os.path.join(im_folder, filename)
        index = filename[1:]
        #img = Image.open(f).convert('RGB')
        #plt.imshow(img)
        #plt.title(filename)
        #plt.show()
        control_imgs.setdefault(index, []).append(f)
    ## load seg maps
    sd_imgs = {}
    im_folder = args.sd_dir
    for filename in os.listdir(im_folder):
        if filename == '.DS_Store':
            continue
        f = os.path.join(im_folder, filename)
        index = filename[1:]
        #img = Image.open(f).convert('RGB')
        #plt.imshow(img)
        #plt.title(filename)
        #plt.show()
        sd_imgs.setdefault(index, []).append(f)
    return seg_imgs, sg2im_imgs, ids, control_imgs, sd_imgs

def get_scores(fake_dir, fid, kid, inception, device, origin=False, path_dic=None): 
    ## load fake image
    if origin:
        dataset = Origin_CustomDataSet(path_dic, fake_dir)
    else:  
        dataset = CustomDataSet(fake_dir)
    print(len(dataset))
    batch_size = 100
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    for images in tqdm.tqdm(data_loader):
        #images = next(iter(data_loader))
        #print(images.dtype, images[0,:,:,:].size())
        images = images.type(torch.uint8).to(device)
        fid.update(images, real=False)
        kid.update(images, real=False)
        inception.update(images)
    
    # get scores
    fid_mean = fid.compute()
    is_mean, is_std = inception.compute()
    kid_mean, kid_std = kid.compute()

    return fid_mean, is_mean, is_std, kid_mean, kid_std
class Origin_CustomDataSet(Dataset):
    def __init__(self, dic_paths, main_dir):
        #self.dic_paths = dic_paths
        self.main_dir = main_dir
        self.list_paths = list(dic_paths.values())
        # transform of the original images
        self.transform = Compose([
        ToTensor(),
        lambda x: x*255
        ])
        #all_imgs = os.listdir(main_dir)
        #self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.list_paths)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir,self.list_paths[idx])
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((299, 299))
        tensor_image = self.transform(image)
        return tensor_image

class CustomDataSet(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        # transform of the original images
        self.transform = Compose([
        ToTensor(),
        lambda x: x*255
        ])
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((299, 299))
        tensor_image = self.transform(image)
        return tensor_image

def main(args):

    # setup
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda')
    if not torch.cuda.is_available():
        print('WARNING: CUDA not available; falling back to CPU')
        device = torch.device('cpu')
    print(device)

    inception = InceptionScore(splits=10)
    fid = FrechetInceptionDistance()
    kid = KernelInceptionDistance(subsets=10, subset_size=100, reset_real_features=False)
    fid.to(device)
    inception.to(device)
    kid.to(device)

    # load origin data
    seg_imgs, _, _, _, _ = load_imgs_vis(args)  
    origin_paths = get_image_paths(args, seg_imgs)
    #print(origin_paths.keys())
    dataset = Origin_CustomDataSet(origin_paths, args.vg_image_dir)
    print(len(dataset))
    
    batch_size = 100
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    for images in tqdm.tqdm(data_loader):
    #images = next(iter(data_loader))
        #print(images.dtype, images[0,:,:,:].size())
        images = images.type(torch.uint8).to(device)
        #plt.imshow(images[0,:,:,:].to("cpu").permute(1, 2, 0))
        #plt.show()
        fid.update(images, real=True)
        kid.update(images, real=True)
    #inception.update(images)
    #FID_mean_real = fid.compute()
    #IS_mean_real, IS_std_real = inception.compute()

    # scores real
    fid_score_real, is_score_real, is_std_real, kid_mean_real, kid_std_real = get_scores(args.vg_image_dir, fid, kid, inception, device, True, origin_paths) 

    # scores control
    fid_score_control, is_score_control, is_std_control, kid_mean_control, kid_std_control = get_scores(args.control_dir, fid, kid, inception, device)

    # scores sd
    fid_score_sd, is_score_sd, is_std_sd, kid_mean_sd, kid_std_sd = get_scores(args.sd_dir, fid, kid, inception, device)

    # scores sg2im
    fid_score_sg2im, is_score_sg2im, is_std_sg2im, kid_mean_sg2im, kid_std_sg2im = get_scores(args.sg2im_dir, fid, kid, inception, device)

    print("FID real IMAGES:", fid_score_real.item(), "IS real IMAGES: ", is_score_real.item(),"+-", is_std_real.item(), "KID real IMAGES: ", kid_mean_real.item(),"+-", kid_std_real.item()) # doublecheck
    print("FID sg2im IMAGES:", fid_score_sg2im.item(), "IS sg2im IMAGES: ", is_score_sg2im.item(),"+-", is_std_sg2im.item(), "KID sg2im IMAGES: ", kid_mean_sg2im.item(),"+-", kid_std_sg2im.item()) # doublecheck
    print("FID controlnet IMAGES:", fid_score_control.item(), "IS controlnet IMAGES: ", is_score_control.item(),"+-", is_std_control.item(), "KID control IMAGES: ", kid_mean_control.item(),"+-", kid_std_control.item()) # doublecheck
    print("FID sd IMAGES:", fid_score_sd.item(), "IS sd IMAGES: ", is_score_sd.item(), "+-", is_std_sd.item(), "KID sd IMAGES: ", kid_mean_sd.item(),"+-", kid_std_sd.item()) # doublecheck
    list = [fid_score_real.item(), is_score_real.item(), is_std_real.item(), kid_mean_real.item(), kid_std_real.item(), 
            fid_score_sg2im.item(), is_score_sg2im.item(), is_std_sg2im.item(), kid_mean_sg2im.item(), kid_std_sg2im.item(),
            fid_score_control.item(), is_score_control.item(), is_std_control.item(), kid_mean_control.item(), kid_std_control.item(),
            fid_score_sd.item(), is_score_sd.item(), is_std_control.item(), kid_mean_sd.item(), kid_std_sd.item()
            ]
    np.save(os.path.join(args.results_dir,'quantitative_results.npy'), np.array(list))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
