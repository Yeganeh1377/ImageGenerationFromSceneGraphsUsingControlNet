import os
import fnmatch2 as fnmatch
import torch, torchvision
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler, StableDiffusionPipeline#, transformers, accelerate, wandb
import matplotlib
from PIL import Image
import accelerate #extra for now

# from input notebook
import argparse, json, os, functools, random, sys, cv2, h5py, pickle
from imageio import imwrite
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# torch and image modules
import torch, torchvision
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import PIL

#import preprocesses from sg2im
sys.path.append("./sg2im")
#from sg2im.model import Sg2ImModel # this imports Sg2ImModel fkc in ./sg2im/sg2im/model.py
# vgdataset
#from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn

#image retransformation
from sg2im.data.utils import imagenet_deprocess_batch, imagenet_preprocess, Resize
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag

# visualise sg for inference phase of controlnet
import sg2im.vis as vis

VG_DIR = os.path.expanduser('/scratch/s194258/datasets/vg')

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])
# models paths
parser.add_argument('--checkpoint_control', default="lllyasviel/control_v11p_sd15_seg")
parser.add_argument('--checkpoint', default="runwayml/stable-diffusion-v1-5")
#parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

# experiment setup
parser.add_argument('--batch_size', default=1, type=int) # default of Justin for testing is 3
parser.add_argument('--train_loader', default=False, type=bool_flag)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--iters', default=20, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple) # must have same size as the pretrained models input
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=2, type=int)
parser.add_argument('--shuffle_val', default=False, type=bool_flag) # I changed
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# inputs
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images')) 
#parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'ADE/train.h5'))
#parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
#parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'ADE/test.h5'))
#parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab_ade.json'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'ADE/train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'ADE/val.h5'))
parser.add_argument('--test_h5', default=os.path.join(VG_DIR, 'ADE/test.h5'))
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

parser.add_argument('--segment_dir', default= os.path.join(VG_DIR, 'source_v5'))
parser.add_argument('--sg2im_dir', default= os.path.join(VG_DIR, 'sg2im_out_v5'))

parser.add_argument('--prompts', default= os.path.join(VG_DIR, 'promptsv5.txt'))
parser.add_argument('--prompts_control', default= os.path.join(VG_DIR, 'prompts_controlv5.txt'))
parser.add_argument('--ids', default= os.path.join(VG_DIR, 'idsv5.npy'))

# output options
parser.add_argument('--control_dir', default= os.path.join(VG_DIR, 'control_out_v5'))
parser.add_argument('--sd_dir', default= os.path.join(VG_DIR, 'sd_out_v5'))

#parser.add_argument('--caption_pkl', default= os.path.join(VG_DIR, 'caption.pkl'))

def main(args):
    print(args,"\n")
    # setup
    
    if not os.path.isdir(args.control_dir): 
        print('Control_out directory "%s" does not exist; creating it' % args.control_dir)
        os.makedirs(args.control_dir)

    if not os.path.isdir(args.sd_dir): 
        print('SD out directory "%s" does not exist; creating it' % args.sd_dir)
        os.makedirs(args.sd_dir)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda')
    if not torch.cuda.is_available():
        print('WARNING: CUDA not available; falling back to CPU')
        device = torch.device('cpu')
    print(f"loading model to {device}!")

    # load img_id mappings
    with open(args.images_json, 'r') as f:
       images = json.load(f)
    image_id_to_image = {i['image_id']: i for i in images}

    ########################### load the pipeline #########################
    controlnet = ControlNetModel.from_pretrained(args.checkpoint_control, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(args.checkpoint, controlnet=controlnet, torch_dtype=torch.float16)
    #pipe = pipe.to("cuda")
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config) # frank  use this, best tradeoff, fastest
    #pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) # decide which scheduler must use: paper and tutorial uses this
    # this command load individual models per demand
    pipe.enable_model_cpu_offload()
    #generator = torch.Generator(device=device).manual_seed(args.seed)
    generator = torch.manual_seed(args.seed)
    ########################## load control images #######################
    img_paths = {}
    im_folder = args.segment_dir
    for filename in os.listdir(im_folder):
        if filename == '.DS_Store':
            continue
        f = os.path.join(im_folder, filename)
        img_paths[filename] = f
    #print("dictionary of control images:\n", img_paths)

    ######################### load prompt file ############################
    #print(f"memory data:{torch.cuda.max_memory_allocated():,}, {torch.cuda.max_memory_reserved():,}")
    f = open(args.prompts_control, "r") 
    # reading the file 
    control_file = f.readlines() 
    ids = np.load(args.ids, allow_pickle=True)
    prompts = {}
    for i in range(len(ids)):
        prompts[str(ids[i])+".png"] = "a realistic image of "+control_file[i][:-2]   
    #print("prompts of all the test images:\n", prompts)
   
    ######################### perform inference ##########################
    for key in img_paths.keys(): # so are sure that it only considers those imgs with segment maps
        
        control_img = Image.open(img_paths[key]).convert('RGB')
        prompt = prompts[key]
        

        #for _ in range(args.num_val_samples):
        with torch.autocast("cuda"):
            #print(f"memory data{torch.cuda.max_memory_allocated():,}, {torch.cuda.max_memory_reserved():,}")
            images = pipe(prompt, image= control_img, guess_mode = True, guidance_scale = 4.0, num_images_per_prompt=args.batch_size, generator=generator, num_inference_steps=args.iters).images
            # save generated images
            for i in range(len(images)):
                img_path = os.path.join(args.control_dir, str(i)+key)
                images[i].save(img_path)
    #torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parser.parse_args()
    results = main(args)