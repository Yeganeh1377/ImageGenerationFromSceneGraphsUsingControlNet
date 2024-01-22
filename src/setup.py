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
from PIL import Image

#import model
sys.path.append("./sg2im")
from sg2im.model import Sg2ImModel # this imports Sg2ImModel fkc in ./sg2im/sg2im/model.py
# vgdataset
#from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
import importlib
from sg2im.data import vg
#importlib.reload(vg)

#image retransformation
from sg2im.data.utils import imagenet_deprocess_batch, imagenet_preprocess, Resize
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag

# visualise sg for inference phase of controlnet
import sg2im.vis as vis

VG_DIR = os.path.expanduser('/scratch/s194258/datasets/vg')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

parser.add_argument('--checkpoint', default='sg2im/scripts/sg2im-models/vg64.pt')
#parser.add_argument('--checkpoint', default='sg2im/scripts/sg2im-models/vg128.pt')
#parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

# experiment setup
parser.add_argument('--batch_size', default=32, type=int) # default of Justin for testing
parser.add_argument('--train_loader', default=False, type=bool_flag)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple) # must have same size as the pretrained models input
#parser.add_argument('--image_size', default='128,128', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=False, type=bool_flag) # I changed
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# inputs
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'ADE/train.h5')) # ade2 if smallest object 70
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'ADE/test.h5'))
#parser.add_argument('--test_h5', default=os.path.join(VG_DIR, 'ADE_v2/test.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab_ade.json'))
#parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--ade_color_json', default=os.path.join(VG_DIR, 'ade_colors.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)
parser.add_argument('--images_json',
    default=os.path.join(VG_DIR, 'image_data.json'))

parser.add_argument('--vocab_to_model_pred_json',
    default=os.path.join(VG_DIR, 'vocab_to_model_pred.json'))
parser.add_argument('--vocab_to_model_obj_json',
    default=os.path.join(VG_DIR, 'vocab_to_model_obj.json'))

#output options
parser.add_argument('--segment_dir', default= os.path.join(VG_DIR, 'source_v5'))
parser.add_argument('--sg2im_dir', default= os.path.join(VG_DIR, 'sg2im_out_v5'))
parser.add_argument('--prompts', default= os.path.join(VG_DIR, 'promptsv5.txt'))
parser.add_argument('--prompts_control', default= os.path.join(VG_DIR, 'prompts_controlv5.txt'))
parser.add_argument('--ids', default= os.path.join(VG_DIR, 'idsv5.npy'))

def get_image_paths(image_id_to_image, image_ids):
  paths = []
  for image_id in image_ids:
    image = image_id_to_image[image_id]
    base, filename = os.path.split(image['url'])
    path = os.path.join(os.path.basename(base), filename)
    paths.append(path)
  return paths
    
#with open(args.images_json, 'r') as f:
#    images = json.load(f)
#image_id_to_image = {i['image_id']: i for i in images}
#paths = get_image_paths(image_id_to_image, data["image_ids"])  

class VgSceneGraphDataset(Dataset):
  def __init__(self, vocab, vocab_to_model_obj, vocab_to_model_pred, h5_path, image_dir, image_size=(256, 256),
               normalize_images=True, max_objects=10, max_samples=None,
               include_relationships=True, use_orphaned_objects=True):
    super(VgSceneGraphDataset, self).__init__()

    self.image_dir = image_dir
    self.image_size = image_size
    self.vocab = vocab
    self.vocab_to_model_obj = vocab_to_model_obj
    self.vocab_to_model_pred = vocab_to_model_pred
    self.num_objects = len(vocab['object_idx_to_name'])
    self.use_orphaned_objects = use_orphaned_objects
    self.max_objects = max_objects
    self.max_samples = max_samples
    self.include_relationships = include_relationships

    transform = [Resize(image_size), T.ToTensor()]
    if normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)

    self.data = {}
    with h5py.File(h5_path, 'r') as f:
      for k, v in f.items():
        if k == 'image_paths':
          self.image_paths = list(v)
        else:
          self.data[k] = torch.IntTensor(np.asarray(v))

  def __len__(self):
    num = self.data['object_names'].size(0) # # images
    if self.max_samples is not None:
      return min(self.max_samples, num)
    return num

  def __getitem__(self, index):
    """
    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (O,)
    - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
    - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    img_path = os.path.join(self.image_dir,  str(self.image_paths[index], 'utf-8'))
    img_id = self.data["image_ids"][index].item()

    with open(img_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    H, W = self.image_size

    # Figure out which objects appear in relationships and which don't
    obj_idxs_with_rels = set()
    obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
    for r_idx in range(self.data['relationships_per_image'][index]):
      s = self.data['relationship_subjects'][index, r_idx].item()
      o = self.data['relationship_objects'][index, r_idx].item()
      obj_idxs_with_rels.add(s)
      obj_idxs_with_rels.add(o)
      obj_idxs_without_rels.discard(s)
      obj_idxs_without_rels.discard(o)

    obj_idxs = list(obj_idxs_with_rels)
    obj_idxs_without_rels = list(obj_idxs_without_rels)
    if len(obj_idxs) > self.max_objects - 1:
      obj_idxs = random.sample(obj_idxs, self.max_objects)
    if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
      num_to_add = self.max_objects - 1 - len(obj_idxs)
      num_to_add = min(num_to_add, len(obj_idxs_without_rels))
      obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
    O = len(obj_idxs) + 1

    objs = torch.LongTensor(O).fill_(-1)
    objs_model = torch.LongTensor(O).fill_(-1)

      
    boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
    obj_idx_mapping = {}
    for i, obj_idx in enumerate(obj_idxs):
      objs[i] = self.data['object_names'][index, obj_idx].item()
      objs_model[i] = self.vocab_to_model_obj[str(objs[i].item())]
      x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
      x0 = float(x) / WW
      y0 = float(y) / HH
      x1 = float(x + w) / WW
      y1 = float(y + h) / HH
      boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
      obj_idx_mapping[obj_idx] = i

    # The last object will be the special __image__ object
    objs[O - 1] = self.vocab['object_name_to_idx']['__image__']
    objs_model[O - 1] = self.vocab_to_model_obj[str(objs[O - 1].item())]

        
    triples = []
    model_triples = []
    for r_idx in range(self.data['relationships_per_image'][index].item()):
      if not self.include_relationships:
        break
      s = self.data['relationship_subjects'][index, r_idx].item()
      p = self.data['relationship_predicates'][index, r_idx].item()
      p_model = self.vocab_to_model_pred[str(p)]
      o = self.data['relationship_objects'][index, r_idx].item()
      s = obj_idx_mapping.get(s, None)
      o = obj_idx_mapping.get(o, None)
      if s is not None and o is not None:
        triples.append([s, p, o])
        model_triples.append([s, p_model, o])

    # Add dummy __in_image__ relationships for all objects
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    in_image_model = self.vocab_to_model_pred[str(in_image)]
    for i in range(O - 1):
      triples.append([i, in_image, O - 1])
      model_triples.append([i, in_image_model, O - 1])

    triples = torch.LongTensor(triples)
    model_triples = torch.LongTensor(model_triples)
    img_id = torch.LongTensor([img_id])
    return img_id, image, objs, objs_model, boxes, triples, model_triples


def vg_collate_fn(batch):
  """
  Collate function to be used when wrapping a VgSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving categories for all objects
  - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
  - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
  """
  # batch is a list, and each element is (image, objs, boxes, triples) 
  all_imgs, all_ids, all_objs, all_objs_model, all_boxes, all_triples, all_triples_model = [], [], [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0
  
  for i, (id, img, objs, objs_model, boxes, triples, triples_model) in enumerate(batch):
    all_imgs.append(img[None])
    all_ids.append(id[None])
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_objs_model.append(objs_model)
    all_boxes.append(boxes)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    triples_model = triples_model.clone()
    triples_model[:, 0] += obj_offset
    triples_model[:, 2] += obj_offset
    all_triples_model.append(triples_model)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O

  all_imgs = torch.cat(all_imgs)
  all_ids = torch.cat(all_ids)
  all_objs = torch.cat(all_objs)
  all_objs_model = torch.cat(all_objs_model)
  all_boxes = torch.cat(all_boxes)
  all_triples = torch.cat(all_triples)
  all_triples_model = torch.cat(all_triples_model)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_ids.squeeze(), all_imgs, all_objs, all_objs_model, all_boxes, all_triples, all_triples_model,
         all_obj_to_img, all_triple_to_img)
  return out

def build_vg_dsets(args):
    # for the sake of complecity we have both train and val splits but only use train in our experimenta
    #load vocab related files
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    with open(args.vocab_to_model_obj_json, 'r') as f:
        vocab_to_model_obj = json.load(f)
    
    with open(args.vocab_to_model_pred_json, 'r') as f:
        vocab_to_model_pred = json.load(f)
  
        
    #vocab = model.vocab
    dset_kwargs = {
    'vocab': vocab,
    'vocab_to_model_obj': vocab_to_model_obj, 
    'vocab_to_model_pred': vocab_to_model_pred, 
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
    }
    dset_kwargs['h5_path'] = args.val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(val_dset) // args.batch_size
    print('There are %d iterations per epoch in valset' % iter_per_epoch)

    #dset_kwargs['h5_path'] = args.test_h5
    #test_dset = VgSceneGraphDataset(**dset_kwargs)
    #iter_per_epoch = len(test_dset) // args.batch_size
    #print('There are %d iterations per epoch in test set' % iter_per_epoch)
    return vocab, val_dset#, test_dset

def build_loaders(args):
    #if args.dataset == 'vg':
    vocab, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn
    #elif args.dataset == 'coco':
    #    vocab, train_dset, val_dset = build_coco_dsets(args)
    #    collate_fn = vg.coco_collate_fn
    
    loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': False, # I changed: Train loader
    'collate_fn': collate_fn,
    }
    #train_loader = DataLoader(train_dset, **loader_kwargs)
    
    #loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    #test_loader = DataLoader(test_dset, **loader_kwargs)
    
    return vocab, val_loader

def build_segment_map(args, vocab, grouped_objs, grouped_masks, ids, imgs_proc_all): 
    # loop over each image
    for g, group in enumerate(grouped_masks):
        
        # get semantic map   
        semantic_map = np.zeros((512, 512))
        components = []
        areas = []
        #summed_mask = np.zeros((64, 64))
        #summed_mask = np.zeros((args.image_size[0], args.image_size[1]))
        #loop over objects/masks in layout of given img
        for i, mask in enumerate(group):
            obj = grouped_objs[g][i]
            obj_name = vocab['object_idx_to_name'][obj]
            mask = mask.sum(0)
            

            # for each obj in img
            cv2_image = np.expand_dims(mask, 2).astype("uint8") 
            cv2_image = cv2_image[2:int(args.image_size[0]-3), 2:int(args.image_size[1]-3)]
            cv2_image = cv2.resize(cv2_image, (512, 512)) # upsample: linear interpolation
            #cv2_image = cv2.medianBlur(cv2_image, 3) # looks like working
            cv2_image = cv2.GaussianBlur(cv2_image,(5,5),0)
            #cv2_image = cv2.blur(cv2_image, (5,5))
            _, thresholded = cv2.threshold(cv2_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #thresh.binary makes it 0,255 b/w for each mask, otsu makes it more intelligent threshold so the var of for and back peaks minimal
            _, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded)

            if (len(stats[:, cv2.CC_STAT_AREA])==1) or (obj_name == ''): # layout is black
                cond = len(stats[:, cv2.CC_STAT_AREA])==1
                #print(np.shape(stats[:, cv2.CC_STAT_AREA]))
                #print(f"{obj_name} if condition met, breaks from all the inner loops and go to next outer loop")
                #print(f"image_id with 0 layout:{ids[g]}")
                break
                
            area = stats[:, cv2.CC_STAT_AREA][1]
            components.append(labels * obj)
            areas.append(-area)
            
        if cond == True: # only works cause the last object of the image is always __image__
            print(f"{obj_name},{ids[g]}")
            continue
            
        sort_idcs = np.argsort(areas)
        for i in sort_idcs:
            img = components[i]
            nonz = np.nonzero(img) # objects
            semantic_map[nonz] = img[nonz] # loc of each component (obj) on segmentation map
        
        color_seg = np.zeros((512, 512, 3), dtype=np.uint8)
        objs = []
        for i, mask in enumerate(group):
            obj = grouped_objs[g][i]
            if obj != 0: # not __image__
                objs.append(vocab['object_idx_to_name'][obj])
                color_seg[semantic_map == obj, :] = vocab["object_idx_to_color"][obj]
                
                
        color_seg = color_seg.astype(np.uint8)
        seg_path = os.path.join(args.segment_dir, str(ids[g])+ ".png" )
        seg_img = Image.fromarray(color_seg)
        seg_img.save(seg_path) # save using PIL

        # save outputs of sg2im images
        path = os.path.join(args.sg2im_dir, str(ids[g])+ ".png")
        img = Image.fromarray(imgs_proc_all[g].transpose(1, 2, 0))
        img.save(path) # save using PIL

        # save id files of all the images without 0.0 layout
        np.save(args.ids, np.array(ids, dtype=np.uint32), allow_pickle=True)

def main(args):
    print(args,"\n")
    # setup
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    if not os.path.isdir(args.segment_dir): 
        print('Segmentation directory "%s" does not exist; creating it' % args.segment_dir)
        os.makedirs(args.segment_dir)

    if not os.path.isdir(args.sg2im_dir): 
        print('sg2im out directory "%s" does not exist; creating it' % args.sg2im_dir)
        os.makedirs(args.sg2im_dir)

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

    print("loading vocab")
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    
    # Load the model, with a bit of care in case there are no GPUs
    checkpoint = torch.load(args.checkpoint, map_location=device) # load the pretrained johnson model
    model = Sg2ImModel(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)
    print("Model Loaded!\n")

    # save mappings once only
    
    #vocab_to_model_obj = {}
    #for i in range(len(vocab['object_idx_to_name'])):
    #    obj = vocab['object_idx_to_name'][i]
    #    j = model.vocab['object_name_to_idx'][obj]
    #    vocab_to_model_obj[i] = j
    #with open(args.vocab_to_model_obj_json, 'w') as f:
    #    json.dump(vocab_to_model_obj, f)

    #vocab_to_model_pred = {}
    #for i in range(len(vocab['pred_idx_to_name'])):
    #    pred = vocab['pred_idx_to_name'][i]
    #    j = model.vocab['pred_name_to_idx'][pred]
    #    vocab_to_model_pred[i] = j
    #with open(args.vocab_to_model_pred_json, 'w') as f:
    #    json.dump(vocab_to_model_pred, f)

    # load mappings from model to vocab
    
    # build the loaders
    _ , val_loader = build_loaders(args)

    # preparing vocab for prompts
    vocab["pred_idx_to_name"][0] = ""
    vocab["object_idx_to_name"][0] = ""
    
    # to save logs for later
    grouped_boxes_all = []
    grouped_objs_all = []
    grouped_masks_all = []
    ids_all = []
    imgs_proc_all = []
    prompts_all = []
    names_all = []
    grouped_triples_names_all = []
    results = {}
    with torch.no_grad(): 
        for batch in val_loader:
            # batch is in vocab codes
            ids, imgs, objs, objs_model, boxes, triples, model_triples, obj_to_img, triple_to_img = batch
            ids, imgs, objs, objs_model, boxes, triples, model_triples, obj_to_img, triple_to_img = ids.to(device), imgs.to(device), objs.to(device), objs_model.to(device), boxes.to(device), triples.to(device),model_triples.to(device), obj_to_img.to(device), triple_to_img.to(device)
    
            #model forward
            out_imgs, boxes_pred, masks_pred, rel_scores, layout = model.forward(objs_model, model_triples, obj_to_img)
            #out_imgs, boxes_pred, masks_pred, rel_scores, layout = model.forward(objs, triples, obj_to_img)
            print(layout.size())
            # get prompts by looping over all the objects and triples + filter the objects with mask of fully 0. but not image
            triples_names = []
            batch_names = []
            for obj in objs:
                batch_names.append(vocab["object_idx_to_name"][obj.item()])
            
            for row in triples:
                o,p,s = row.tolist()
                triples_names.append(batch_names[o]+ " "+ vocab["pred_idx_to_name"][p]+" "+ batch_names[s])


            # get the grouped objects and boxes and layouts and the names and outputs and the respective ids.
            unique_indices, counts = np.unique(obj_to_img.cpu().numpy(), return_counts=True)
            unique_idx, triple_counts = np.unique(triple_to_img.cpu().numpy(), return_counts=True)
            
            batch_grouped_boxes = np.split(boxes_pred.cpu().numpy(), np.cumsum(counts)[:-1]) # number of objects
            batch_grouped_objs = np.split(objs.cpu().numpy(), np.cumsum(counts)[:-1])
            batch_grouped_masks = np.split(layout.cpu().numpy(), np.cumsum(counts)[:-1])
            
            batch_grouped_triples_names = np.split(triples_names, np.cumsum(triple_counts)[:-1])
                
            batch_grouped_triples = np.split(model_triples, np.cumsum(triple_counts)[:-1])
            batch_grouped_names = np.split(batch_names, np.cumsum(counts)[:-1])
            
            batch_ids = ids.cpu().tolist()
            batch_imgs_proc = imagenet_deprocess_batch(out_imgs.cpu())
            
            for g in range(len(batch_grouped_triples_names)):
                #print(batch_grouped_triples_names[g])
                prompt = ",".join(map(str,batch_grouped_triples_names[g]))+"\n"
                control_prompt = ",".join(map(str,batch_grouped_names[g]))+"\n"
                prompts_all.append(prompt)
                names_all.append(control_prompt)
                
            grouped_boxes_all.extend(batch_grouped_boxes)
            grouped_objs_all.extend(batch_grouped_objs)
            grouped_masks_all.extend(batch_grouped_masks)
            ids_all.extend(batch_ids)
            imgs_proc_all.extend(batch_imgs_proc.numpy())
            grouped_triples_names_all.extend(batch_grouped_triples_names)

        prompts_file = open(args.prompts, 'w')
        prompts_control_file = open(args.prompts_control, 'w')

        prompts_file.writelines(prompts_all)
        prompts_control_file.writelines(names_all)
        
        print("model statistics:")
        print(f"total_number_of_obj: {len(model.vocab['object_idx_to_name']), len(model.vocab['object_idx_to_name'])}, total_number_of_relations{len(model.vocab['pred_idx_to_name'])}") 
        # objects in vocabulary of the model
        print(f"size of ObjEmbedding: {model.obj_embeddings}, Size of Relations Embeddings: {model.pred_embeddings}\n")
        
        print("vocabulary statistics:")
        print(f"total_number_of_obj: {len(vocab['object_idx_to_name'])}, total_number_of_relations{len(vocab['pred_idx_to_name'])}\n") 
    
        print("Sizes of the dataloader Attributes (batch):")    
        print(f"ids: {ids.size()}, imgs: {imgs.size()}, objs: {objs.size()}, boxes: {boxes.size()}, triples: {triples.size()}, obj_to_img: {obj_to_img.size()}, triple_to_img: {triple_to_img.size()}\n")
    
        print("Sizes of the Output of the Model:")
        print(f"batch_imgs_proc:{batch_imgs_proc.size()}, imgs: {out_imgs.size()}, boxes_pred: {boxes_pred.size()}, masks_pred: {masks_pred.size()}, rel_scores: {rel_scores.size()}, layout: {layout.size()}\n")
    
        print("Batch Grouped Sizes(input to process Batch):")
        print(f"batch_grouped_boxes:{len(grouped_boxes_all)}, batch_grouped_objs:{len(batch_grouped_objs)}, batch_grouped_masks:{len(batch_grouped_masks)}")

    # generating prompts
    
    print("Grouped Sizes(results):")
    print(f"grouped_boxes:{len(grouped_boxes_all)},grouped_objs:{len(grouped_objs_all)},grouped_masks:{len(grouped_masks_all)},ids:{len(ids_all)},sg2im outputs:{len(imgs_proc_all)}")
    print(f"grouped_objs:{len(grouped_objs_all)},ids:{len(ids_all)},sg2im outputs:{len(imgs_proc_all)}, grouped_triples_names: {len(grouped_triples_names_all)} ")
    
    #results["grouped_boxes_all"] = grouped_boxes_all
    #results["grouped_objs_all"] = grouped_objs_all
    #results["grouped_masks_all"] = grouped_masks_all
    #results["ids_all"] = ids_all
    #results["prompts_all"] = grouped_triples_names_all
    # save results
    #with open(args.caption_pkl, 'wb') as f:
    #    pickle.dump(results, f)
    
    # generate segmentation maps and save the output of sg2im and segmentation maps   
    build_segment_map(args, vocab, grouped_objs_all, grouped_masks_all, ids_all, imgs_proc_all)    
    print("done")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    