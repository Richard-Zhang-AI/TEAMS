#!/usr/bin/env python3

import argparse
import os
import sys
import torch
import importlib
from termcolor import colored



def make_network(cfg):

    from lib.networks.make_network import make_network as create_network
    return create_network(cfg)

def make_optimizer(cfg, network):
 
    if cfg.train.optim == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr)
    elif cfg.train.optim == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=cfg.train.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr)
    return optimizer

def make_lr_scheduler(cfg, optimizer):

    from torch.optim.lr_scheduler import MultiStepLR
    return MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)

def make_recorder(cfg):

    from lib.train.recorder import make_recorder
    return make_recorder(cfg)

def load_model(network, optimizer, scheduler, recorder, model_dir, resume=True, epoch=-1):
    if not resume:
        # os.system('rm -rf {}'.format(model_dir))
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!!', 'red'))
        return 0

    pths = []
    for pth in os.listdir(model_dir):
        if pth.endswith('.pth'):
            pths.append(int(pth.split('.')[0]))
            
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    print(f'📦 Loading checkpoint: {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
        print("✅ Found 'state_dict' in checkpoint")
    elif 'net' in checkpoint:
        pretrained_state_dict = checkpoint['net']
        print("✅ Found 'net' in checkpoint")
    else:
        print("⚠️  No 'state_dict' or 'net' found, using entire checkpoint")
        pretrained_state_dict = checkpoint
 
    current_state_dict = network.state_dict()
    

    matched_weights = {}
    missing_keys = []
    shape_mismatched = []
    
  
    component_stats = {
        'yolo': {'matched': 0, 'missed': 0, 'shape_mismatch': 0},
        'cnn_proj': {'matched': 0, 'missed': 0, 'shape_mismatch': 0},
        'gcn': {'matched': 0, 'missed': 0, 'shape_mismatch': 0},
        'clinical_bert': {'matched': 0, 'missed': 0, 'shape_mismatch': 0},
        'bert_dim_reduction': {'matched': 0, 'missed': 0, 'shape_mismatch': 0},
        'other': {'matched': 0, 'missed': 0, 'shape_mismatch': 0}
    }
    
    print(f"🔍 Analyzing {len(pretrained_state_dict)} pretrained parameters...")
    
    for name, param in pretrained_state_dict.items():
        if name in current_state_dict:
            if param.shape == current_state_dict[name].shape:
                matched_weights[name] = param
                
                # 统计组件匹配情况
                if name.startswith('yolo.'):
                    component_stats['yolo']['matched'] += 1
                elif name.startswith('cnn_proj'):
                    component_stats['cnn_proj']['matched'] += 1
                elif name.startswith('gcn'):
                    component_stats['gcn']['matched'] += 1
                elif name.startswith('clinical_bert'):
                    component_stats['clinical_bert']['matched'] += 1
                elif name.startswith('bert_dim_reduction'):
                    component_stats['bert_dim_reduction']['matched'] += 1
                else:
                    component_stats['other']['matched'] += 1
            else:
                shape_mismatched.append(f"{name}: {param.shape} -> {current_state_dict[name].shape}")
                
       
                if name.startswith('yolo.'):
                    component_stats['yolo']['shape_mismatch'] += 1
                elif name.startswith('cnn_proj'):
                    component_stats['cnn_proj']['shape_mismatch'] += 1
                elif name.startswith('gcn'):
                    component_stats['gcn']['shape_mismatch'] += 1
                elif name.startswith('clinical_bert'):
                    component_stats['clinical_bert']['shape_mismatch'] += 1
                elif name.startswith('bert_dim_reduction'):
                    component_stats['bert_dim_reduction']['shape_mismatch'] += 1
                else:
                    component_stats['other']['shape_mismatch'] += 1
        else:
            missing_keys.append(name)
            
    
            if name.startswith('yolo.'):
                component_stats['yolo']['missed'] += 1
            elif name.startswith('cnn_proj'):
                component_stats['cnn_proj']['missed'] += 1
            elif name.startswith('gcn'):
                component_stats['gcn']['missed'] += 1
            elif name.startswith('clinical_bert'):
                component_stats['clinical_bert']['missed'] += 1
            elif name.startswith('bert_dim_reduction'):
                component_stats['bert_dim_reduction']['missed'] += 1
            else:
                component_stats['other']['missed'] += 1
    

    if matched_weights:
        print(f"✅ Loading {len(matched_weights)} matched parameters...")
        network.load_state_dict(matched_weights, strict=False)
        print(f"✅ Weights loaded successfully")
    else:
        print("⚠️  No matched weights found!")
    

    print(f"\n📊 Component-wise Loading Statistics:")
    for component, stats in component_stats.items():
        if stats['matched'] > 0 or stats['missed'] > 0 or stats['shape_mismatch'] > 0:
            print(f"  🔹 {component.upper():8s}: "
                  f"✅{stats['matched']:3d} "
                  f"❌{stats['missed']:3d} "
                  f"⚠️ {stats['shape_mismatch']:3d}")
    
    total_matched = sum(s['matched'] for s in component_stats.values())
    total_missed = sum(s['missed'] for s in component_stats.values())
    total_mismatched = sum(s['shape_mismatch'] for s in component_stats.values())
    
    print(f"\n📈 Summary:")
    print(f"  ✅ Matched: {total_matched}")
    print(f"  ❌ Missing: {total_missed}")
    print(f"  ⚠️  Shape mismatch: {total_mismatched}")
    print(f"  📊 Success rate: {total_matched/(total_matched+total_missed+total_mismatched)*100:.1f}%")
    
  
    if shape_mismatched and len(shape_mismatched) <= 5:
        print(f"\n⚠️  Shape mismatches:")
        for mismatch in shape_mismatched:
            print(f"    - {mismatch}")
    elif shape_mismatched:
        print(f"\n⚠️  Shape mismatches ({len(shape_mismatched)} total):")
        for mismatch in shape_mismatched[:3]:
            print(f"    - {mismatch}")
        print(f"    ... and {len(shape_mismatched) - 3} more")
    

    try:
        if 'optim' in checkpoint:
            optimizer.load_state_dict(checkpoint['optim'])
            print("✅ Optimizer state loaded")
    except Exception as e:
        print(f"⚠️  Failed to load optimizer state: {e}")
    
    try:
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("✅ Scheduler state loaded")
    except Exception as e:
        print(f"⚠️  Failed to load scheduler state: {e}")
    
    try:
        if 'recorder' in checkpoint:
            recorder.load_state_dict(checkpoint['recorder'])
            print("✅ Recorder state loaded")
    except Exception as e:
        print(f"⚠️  Failed to load recorder state: {e}")
    
    begin_epoch = 0
    if 'epoch' in checkpoint:
        begin_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming from epoch {begin_epoch}")
    
    return begin_epoch

def save_model(network, optimizer, scheduler, recorder, epoch, model_dir):

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{epoch}.pth")
    
    checkpoint = {
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    
    torch.save(checkpoint, model_path)
    print(f"✅ Model saved: {model_path}")

def make_data_loader(cfg, is_train=True):

    from lib.datasets import make_data_loader as make_dataloader
    return make_dataloader(cfg, is_train)

def make_trainer(network, cfg):

    from lib.train import make_trainer
    return make_trainer(cfg, network)

def train_traditional(cfg, network, trainer):
  
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    # gradient_accumulation 
    gradient_accumulation_steps = 4
    print(f"🔄 Gradient accumulation: {gradient_accumulation_steps} step")

  
    target_epoch = getattr(cfg, 'load_epoch', None)
    if target_epoch is not None and target_epoch != -1:
        epoch_to_load = target_epoch
    else:
        epoch_to_load = -1
    
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume, epoch=epoch_to_load)
    train_loader = make_data_loader(cfg, is_train=True)

    print("begin_epoch:", begin_epoch, "train.epoch:", cfg.train.epoch)
 
    log_path = os.path.join(cfg.model_dir, 'training_log.txt')
    os.makedirs(cfg.model_dir, exist_ok=True)
    log_file = open(log_path, 'a', encoding='utf-8')

    for epoch in range(begin_epoch, cfg.train.epoch):
        print(f"Epoch {epoch} ···")
        log_file.write(f"Epoch {epoch} ···\n")
        log_file.flush()  
        # if epoch > 50:
        #     break
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder, log_file=log_file)
        scheduler.step()

        if (epoch + 1) % cfg.train.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

    return network

def train(cfg, network, trainer):

    return train_traditional(cfg, network, trainer)



def main():
    parser = argparse.ArgumentParser(description='EnergySnake Training Script')
    parser.add_argument('--cfg_file', default='configs/sbd_snake.yaml', type=str)
    parser.add_argument('--type', type=str, default="")
    parser.add_argument('--det', type=str, default='')
    parser.add_argument('-f', type=str, default='')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

   
    from lib.config import cfg
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    

    if hasattr(cfg, 'gpus'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.gpus))

    torch.manual_seed(cfg.random_num)
    torch.cuda.manual_seed(cfg.random_num)
    

    print("=" * 50)
    print("🚀 Train start")
    print("=" * 50)

    network = make_network(cfg)

    trainer = make_trainer(network, cfg)

    trained_network = train(cfg, network, trainer)
    
    print("🎉 Train completion!")

if __name__ == '__main__':
    main()