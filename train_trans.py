import os
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random

from configs.config_transformer import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.T_MARL import ChangeDetectorDoubleAttDyn, AddSpatialInfo
from models.transformer_decoder_cps import DynamicSpeaker
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_sequence_transformer, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss, LabelSmoothingLoss

from utils.vis_utils import visualize_att
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
# parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--visualize_every', type=int, default=10)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
# print(os.path.basename(args.cfg).replace('.yaml', ''))
# assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')



# Device configuration
use_cuda = torch.cuda.is_available()
gpu_ids = cfg.gpu_id
torch.backends.cudnn.enabled = False
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sample_dir = os.path.join(output_dir, 'eval_gen_samples')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
sample_subdir_format = '%s_samples_%d'

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = os.path.join(output_dir, 'snapshots')
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = '%s_checkpoint_%d.pt'

train_logger = Logger(cfg, output_dir, is_train=True)
val_logger = Logger(cfg, output_dir, is_train=False)

random.seed(1111)
np.random.seed(1111)
torch.manual_seed(1111)

# Create model
# Create model
change_detector = ChangeDetectorDoubleAttDyn(cfg)
if torch.cuda.device_count() > 1:
    change_detector = torch.nn.DataParallel(change_detector, device_ids=[0,1])
change_detector.to(device)

speaker = DynamicSpeaker(cfg)
if torch.cuda.device_count() > 1:
    speaker = torch.nn.DataParallel(speaker, device_ids=[0,1])
speaker.to(device)

spatial_info = AddSpatialInfo()
if torch.cuda.device_count() > 1:
    spatial_info = torch.nn.DataParallel(spatial_info, device_ids=[0,1])
spatial_info.to(device)

print(change_detector)
print(speaker)
print(spatial_info)

with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(speaker, file=f)
    print(spatial_info, file=f)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)

all_params = list(change_detector.parameters()) + list(speaker.parameters())
optimizer = build_optimizer(all_params, cfg)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.train.optim.step_size,
    gamma=cfg.train.optim.gamma)

# Train loop
t = 0
epoch = 0

set_mode('train', [change_detector, speaker])
# ss_prob = speaker.ss_prob

while t < cfg.train.max_iter:
    epoch += 1
    print('Starting epoch %d' % epoch)
    lr_scheduler.step()
    speaker_loss_avg = AverageMeter()
    speaker_pos_loss_avg = AverageMeter()
    contra_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()
    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()

        d_feats, nsc_feats, sc_feats, \
        labels, labels_with_ignore, no_chg_labels, no_chg_labels_with_ignore, \
        sc_pos_labels, sc_pos_labels_with_ignore, nsc_pos_labels, nsc_pos_labels_with_ignore,\
        masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
        d_img_paths, nsc_img_paths, sc_img_paths = batch

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        labels_with_ignore = labels_with_ignore.squeeze(1)
        no_chg_labels = no_chg_labels.squeeze(1)
        no_chg_labels_with_ignore = no_chg_labels_with_ignore.squeeze(1)

        sc_pos_labels = sc_pos_labels.squeeze(1)
        sc_pos_labels_with_ignore = sc_pos_labels_with_ignore.squeeze(1)
        nsc_pos_labels = nsc_pos_labels.squeeze(1)
        nsc_pos_labels_with_ignore = nsc_pos_labels_with_ignore.squeeze(1)
        masks = masks.squeeze(1).float()
        no_chg_masks = no_chg_masks.squeeze(1).float()

        d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
        d_feats, nsc_feats, sc_feats = \
            spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
        no_chg_labels, no_chg_labels_with_ignore, no_chg_masks = no_chg_labels.to(device), no_chg_labels_with_ignore.to(device), no_chg_masks.to(device)
        sc_pos_labels, sc_pos_labels_with_ignore = sc_pos_labels.to(device), sc_pos_labels_with_ignore.to(device)
        nsc_pos_labels, nsc_pos_labels_with_ignore = nsc_pos_labels.to(device), nsc_pos_labels_with_ignore.to(device)
        aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

        optimizer.zero_grad()

        encoder_output_pos, chg_pos_att_bef, chg_pos_att_aft, contra_pos = change_detector(d_feats, sc_feats)
        encoder_output_neg, chg_neg_att_bef, chg_neg_att_aft, contra_neg = change_detector(d_feats, nsc_feats)
        contra_pos = contra_pos.mean()
        contra_neg = contra_neg.mean()

        loss_pos, _, speaker_pos_loss_pos = speaker.module._forward(encoder_output_pos,
                                              labels, masks, labels_with_ignore=labels_with_ignore, pos_labels_with_ignore=sc_pos_labels_with_ignore)

        loss_neg, _, speaker_pos_loss_neg = speaker.module._forward(encoder_output_neg,
                                              no_chg_labels, no_chg_masks, labels_with_ignore=no_chg_labels_with_ignore, pos_labels_with_ignore=nsc_pos_labels_with_ignore)

        # speaker_loss = 0.5 * lang_criterion(speaker_output_pos[:, 1:], labels[:, 1:], masks[:, 1:]) + \
        #                0.5 * lang_criterion(speaker_output_neg[:, 1:], no_chg_labels[:, 1:], no_chg_masks[:, 1:])
        att_sum = (chg_pos_att_bef.sum() + chg_pos_att_aft.sum()) / (2 * batch_size)
        speaker_loss = 0.5 * loss_pos + 0.5 * loss_neg
        speaker_loss_val = speaker_loss.item()

        contra_loss = 0.5 * contra_pos + 0.5 * contra_neg
        contra_loss_val = contra_loss.item()
        # contra_loss.detach_().requires_grad_(True)

        speaker_pos_loss = 0.5 * speaker_pos_loss_pos + 0.5 * speaker_pos_loss_neg
        speaker_pos_loss_val = speaker_pos_loss.item()

        total_loss = speaker_loss + 0.4 * speaker_pos_loss + 2.5e-03 * att_sum + 0.1 * contra_loss #pos0.4, contra0.1
        total_loss_val = total_loss.item()

        speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
        contra_loss_avg.update(contra_loss_val, 2 * batch_size)
        speaker_pos_loss_avg.update(speaker_pos_loss_val, 2 * batch_size)
        total_loss_avg.update(total_loss_val, 2 * batch_size)

        stats = {}

        stats['speaker_loss'] = speaker_loss_val
        stats['avg_speaker_loss'] = speaker_loss_avg.avg
        stats['contra_loss'] = contra_loss_val
        stats['avg_contra_loss'] = contra_loss_avg.avg
        stats['speaker_pos_loss'] = speaker_pos_loss_val
        stats['avg_speaker_pos_loss'] = speaker_pos_loss_avg.avg
        stats['total_loss'] = total_loss_val
        stats['avg_total_loss'] = total_loss_avg.avg

        #results, sample_logprobs = model(d_feats, q_feats, labels, cfg=cfg, mode='sample')
        # total_loss.detach_().requires_grad_(True)
        total_loss.backward()
        if cfg.train.grad_clip != -1.0:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(change_detector.parameters(), cfg.train.grad_clip)
            nn.utils.clip_grad_norm_(speaker.parameters(), cfg.train.grad_clip)

        optimizer.step()
        # lr_scheduler.step()

        iter_end_time = time.time() - iter_start_time

        t += 1

        if t % cfg.train.log_interval == 0:
            train_logger.print_current_stats(epoch, i, t, stats, iter_end_time)
            train_logger.plot_current_stats(
                epoch,
                float(i * batch_size) / train_size, stats, 'loss')

        if t % cfg.train.snapshot_interval == 0:
            speaker_state = speaker.module.state_dict()
            chg_det_state = change_detector.module.state_dict()
            checkpoint = {
                'change_detector_state': chg_det_state,
                'speaker_state': speaker_state,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % (exp_name, t))
            save_checkpoint(checkpoint, save_path)

            print('Running eval at iter %d' % t)
            set_mode('eval', [change_detector, speaker])
            with torch.no_grad():
                test_iter_start_time = time.time()

                idx_to_word = train_dataset.get_idx_to_word()

                if args.visualize:
                    sample_subdir_path = sample_subdir_format % (exp_name, t)
                    sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
                    if not os.path.exists(sample_save_dir):
                        os.makedirs(sample_save_dir)
                sent_subdir_path = sent_subdir_format % (exp_name, t)
                sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
                if not os.path.exists(sent_save_dir):
                    os.makedirs(sent_save_dir)


                result_sents_pos = {}
                result_sents_neg = {}
                for val_i, val_batch in enumerate(val_loader):
                    d_feats, nsc_feats, sc_feats, \
                    labels, labels_with_ignore, no_chg_labels, no_chg_labels_with_ignore, masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
                    sc_pos_labels, sc_pos_labels_with_ignore, nsc_pos_labels, nsc_pos_labels_with_ignore, \
                    d_img_paths, nsc_img_paths, sc_img_paths = val_batch

                    val_batch_size = d_feats.size(0)

                    d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
                    d_feats, nsc_feats, sc_feats = \
                        spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
                    labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
                    no_chg_labels, no_chg_labels_with_ignore, no_chg_masks = no_chg_labels.to(device), no_chg_labels_with_ignore.to(device), no_chg_masks.to(device)
                    sc_pos_labels, sc_pos_labels_with_ignore = sc_pos_labels.to(device), sc_pos_labels_with_ignore.to(
                        device)
                    nsc_pos_labels, nsc_pos_labels_with_ignore = nsc_pos_labels.to(
                        device), nsc_pos_labels_with_ignore.to(device)
                    aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

                    encoder_output_pos, chg_pos_att_bef, chg_pos_att_aft, _ = change_detector(d_feats, sc_feats)
                    encoder_output_neg, chg_neg_att_bef, chg_neg_att_aft, _ = change_detector(d_feats, nsc_feats)


                    speaker_output_pos = speaker.module.sample(encoder_output_pos)


                    speaker_output_neg = speaker.module.sample(encoder_output_neg)


                    gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:]) # no start
                    gen_sents_neg = decode_sequence_transformer(idx_to_word, speaker_output_neg[:, 1:])

                    for val_j in range(speaker_output_pos.size(0)):
                        gts = decode_sequence_transformer(idx_to_word, labels[val_j][:, 1:])
                        gts_neg = decode_sequence_transformer(idx_to_word, no_chg_labels[val_j][:, 1:])
                        if args.visualize and val_j % args.visualize_every == 0:
                            visualize_att(d_img_paths[val_j], sc_img_paths[val_j],
                                          chg_pos_att_bef[val_j], dummy[val_j], chg_pos_att_aft[val_j],
                                          pos_dynamic_atts[val_j], gen_sents_pos[val_j], gts,
                                          -1, -1, sample_save_dir, 'sc_')
                            visualize_att(d_img_paths[val_j], nsc_img_paths[val_j],
                                          chg_neg_att_bef[val_j], dummy[val_j], chg_neg_att_aft[val_j],
                                          neg_dynamic_atts[val_j], gen_sents_neg[val_j], gts_neg,
                                          -1, -1, sample_save_dir, 'nsc_')
                        sent_pos = gen_sents_pos[val_j]
                        sent_neg = gen_sents_neg[val_j]
                        image_id = d_img_paths[val_j].split('_')[-1]
                        result_sents_pos[image_id] = sent_pos
                        result_sents_neg[image_id + '_n'] = sent_neg
                        message = '%s results:\n' % d_img_paths[val_j]
                        message += '\t' + sent_pos + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts:
                            message += gt + '\n'
                        message += '===================================\n'
                        message += '%s results:\n' % nsc_img_paths[val_j]
                        message += '\t' + sent_neg + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts_neg:
                            message += gt + '\n'
                        message += '===================================\n'
                        print(message)


                test_iter_end_time = time.time() - test_iter_start_time
                result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')
                result_save_path_neg = os.path.join(sent_save_dir, 'nsc_results.json')
                coco_gen_format_save(result_sents_pos, result_save_path_pos)
                coco_gen_format_save(result_sents_neg, result_save_path_neg)

            set_mode('train', [change_detector, speaker])
