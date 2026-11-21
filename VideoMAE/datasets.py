import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.cropper = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([
            Stack(roll=False), ToTorchFormatTensor(div=True), normalize,
        ])
        self.mask_ratio = args.mask_ratio
        self.motion_enabled = bool(getattr(args, 'motion_enabled', False))
        self.motion_guided_mask = bool(getattr(args, 'motion_guided_mask', False))
        self.motion_beta = float(getattr(args, 'motion_beta', 2.0))
        self.motion_alpha = float(getattr(args, 'motion_alpha', 1.0))
        self.motion_percentile = float(getattr(args, 'motion_percentile', 99.0))
        self.masked_position_generator = TubeMaskingGenerator(args.window_size, args.mask_ratio)

    def __call__(self, images, motion_maps=None):
        image_group, label = images
        if motion_maps is None:
            process_data, _ = self.transform(self.cropper((image_group, label)))
            return process_data, self.masked_position_generator()

        # Motion maps are raw RAFT magnitudes [T_tubelet,H,W], aligned to RGB.
        motion_maps = np.asarray(motion_maps, dtype=np.float32)
        if motion_maps.ndim != 3 or motion_maps.shape[0] != len(image_group)//2:
            raise ValueError(f"Motion maps must be [{len(image_group)//2},H,W]; got {motion_maps.shape}")
        w, h = image_group[0].size
        if motion_maps.shape[1:] != (h, w):
            raise ValueError(f"RGB size {(h,w)} and RAFT size {motion_maps.shape[1:]} differ; expected preprocessing alignment")
        crop_w, crop_h, off_w, off_h = self.cropper._sample_crop_size((w,h))
        cropped_rgb = [im.crop((off_w,off_h,off_w+crop_w,off_h+crop_h)).resize(
            (self.cropper.input_size[0], self.cropper.input_size[1]), Image.BILINEAR) for im in image_group]
        cropped_motion = motion_maps[:, off_h:off_h+crop_h, off_w:off_w+crop_w]
        # Motion is downsampled to the model's patch grid using area interpolation.
        motion_t = torch.from_numpy(cropped_motion).unsqueeze(1).float()
        motion_t = torch.nn.functional.interpolate(motion_t, size=(14,14), mode='area').squeeze(1)
        motion_t = motion_t.numpy()
        # Robust clip-level normalization after crop and resize; constant/near-static -> zeros.
        hi = float(np.percentile(motion_t, self.motion_percentile))
        motion_t = np.clip(motion_t, 0, hi) if hi > 1e-8 else np.zeros_like(motion_t)
        lo, hi = float(motion_t.min()), float(motion_t.max())
        if hi-lo <= 1e-6:
            motion_t = np.zeros_like(motion_t, dtype=np.float32)
        else:
            motion_t = ((motion_t-lo)/(hi-lo)).astype(np.float32)
        process_data, _ = self.transform((cropped_rgb, label))
        if self.motion_guided_mask:
            from motion_utils import make_motion_guided_tube_mask
            mask = make_motion_guided_tube_mask(motion_t, self.mask_ratio, self.motion_beta)
        else:
            mask = self.masked_position_generator()
        return process_data, mask, torch.from_numpy(motion_t).float()

    def __repr__(self):
        return f"(DataAugmentationForVideoMAE, transform={self.transform}, motion_enabled={self.motion_enabled})"


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=os.path.abspath("datasets/UCF-101"),
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        motion_enabled=getattr(args, 'motion_enabled', False),
        motion_root=getattr(args, 'motion_root', None),
        motion_gap=getattr(args, 'motion_gap', 4),
        motion_missing_policy=getattr(args, 'motion_missing_policy', 'zero'))
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
