import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/7292452634137d8f5d4478e44727ec1166a89125/dataset.py
def downsample_hrtf(hr_hrtf, hrtf_size, upscale_factor):
    # downsample hrtf
    if upscale_factor == hrtf_size:
        mid_pos = int(hrtf_size / 2)
        lr_hrtf = hr_hrtf[:, :, mid_pos, mid_pos, None, None]
    else:
        lr_hrtf = torch.nn.functional.interpolate(hr_hrtf, scale_factor=1 / upscale_factor)

    return lr_hrtf

class TrainValidHRTFDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        hrtf_dir (str): Train/Valid dataset address.
        hrtf_size (int): High resolution hrtf size.
        upscale_factor (int): hrtf up scale factor.
        transform (callable): A function/transform that takes in an HRTF and returns a transformed version.
    """

    def __init__(self, hrtf_dir: str, hrtf_size: int, upscale_factor: int, transform=None, run_validation =True) -> None:
        super(TrainValidHRTFDataset, self).__init__()
        # Get all hrtf file names in folder
        self.hrtf_file_names = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in os.listdir(hrtf_dir)
                                if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name))]

        if run_validation:
            valid_hrtf_file_names = []
            for hrtf_file_name in self.hrtf_file_names:
                file = open(hrtf_file_name, 'rb')
                hrtf = pickle.load(file)
                if not np.isnan(np.sum(hrtf.cpu().data.numpy())):
                    valid_hrtf_file_names.append(hrtf_file_name)
            self.hrtf_file_names = valid_hrtf_file_names

        # Specify the high-resolution hrtf size, with equal length and width
        self.hrtf_size = hrtf_size
        # How many times the high-resolution hrtf is the low-resolution hrtf
        self.upscale_factor = upscale_factor
        # transform to be applied to the data
        self.transform = transform

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of hrtf data
        with open(self.hrtf_file_names[batch_index], "rb") as file:
            hrtf = pickle.load(file)

        # hrtf processing operations
        if self.transform is not None:
            # If using a transform, treat panels as batch dim such that dims are (panels, channels, X, Y)
            hr_hrtf = torch.permute(hrtf, (0, 3, 1, 2))
            # Then, transform hr_hrtf to normalize and swap panel/channel dims to get channels first
            hr_hrtf = torch.permute(self.transform(hr_hrtf), (1, 0, 2, 3))
        else:
            # If no transform, go directly to (channels, panels, X, Y)
            hr_hrtf = torch.permute(hrtf, (3, 0, 1, 2))

        # downsample hrtf
        lr_hrtf = downsample_hrtf(hr_hrtf, self.hrtf_size, self.upscale_factor)

        return {"lr": lr_hrtf, "hr": hr_hrtf, "filename": self.hrtf_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.hrtf_file_names)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
#*********************Added Code to Process The Noisy Files That Follows the Structure of the TrainValidHRTFDataset Class************************************************************************
class TrainNoisyHRTFDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        clean_hrtf_dir (str): Train/Valid dataset address for clean HRTFs
        noisy_hrtf_dir (str): Train/Valid dataset address for noisy HRTFs
        hrtf_size (int): High resolution hrtf size.
        upscale_factor (int): HRTF up scale factor.
        clean_hrtf_ratio (int): ratio determining how many clean HRTFs are included in the training of the GAN
        transform (callable): A function/transform that takes in an HRTF and returns a transformed version.
    """

    def __init__(self, clean_hrtf_dir: str, noisy_hrtf_dir: str, hrtf_size: int, upscale_factor: int, clean_hrtf_ratio: float, transform=None, run_validation=True) -> None:
        super(TrainNoisyHRTFDataset, self).__init__()

        # Get all noisy HRTF file names
        noisy_files = [os.path.join(noisy_hrtf_dir, hrtf_file_name) 
                       for hrtf_file_name in os.listdir(noisy_hrtf_dir)
                       if os.path.isfile(os.path.join(noisy_hrtf_dir, hrtf_file_name))]

        # Get all clean HRTF file names
        clean_files = [os.path.join(clean_hrtf_dir, hrtf_file_name) 
                       for hrtf_file_name in os.listdir(clean_hrtf_dir)
                       if os.path.isfile(os.path.join(clean_hrtf_dir, hrtf_file_name))]

        # Apply the clean HRTF ratio
        if clean_hrtf_ratio < 1.0:
            selected_clean_count = int(len(clean_files) * clean_hrtf_ratio)
            clean_files = clean_files[:selected_clean_count]

        # Combine noisy and (subset of) clean HRTF files
        self.hrtf_file_names = noisy_files + clean_files
        
        if run_validation:
            valid_hrtf_file_names = []
            for hrtf_file_name in self.hrtf_file_names:
                file = open(hrtf_file_name, 'rb')
                hrtf = pickle.load(file)
                if not np.isnan(np.sum(hrtf.cpu().data.numpy())):
                    valid_hrtf_file_names.append(hrtf_file_name)
            self.hrtf_file_names = valid_hrtf_file_names


        # Specify the high-resolution HRTF size, with equal length and width
        self.hrtf_size = hrtf_size
        # How many times the high-resolution HRTF is the low-resolution HRTF
        self.upscale_factor = upscale_factor
        # Transform to be applied to the data
        self.transform = transform

#********************* End of Added Code to Process The Noisy Files************************************************************************
    
    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of hrtf data
        with open(self.hrtf_file_names[batch_index], "rb") as file:
            hrtf = pickle.load(file)

        # hrtf processing operations
        if self.transform is not None:
            # If using a transform, treat panels as batch dim such that dims are (panels, channels, X, Y)
            hr_hrtf = torch.permute(hrtf, (0, 3, 1, 2))
            # Then, transform hr_hrtf to normalize and swap panel/channel dims to get channels first
            hr_hrtf = torch.permute(self.transform(hr_hrtf), (1, 0, 2, 3))
        else:
            # If no transform, go directly to (channels, panels, X, Y)
            hr_hrtf = torch.permute(hrtf, (3, 0, 1, 2))

            # Ensure the transformed data is valid
        assert torch.isfinite(hr_hrtf).all(), "Transformed tensor contains NaN or Inf"

        # downsample hrtf
        lr_hrtf = downsample_hrtf(hr_hrtf, self.hrtf_size, self.upscale_factor)

        assert torch.isfinite(lr_hrtf).all(), "Downsampled tensor contains NaN or Inf"

        return {"lr": lr_hrtf, "hr": hr_hrtf, "filename": self.hrtf_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.hrtf_file_names)
