import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np
import rawpy


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
RAW_EXTENSIONS = (".ARW", ".RAF")
FLIP_LEFT_RIGHT = (
    Image.Transpose.FLIP_LEFT_RIGHT
    if hasattr(Image, "Transpose")
    else Image.FLIP_LEFT_RIGHT
)


def _load_sid_allowed_filenames(split_file):
    allowed = set()
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for token in line.split():
                base = os.path.basename(token)
                if base.upper().endswith(RAW_EXTENSIONS):
                    allowed.add(base)
    return allowed


class PairedLowLightDataset(Dataset):
    def __init__(self, low_light_dir, ground_truth_dir, patch_size=256, is_training=True):
        """
        Loads paired images (dark and bright) for supervised training (e.g., LOL dataset).
        """
        self.low_light_dir = low_light_dir
        self.ground_truth_dir = ground_truth_dir
        self.is_training = is_training
        self.patch_size = patch_size

        # Get list of filenames (assuming names match exactly in both folders)
        self.image_filenames = sorted(
            [
                name
                for name in os.listdir(low_light_dir)
                if name.lower().endswith(IMAGE_EXTENSIONS)
                and os.path.isfile(os.path.join(low_light_dir, name))
                and os.path.isfile(os.path.join(ground_truth_dir, name))
            ]
        )
        if not self.image_filenames:
            raise FileNotFoundError(
                f"No paired image files found in '{low_light_dir}' and '{ground_truth_dir}'."
            )

        # Basic tensor conversion
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        file_name = self.image_filenames[idx]

        # Load images
        low_img = Image.open(os.path.join(self.low_light_dir, file_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.ground_truth_dir, file_name)).convert('RGB')

        if low_img.size != gt_img.size:
            raise ValueError(
                f"Mismatched pair size for '{file_name}': low={low_img.size}, gt={gt_img.size}"
            )

        if self.is_training:
            # Data Augmentation: Random Crop
            w, h = low_img.size
            if w < self.patch_size or h < self.patch_size:
                # Ensure every training sample can produce a fixed-size patch.
                new_w = max(w, self.patch_size)
                new_h = max(h, self.patch_size)
                low_img = low_img.resize((new_w, new_h), Image.BILINEAR)
                gt_img = gt_img.resize((new_w, new_h), Image.BILINEAR)
                w, h = new_w, new_h

            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
            low_img = low_img.crop((x, y, x + self.patch_size, y + self.patch_size))
            gt_img = gt_img.crop((x, y, x + self.patch_size, y + self.patch_size))

            # Data Augmentation: Random Horizontal Flip
            if random.random() < 0.5:
                low_img = low_img.transpose(FLIP_LEFT_RIGHT)
                gt_img = gt_img.transpose(FLIP_LEFT_RIGHT)

        # Convert to PyTorch Tensors
        low_tensor = self.to_tensor(low_img)
        gt_tensor = self.to_tensor(gt_img)

        return low_tensor, gt_tensor


class SIDDataset(Dataset):
    def __init__(
        self,
        short_dir,
        long_dir,
        patch_size=256,
        is_training=True,
        split_file=None,
        raw_half_size=False,
        eval_resize=None,
    ):
        """
        Loads and pairs the many-to-one SID Dataset from RAW files.
        """
        self.short_dir = short_dir
        self.long_dir = long_dir
        self.patch_size = patch_size
        self.is_training = is_training
        self.raw_half_size = raw_half_size
        self.eval_resize = eval_resize

        # 1. Get all files (Supports both Sony .ARW and Fuji .RAF)
        self.short_files = sorted(
            glob.glob(os.path.join(short_dir, "*.ARW"))
            + glob.glob(os.path.join(short_dir, "*.RAF"))
        )
        self.long_files = sorted(
            glob.glob(os.path.join(long_dir, "*.ARW"))
            + glob.glob(os.path.join(long_dir, "*.RAF"))
        )
        if split_file:
            allowed = _load_sid_allowed_filenames(split_file)
            self.short_files = [
                path for path in self.short_files if os.path.basename(path) in allowed
            ]
            if not self.short_files:
                raise FileNotFoundError(
                    f"No SID short-exposure files from split '{split_file}' found in '{short_dir}'."
                )

        # 2. Map Long Exposures by Scene ID (e.g., "10003")
        self.long_dict = {}
        for long_path in self.long_files:
            filename = os.path.basename(long_path)
            scene_id = filename.split('_')[0]
            self.long_dict[scene_id] = long_path

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.short_files)

    def _process_raw(self, raw_path):
        """Uses rawpy to convert raw sensor data into an RGB numpy array"""
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True,
                                  half_size=self.raw_half_size,
                                  no_auto_bright=True,
                                  output_bps=8)
        return rgb

    def __getitem__(self, idx):
        short_path = self.short_files[idx]

        # 3. Pair them by Scene ID instead of raw length
        filename = os.path.basename(short_path)
        scene_id = filename.split('_')[0]
        if scene_id not in self.long_dict:
            raise FileNotFoundError(
                f"No matching long exposure found for scene '{scene_id}' from '{filename}'."
            )
        long_path = self.long_dict[scene_id]

        # 4. Load RAW data
        short_img = self._process_raw(short_path)
        long_img = self._process_raw(long_path)

        if self.is_training:
            # Data Augmentation: Random Crop on Numpy Arrays
            h, w, _ = short_img.shape
            if h < self.patch_size or w < self.patch_size:
                short_img = np.array(
                    Image.fromarray(short_img).resize(
                        (max(w, self.patch_size), max(h, self.patch_size)), Image.BILINEAR
                    )
                )
                long_img = np.array(
                    Image.fromarray(long_img).resize(
                        (max(w, self.patch_size), max(h, self.patch_size)), Image.BILINEAR
                    )
                )
                h, w, _ = short_img.shape

            x = random.randint(0, h - self.patch_size)
            y = random.randint(0, w - self.patch_size)
            short_img = short_img[x:x + self.patch_size, y:y + self.patch_size, :]
            long_img = long_img[x:x + self.patch_size, y:y + self.patch_size, :]

            # Data Augmentation: Random Flip
            if random.random() < 0.5:
                short_img = np.fliplr(short_img)
                long_img = np.fliplr(long_img)
        elif self.eval_resize is not None:
            short_img = np.array(
                Image.fromarray(short_img).resize(
                    (self.eval_resize, self.eval_resize), Image.BILINEAR
                )
            )
            long_img = np.array(
                Image.fromarray(long_img).resize(
                    (self.eval_resize, self.eval_resize), Image.BILINEAR
                )
            )

        # 5. Format for PyTorch
        short_img = np.ascontiguousarray(short_img)
        long_img = np.ascontiguousarray(long_img)

        short_tensor = self.to_tensor(short_img)
        long_tensor = self.to_tensor(long_img)

        return short_tensor, long_tensor