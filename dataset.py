from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from tqdm import tqdm


class CoverDataset(Dataset):
    def __init__(
        self,
        dir_path: str,
        imsize: int,
        transforms,
        use_minmax: bool = True,
        use_ram: bool = False,
    ) -> None:
        super().__init__()
        self.images = os.listdir(dir_path)
        self.images = [os.path.join(dir_path, i) for i in self.images]
        self.imsize = imsize
        self.transforms = transforms
        self.use_minmax = use_minmax
        self.use_ram = use_ram
        if use_ram:
            self.load()

    def load(self):
        self.loaded_images = np.zeros(
            (len(self.images), self.imsize, self.imsize, 3), dtype="uint8"
        )
        for idx, image_path in tqdm(
            enumerate(self.images), desc="Loading dataset", total=len(self.images)
        ):
            img = cv2.imread(image_path)[:, :, ::-1]
            img = cv2.resize(img, (self.imsize, self.imsize))
            self.loaded_images[idx] = img

    def __getitem__(self, index) -> np.array:
        if self.use_ram:
            img = self.loaded_images[index]
        else:
            img = cv2.imread(self.images[index])[:, :, ::-1]
            img = cv2.resize(img, (self.imsize, self.imsize))

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        # minmax norm
        if self.use_minmax:
            img = img - img.min()
            if img.max() != 0:
                img = img / img.max()
            img = img - 0.5
            img = img * 2
        img = img.transpose(2, 0, 1)
        return img

    def __len__(self):
        return len(self.images)
