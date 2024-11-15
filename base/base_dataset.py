import random, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from math import ceil
from torchvision import models
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class BaseDataSet(Dataset):
    def __init__(self, data_dir, split, mean, std, base_size=None, augment=True, val=False,
                 jitter=False, use_weak_lables=False, weak_labels_output=None, crop_size=None, scale=False, flip=False,
                 rotate=False,
                 blur=False, return_id=False, percnt_lbl=None):

        self.root = data_dir
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        self.jitter = jitter
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.return_id = return_id
        self.percnt_lbl = percnt_lbl
        self.val = val

        self.a = 0
        self.image_A1 = torch.empty(0)
        self.image_B1 = torch.empty(0)
        self.label1 = torch.empty(0)
        self.features = []
        self.pre_images = []
        self.post_images = []
        self.labels = []

        self.use_weak_lables = use_weak_lables
        self.weak_labels_output = weak_labels_output

        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur

        self.jitter_tf = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        self.files = []
        self.images_A = []
        self.images_B = []
        self.l = []
        self._set_files()
        self.chazhi()
        self.new_data_start_index = len(self.files)

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError

    def chazhi(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def _rotate(self, image, label):
        # Rotate the image with an angle between -10 and 10
        h, w, _ = image.shape
        angle = random.randint(-10, 10)
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)  # , borderMode=cv2.BORDER_REFLECT)
        label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)  # ,  borderMode=cv2.BORDER_REFLECT)
        return image, label

    def _crop(self, image_A, image_B, label):
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 2:
            crop_h, crop_w = self.crop_size
        elif isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size
        else:
            raise ValueError

        h, w, _ = image_A.shape
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            image_A = cv2.copyMakeBorder(image_A, value=self.image_padding, **pad_kwargs)
            image_B = cv2.copyMakeBorder(image_B, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)  # use 0 for padding

        # Cropping
        h, w, _ = image_A.shape
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image_A = image_A[start_h:end_h, start_w:end_w]
        image_B = image_B[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        return image_A, image_B, label

    def _blur(self, image, label):
        # Gaussian Blud (sigma between 0 and 1.5)
        sigma = random.random() * 1.5
        ksize = int(3.3 * sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def _flip(self, image_A, image_B, label):
        # Random H flip
        if random.random() > 0.5:
            image_A = np.fliplr(image_A).copy()
            image_B = np.fliplr(image_B).copy()
            label = np.fliplr(label).copy()
        return image_A, image_B, label

    def _resize(self, image_A, image_B, label, bigger_side_to_base_size=True):
        if isinstance(self.base_size, int):
            h, w, _ = image_A.shape
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
                # longside = random.randint(int(self.base_size*0.5), int(self.base_size*1))
            else:
                longside = self.base_size

            if bigger_side_to_base_size:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
                    int(1.0 * longside * h / w + 0.5), longside)
            else:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (
                    int(1.0 * longside * h / w + 0.5), longside)
            image_A = np.asarray(Image.fromarray(np.uint8(image_A)).resize((w, h), Image.BICUBIC))
            image_B = np.asarray(Image.fromarray(np.uint8(image_B)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image_A, image_B, label

        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            h, w, _ = image_A.shape
            if self.scale:
                scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                h, w = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                h, w = self.base_size
            image_A = np.asarray(Image.fromarray(np.uint8(image_A)).resize((w, h), Image.BICUBIC))
            image_B = np.asarray(Image.fromarray(np.uint8(image_B)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image_A, image_B, label

        else:
            raise ValueError

    def _val_augmentation(self, image_A, image_B, label):
        if self.base_size is not None:
            image_A, image_B, label = self._resize(image_A, image_B, label)
            image_A = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image_A))))
            image_B = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image_B))))
            return image_A, image_B, label

        image_A = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image_A))))
        image_B = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image_B))))
        return image_A, image_B, label

    def _augmentation(self, image_A, image_B, label):
        h, w, _ = image_A.shape

        if self.base_size is not None:
            image_A, image_B, label = self._resize(image_A, image_B, label)

        if self.crop_size is not None:
            image_A, image_B, label = self._crop(image_A, image_B, label)

        if self.flip:
            image_A, image_B, label = self._flip(image_A, image_B, label)

        image_A = Image.fromarray(np.uint8(image_A))
        image_B = Image.fromarray(np.uint8(image_B))

        image_A = self.jitter_tf(image_A) if self.jitter else image_A
        image_B = self.jitter_tf(image_B) if self.jitter else image_B

        return self.normalize(self.to_tensor(image_A)), self.normalize(self.to_tensor(image_B)), label

    def interpolate_pixel(self,original_pixel, new_pixel_pos, image, weight):
        interpolated_values = []
        for channel in range(3):
            original_value = image[channel, original_pixel[0], original_pixel[1]]
            new_value = image[channel, new_pixel_pos[0], new_pixel_pos[1]]
            interpolated_value =weight * original_value + (1 - weight) * new_value
            interpolated_values.append(interpolated_value)

        return interpolated_values

    def jisuan(self, label):
        return torch.sum(label == 1)

    def yuanlai(self, image_A, image_B, labels):


        aaaa = []
        bbbb = []
        cccc = []
        a = 0
        b = 0
        total = labels[0].shape[0] * labels[0].shape[1]

        for label in labels:
            if label.ndim == 3:
                label = label[:, :, 0]

            label = torch.from_numpy(label)
            label[label >= 1] = 1


            if torch.any(label > 0):
                change = self.jisuan(label)
                ratio = change / total

                a += 1
                b += ratio
        c = b / a


        for pre_image, post_images, label in zip( image_A, image_B, labels):
            image_a = pre_image.transpose(2, 0, 1)
            image_b = post_images.transpose(2, 0, 1)
            pre_image = torch.from_numpy(image_a)
            post_images = torch.from_numpy(image_b)
            if label.ndim == 3:
                label = label[:, :, 0]

            label = torch.from_numpy(label)
            label[label >= 1] = 1


            if torch.any(label > 0):
                # total = label.shape[0] * label.shape[1]
                change = self.jisuan(label)
                ratio = change / total

                if ratio < 0.02:
                    weight = 0.8
                    iterations = 3
                elif ratio < c and ratio >= 0.02:
                    weight = 0.8
                    iterations = 2
                else:
                    weight = 0.8
                    iterations = 1
                label = label.numpy().astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                dilated_label = cv2.dilate(label, kernel, iterations=iterations)
                new_change_area = np.bitwise_and(dilated_label, np.bitwise_not(label))
                new_change_positions = np.argwhere(new_change_area == 1)


                label_positions = np.argwhere(label == 1)


                for pos in new_change_positions:
                    random_label_pos = random.choice(label_positions)
                    interpolated_values = self.interpolate_pixel(random_label_pos, pos, pre_image, weight)
                    for channel in range(3):
                        pre_image[channel, pos[0], pos[1]] = interpolated_values[channel]
                    interpolated_values = self.interpolate_pixel(random_label_pos, pos, post_images, weight)
                    for channel in range(3):
                        post_images[channel, pos[0], pos[1]] = interpolated_values[channel]

                pre_image = pre_image.cpu()
                post_images = post_images.cpu()
                pre_image = pre_image.numpy().transpose(1, 2, 0)
                post_images = post_images.numpy().transpose(1, 2, 0)


                aaaa.append(pre_image)
                bbbb.append(post_images)
                cccc.append(dilated_label)


        return aaaa, bbbb, cccc

    def __len__(self):
        return self.new_data_start_index + len(self.images_A)

    def __getitem__(self, index):

        if index < self.new_data_start_index:
            image_A, image_B, label, image_id = self._load_data(index)
            if label.ndim == 3:
                label = label[:, :, 0]

            if self.val:
                image_A, image_B, label = self._val_augmentation(image_A, image_B, label)
            elif self.augment:
                image_A, image_B, label = self._augmentation(image_A, image_B, label)

            label[label >= 1] = 1
            label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        else:
            new_index = index - self.new_data_start_index
            image_A = self.images_A[new_index]
            image_B = self.images_B[new_index]
            label = self.l[new_index]

            image_A, image_B, label = self._augmentation(image_A, image_B, label)

            label[label >= 1] = 1
            label = torch.from_numpy(np.array(label, dtype=np.int32)).long()

        return image_A, image_B, label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
