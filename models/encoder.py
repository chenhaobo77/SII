from models.backbones.resnet_backbone import ResNetBackbone
from utils.helpers import initialize_weights
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import cv2
import numpy as np

resnet50 = {
    "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",
}


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class Encoder(nn.Module):
    def __init__(self, pretrained):
        super(Encoder, self).__init__()

        if pretrained and not os.path.isfile(resnet50["path"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
            os.system('sh models/backbones/get_pretrained_model.sh')

        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained)
        self.base = nn.Sequential(
            nn.Sequential(model.prefix, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])

    def resize_labels(self, labels, target_size=(32, 32)):

        labels = labels.unsqueeze(1).float()
        resized_labels = F.interpolate(labels, size=target_size, mode='nearest')  # 最近邻上采样
        return resized_labels

    def interpolate_features(self, feature_map1, feature_map2, labels, alpha=0.8):

        labels_expanded = labels.expand_as(feature_map1)

        interpolated_features = alpha * feature_map1 * labels_expanded + (1 - alpha) * feature_map2 * labels_expanded

        non_interpolated_features = feature_map2 * (1 - labels_expanded)

        combined_features = interpolated_features + non_interpolated_features
        # combined_features = interpolated_features
        return combined_features

    def forward(self, A, B, C):
        a = self.base(A)
        b = self.base(B)
        # feature_A = []
        # feature_B = []
        # labels_list = []
        # labels_list_t = []
        # feature_A1 = []
        # feature_B1 = []
        # labels_list1 = []
        # labels_list_t1 = []
        # if C is not None:
        #     for c in C:
        #         print("Unique values in label before dilation:", np.unique(c.cpu()))
            # transform = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
            #     transforms.ToTensor()
            # ])
            # resized_labels = torch.stack([transform(label.unsqueeze(0)).squeeze(0) for label in C.float()])

            # print("Resized size:", resized_labels)

            # for label in resized_labels:
            #     if torch.any(label > 0):
            #         print("Unique values in label before dilation:", np.unique(label))

            # for label in resized_labels:
            #     if torch.any(label > 0):
            #         label = label.numpy().astype(np.uint8)
            #         print("Unique values in label before dilation:", np.unique(label))
            #         kernel = np.ones((3, 3), np.uint8)
            #         dilated_label = cv2.dilate(label, kernel, iterations=1)
            #
            #         new_change_area = np.bitwise_and(dilated_label, np.bitwise_not(label))
            #
            #         new_change_positions = np.argwhere(new_change_area == 1)
            #         print("新增变化区域的位置：")
            #         print(new_change_positions)
            #
            #         label_position = np.argwhere(label == 1)
            #         print("原始标签的位置：")
            #         print(label_position)
        #     c = self.resize_labels(C)
        #     for changebefore, changeafter, label, label_t in zip(a, b, c, C):
        #         if torch.any(label > 0):  # 假设标签值大于0表示有变化区域
        #             feature_A.append(changebefore)
        #             feature_B.append(changeafter)
        #             labels_list.append(label)
        #             labels_list_t.append(label_t)
        #         else:
        #             feature_A1.append(changebefore)
        #             feature_B1.append(changeafter)
        #             labels_list1.append(label)
        #             labels_list_t1.append(label_t)
        #
        # if labels_list:
        #     feature_A = torch.stack(feature_A, dim=0)
        #     feature_B = torch.stack(feature_B, dim=0)
        #     labels_list = torch.stack(labels_list, dim=0)
        #     labels_list_t = torch.stack(labels_list_t, dim=0)
        #     feature_A1 = torch.stack(feature_A1, dim=0)
        #     feature_B1 = torch.stack(feature_B1, dim=0)
        #     labels_list1 = torch.stack(labels_list1, dim=0)
        #     labels_list_t1 = torch.stack(labels_list_t1, dim=0)
        #
        #     # print(feature_A.shape)
        #
        #     feature_A1 = feature_A1[:len(feature_A)]
        #     feature_B1 = feature_B1[:len(feature_A)]
        #     labels_list1 = labels_list1[:len(feature_A)]
        #     labels_list_t1 = labels_list_t1[:len(feature_A)]
        #
        #     # print(feature_A1.shape)
        #
        #     if feature_A.shape == feature_A1.shape:
        #         feature_A = self.interpolate_features(feature_A, feature_A1, labels_list)
        #         feature_B = self.interpolate_features(feature_B, feature_B1, labels_list)
        #         labels_list_t_c = labels_list_t
        #         C = torch.cat((C, labels_list_t_c), dim=0)
        #         a = torch.cat((a, feature_A), dim=0)
        #         b = torch.cat((b, feature_B), dim=0)
        #
        #
        #
        #     if len(feature_A) >= 2:
        #         if len(feature_A) ==2:
        #             feature_A = feature_A[:1]
        #             feature_A1 = feature_A[1:]
        #             feature_B = feature_B[:1]
        #             feature_B1 = feature_B[1:]
        #             labels_list = labels_list[:1]
        #             labels_list1 = labels_list[1:]
        #             labels_list_t = labels_list_t[:1]
        #             labels_list_t1 = labels_list_t[1:]
        #
        #             labels_expanded1 = labels_list.expand_as(feature_A)
        #             labels_expanded2 = labels_list1.expand_as(feature_A1)
        #             # interpolated_features = 0.5 * feature_map1 * labels_expanded + (1 - alpha) * feature_map2 * labels_expanded
        diff = torch.abs(a - b)
        x = self.psp(diff)

        return x, C

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()
