import os
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import random


class MFI_Dataset():
    def __init__(self, datasetPath, phase, use_dataTransform, resize, imgSzie, fusion_type):
        super(MFI_Dataset, self).__init__()
        self.datasetPath = datasetPath
        self.phase = phase
        self.use_dataTransform = use_dataTransform
        self.resize = resize
        self.imgSzie = imgSzie
        self.fusion_type = fusion_type

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)])

    def __len__(self):
        dirsName = os.listdir(self.datasetPath)
        assert len(dirsName) >= 2, "Please check that the dataset is formatted correctly."
        dirsPath = os.path.join(self.datasetPath, dirsName[0])
        return len(os.listdir(dirsPath))

    def __getitem__(self, index):
        if self.phase == "train":

            if self.fusion_type == "medical":
                # source image1
                #print('organize the medical fusion data')
                sourceImg1_dirPath = os.path.join(self.datasetPath, "MRI")
                sourceImg1_names = os.listdir(sourceImg1_dirPath)
                sourceImg1_names.sort(key=lambda x:int(x[:-4]))
                # print(sourceImg1_names)
                sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
                sourceImg1 = Image.open(sourceImg1_path).convert('L')
                sourceImg1 = np.array(sourceImg1)

                # print(np.max(sourceImg1))
                # plt.figure()
                # plt.imshow(sourceImg1, cmap='gray')

                # source image2
                sourceImg2_dirPath = os.path.join(self.datasetPath, "PET")
                sourceImg2_names = os.listdir(sourceImg2_dirPath)
                sourceImg2_names.sort(key=lambda x:int(x[:-4]))
                # print(sourceImg2_names)
                sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
                sourceImg2 = Image.open(sourceImg2_path).convert('RGB')
                sourceImg2 = np.array(sourceImg2)

                # print(np.max(sourceImg2))
                # plt.figure()
                # plt.imshow(sourceImg2)
                # plt.show()

                # convert RGB image to YCbCr
                ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
                sourceImg2 = ycbcr_sourceImg2[:, :, 0]

                if self.resize:
                    sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                    sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))

                if self.use_dataTransform:
                    sourceImg1 = self.transform(sourceImg1)
                    sourceImg2 = self.transform(sourceImg2)

                return [sourceImg1, sourceImg2]

            elif self.fusion_type == "vis-inf":
                # source image1
                #print('organize the vis-inf fusion data')
                sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
                sourceImg1_names = os.listdir(sourceImg1_dirPath)
                sourceImg1_names = sorted(sourceImg1_names)
                sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
                sourceImg1 = cv2.imread(sourceImg1_path, 0)
                # cv2.imshow('source_1', sourceImg1)

                # convert RGB image to YCbCr
                if len(sourceImg1.shape) == 3 & sourceImg1.shape[-1] > 1:
                    ycbcr_sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_RGB2YCR_CB)
                    sourceImg1 = ycbcr_sourceImg1[:, :, 0]

                # source image2
                sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
                sourceImg2_names = os.listdir(sourceImg2_dirPath)
                sourceImg2_names = sorted(sourceImg2_names)
                sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
                sourceImg2 = cv2.imread(sourceImg2_path)
                # cv2.imshow('source_2', sourceImg2)

                # convert RGB image to YCbCr
                if len(sourceImg2.shape) == 3 & sourceImg2.shape[-1] > 1:
                    ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
                    sourceImg2 = ycbcr_sourceImg2[:, :, 0]

                x, y = sourceImg1.shape
                x_dim = random.randint(0, x-256)
                y_dim = random.randint(0, y-256)
                sourceImg1 = sourceImg1[x_dim:x_dim+256, y_dim:y_dim+256]
                sourceImg2 = sourceImg2[x_dim:x_dim+256, y_dim:y_dim+256]

                # plt.figure()
                # plt.imshow(sourceImg2, cmap='Greys_r')
                # plt.show()

                if self.resize:
                    sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                    sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))

                if self.use_dataTransform:  # 归一化到[-1,1]
                    sourceImg1 = self.transform(sourceImg1)
                    sourceImg2 = self.transform(sourceImg2)

                return [sourceImg1, sourceImg2]

            else:

                # source image1
                sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
                sourceImg1_names = os.listdir(sourceImg1_dirPath)
                sourceImg1_names = sorted(sourceImg1_names)
                sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
                sourceImg1 = cv2.imread(sourceImg1_path)
                #cv2.imshow('source_1', sourceImg1)

                # convert RGB image to YCbCr
                if len(sourceImg1.shape) == 3 & sourceImg1.shape[-1] > 1:
                    ycbcr_sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_RGB2YCR_CB)
                    sourceImg1 = ycbcr_sourceImg1[:, :, 0]

                # source image2
                sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
                sourceImg2_names = os.listdir(sourceImg2_dirPath)
                sourceImg2_names = sorted(sourceImg2_names)
                sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
                sourceImg2 = cv2.imread(sourceImg2_path)
                #cv2.imshow('source_2', sourceImg2)

                # convert RGB image to YCbCr
                if len(sourceImg2.shape) == 3 & sourceImg2.shape[-1] > 1:
                    ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
                    sourceImg2 = ycbcr_sourceImg2[:, :, 0]

                # plt.figure()
                # plt.imshow(sourceImg2)
                # plt.show()

                x, y = sourceImg1.shape
                x_dim = random.randint(0, x - 256)
                y_dim = random.randint(0, y - 256)
                sourceImg1 = sourceImg1[x_dim:x_dim + 256, y_dim:y_dim + 256]
                sourceImg2 = sourceImg2[x_dim:x_dim + 256, y_dim:y_dim + 256]

                if self.resize:
                    sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                    sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))

                if self.use_dataTransform:
                    sourceImg1 = self.transform(sourceImg1)
                    sourceImg2 = self.transform(sourceImg2)

                return [sourceImg1, sourceImg2]


        if self.phase == "valid":
            if self.fusion_type == "medical":
                print('test medical image fusion')
                # source image1
                sourceImg1_dirPath = os.path.join(self.datasetPath, "MRI")
                sourceImg1_names = os.listdir(sourceImg1_dirPath)
                sourceImg1_names.sort(key=lambda x:int(x[:-4]))
                print(sourceImg1_names)
                sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
                sourceImg1 = Image.open(sourceImg1_path).convert('L')
                sourceImg1 = np.asarray(sourceImg1)

                # source image2
                sourceImg2_dirPath = os.path.join(self.datasetPath, "PET")
                sourceImg2_names = os.listdir(sourceImg2_dirPath)
                sourceImg2_names.sort(key=lambda x:int(x[:-4]))
                print(sourceImg2_names)
                sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
                sourceImg2 = Image.open(sourceImg2_path).convert('RGB')
                sourceImg2 = np.asarray(sourceImg2)

                # convert RGB image to YCbCr
                ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
                # source_img2 = cv2.cvtColor(ycbcr_sourceImg2, cv2.COLOR_YCR_CB2RGB)
                sourceImg2 = ycbcr_sourceImg2[:, :, 0]
                sourceImg2_cr = ycbcr_sourceImg2[:, :, 1]
                sourceImg2_cb = ycbcr_sourceImg2[:, :, 2]

                if self.resize:
                    sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                    sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))
                if self.use_dataTransform:
                    sourceImg1 = self.transform(sourceImg1)
                    sourceImg2 = self.transform(sourceImg2)

                return [sourceImg1, sourceImg2, sourceImg2_cr, sourceImg2_cb]

            elif self.fusion_type == "vis-inf":
                print('test vis_inf image fusion')
                # source image1
                #print('organize the vis-inf fusion data')
                sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
                sourceImg1_names = os.listdir(sourceImg1_dirPath)
                sourceImg1_names = sorted(sourceImg1_names)
                print(sourceImg1_names)
                sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
                sourceImg1 = cv2.imread(sourceImg1_path)
                sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_RGB2GRAY)
                # cv2.imshow('source_1', sourceImg1)

                # source image2
                sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
                sourceImg2_names = os.listdir(sourceImg2_dirPath)
                sourceImg2_names = sorted(sourceImg2_names)
                print(sourceImg2_names)
                sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
                sourceImg2 = cv2.imread(sourceImg2_path)
                # cv2.imshow('source_2', sourceImg2)

                # convert RGB image to YCbCr
                ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
                sourceImg2 = ycbcr_sourceImg2[:, :, 0]
                sourceImg2_cr = ycbcr_sourceImg2[:, :, 1]
                sourceImg2_cb = ycbcr_sourceImg2[:, :, 2]

                # plt.figure()
                # plt.imshow(sourceImg2, cmap='Greys_r')
                # plt.show()

                if self.resize:
                    sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                    sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))

                if self.use_dataTransform:
                    sourceImg1 = self.transform(sourceImg1)
                    sourceImg2 = self.transform(sourceImg2)

                return [sourceImg1, sourceImg2, sourceImg2_cr, sourceImg2_cb]

            else:
                # if self.fusion_type == "multi-focus":
                #     print('organize the multi-focus fusion data')
                # else:
                #     print('organize the multi-exposure fusion data')
                print('test multi_exposure image fusion')

                # source image1
                sourceImg1_dirPath = os.path.join(self.datasetPath, "source_1")
                sourceImg1_names = os.listdir(sourceImg1_dirPath)
                sourceImg1_names = sorted(sourceImg1_names)
                sourceImg1_path = os.path.join(sourceImg1_dirPath, sourceImg1_names[index])
                sourceImg1 = cv2.imread(sourceImg1_path)
                # cv2.imshow('source_1', sourceImg1)

                # convert RGB image to YCbCr
                ycbcr_sourceImg1 = cv2.cvtColor(sourceImg1, cv2.COLOR_RGB2YCR_CB)
                sourceImg1 = ycbcr_sourceImg1[:, :, 0]
                sourceImg1_cr = ycbcr_sourceImg1[:, :, 1]
                sourceImg1_cb = ycbcr_sourceImg1[:, :, 2]

                # source image2
                sourceImg2_dirPath = os.path.join(self.datasetPath, "source_2")
                sourceImg2_names = os.listdir(sourceImg2_dirPath)
                sourceImg2_names = sorted(sourceImg2_names)
                sourceImg2_path = os.path.join(sourceImg2_dirPath, sourceImg2_names[index])
                sourceImg2 = cv2.imread(sourceImg2_path)
                # cv2.imshow('source_2', sourceImg2)

                # convert RGB image to YCbCr
                ycbcr_sourceImg2 = cv2.cvtColor(sourceImg2, cv2.COLOR_RGB2YCR_CB)
                sourceImg2 = ycbcr_sourceImg2[:, :, 0]
                sourceImg2_cr = ycbcr_sourceImg2[:, :, 1]
                sourceImg2_cb = ycbcr_sourceImg2[:, :, 2]

                # plt.figure()
                # plt.imshow(sourceImg2)
                # plt.show()

                if self.resize:
                    sourceImg1 = cv2.resize(sourceImg1, (self.imgSzie, self.imgSzie))
                    sourceImg2 = cv2.resize(sourceImg2, (self.imgSzie, self.imgSzie))

                if self.use_dataTransform:
                    sourceImg1 = self.transform(sourceImg1)
                    sourceImg2 = self.transform(sourceImg2)

                return [sourceImg1, sourceImg2, sourceImg1_cr, sourceImg1_cb, sourceImg2_cr, sourceImg2_cb]