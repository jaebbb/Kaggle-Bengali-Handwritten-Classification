import pandas as pd
import os
import cv2
from PIL import Image
import torch
import numpy as np

from torch.utils.data import Dataset

class BengaliImageDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):

        self.data = pd.read_csv(csv_file)

        self.data_dummie_labels = pd.get_dummies(
            self.data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']],
            columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
        )


        self.path = path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_name = os.path.join(self.path, self.data.loc[idx, 'image_id'] + '.png')
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)


        if self.transform is not None:
            res = self.transform(image=img)
            image = res['image']
        #print(image[0,0,0])
        image = image.unsqueeze(0)
        image = image.repeat(3, 1, 1)
 
        values, idx_graph = torch.max(torch.tensor(self.data_dummie_labels.iloc[idx, 0:168]), 0)
        values, idx_vowel = torch.max(torch.tensor(self.data_dummie_labels.iloc[idx, 168:179]), 0)
        values, idx_conso = torch.max(torch.tensor(self.data_dummie_labels.iloc[idx, 179:186]), 0)



        if self.labels:
            return {
                'image': image,
                'l_graph': idx_graph.type(torch.LongTensor),
                'l_vowel': idx_vowel.type(torch.LongTensor),
                'l_conso': idx_conso.type(torch.LongTensor),

            }
        else:
            return {'image': image}




def make_square(img, target_size=256):
    img = img[0:-1, :]
    height, width = img.shape

    x = target_size
    y = target_size

    square = np.ones((x, y), np.uint8) * 255
    square[(y - height) // 2:y - (y - height) // 2, (x - width) // 2:x - (x - width) // 2] = img

    return square


class BengaliParquetDataset(Dataset):

    def __init__(self, csv_file, parquet_file, transform=None):

        self.csv_data = pd.read_csv(csv_file)


        self.data_dummie_labels = pd.get_dummies(
            self.csv_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']],
            columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
        )


        self.df = self.csv_data.set_index('image_id', drop = True)

        print('loading...')
        self.data = pd.concat(
                    pd.read_parquet('/%s'%f , engine='pyarrow')
                    for f in parquet_file
                )

#        self.data = pd.read_parquet(parquet_file)
        self.transform = transform
#        print(len(self.data))
#        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        HEIGHT = 137
        WIDTH = 236
        TARGET_SIZE = 256

        tmp = self.data.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH)
        img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3))
        img[..., 0] = make_square(tmp, target_size=TARGET_SIZE)


        img = np.array(img, dtype=np.uint8)


        image = Image.fromarray(img)
        image = image.convert('RGB')
        image_id = self.data.iloc[idx, 0]


        if self.transform:
            img = self.transform(image)


        grapheme_root, vowel_diacritic, consonant_diacritic, grapheme  =  self.df.loc[image_id].values

        return {
            'image_id': image_id,
            'image': img,
            'l_graph': grapheme_root,
            'l_vowel': vowel_diacritic,
            'l_conso': consonant_diacritic,

        }
