# Bengali.AI Handwritten Grapheme Classification  
**Kaggle Competition (Dec 20, 2019  ~ Mar 17,2020)**  
**Top 4% Result (64th/2063)**  
# Introduction  
For this competition, you’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.  
By participating in the competition, you’ll hopefully accelerate Bengali handwritten optical character recognition research and help enable the digitalization of educational resources. Moreover, the methods introduced in the competition will also empower cousin languages in the Indian subcontinent.  

<img width="290" alt="image" src="https://user-images.githubusercontent.com/52495256/75094244-be24e600-55cc-11ea-9e11-a4b915964226.png">  
 
https://www.kaggle.com/c/bengaliai-cv19  

# Install  
### Requirements  
  - apex  
  - efficientnet-pytorch  
```bash
$ pip install -r requirements.txt
```
### Donwnload  
  - Download the preprocessing dataset with 3 x 256 x 256 size
```
$ cd data/bengaliai/train
$ sh download.sh
```  

# Run  
```bash
$ sh run.sh
```  



#  Results  

|   | Model            |Augmentation      |  amp opt_level |esemble  | elapsed time(sec) per 1 epoch |Accuracy(PB) |
|:--:|:----------------:|:----------------:|:----------------:|:-------------:|:--------:|:-------:|
|**1**|  EfficientNet-b0 |  Rotate, Normalize |  O1      |None        | 1041               | 0.9699  |
|**2**|  EfficientNet-b5 |  Rotate, Normalize | O1      |None        | 2696                |  0.9736  |
|**3**|  GhostNet |  Gridmask(3), Normalize |  O1      |None        | 1985                | 0.9741  |
|**4**|  EfficientNet-b5 |  Gridmask (1), Normalize| O1      |None        | 3319         |  0.9815  |
|**5**|  EfficientNet-b5 |  Gridmask (3), Normalize| O1      |None        | 3588       |  0.9831  |
|**6**|  EfficientNet-b5 |  Gridmask (3), Normalize| O0      |None        | 3633    |  0.9839  |
|**7**|  SE_ResNeXt-50 |  Gridmask (3), Normalize |  O0      |None        | 2679                | 0.9841  |
|**8**|  SE_ResNeXt-50(layer4 weight init) |  Gridmask (3), Normalize | O0  |None        |  2705  | 0.9857  |
|**9**|  1 EfficientNet + 2 SE_ResNeXt-50 |  Gridmask (3), Normalize | O0  |True        | X  |  **0.9867**  |


# Main Contributors  
HeeChul Jung, Chaehyeon Lee, Jaehyeop Choi  

# Reference  
[1] EfficientNet : https://github.com/lukemelas/EfficientNet-PyTorch.git  
[2] SE_ResNeXt-50 : https://github.com/Cadene/pretrained-models.pytorch  
[3] GhostNet : https://github.com/huawei-noah/ghostnet  

