# Bengali.AI Handwritten Grapheme Classification  
**Kaggle Competition (Dec 20, 2019  ~ Mar 17,2020)**  
**Top 3% Result (33/1536)**  
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

|   | Model            |Augmentation      | learning_rate       | amp opt_level |esemble  | Accuracy |
|:--:|:----------------:|:----------------:|:----------------:|:-------------:|:--------:|:-------:|
|**1**|  efficientnet-b5 |  Rotate, Normalize | 0.01                | O1      |None        |  0.9736  |
|**2**|  ghostnet |  Gridmask(3), Normalize | 0.01                | O1      |None        |  0.9741  |
|**3**|  efficientnet-b5 |  Gridmask (1), Normalize| 0.01         | O1      |None        |  0.9815  |
|**4**|  efficientnet-b5 |  Gridmask (3), Normalize| 0.01       | O1      |None        |  0.9831  |
|**5**|  efficientnet-b5 |  Gridmask (3), Normalize| 0.005    | O0      |None        |  0.9839  |
|**6**|  se_resnext101_32x4 |  Gridmask (3), Normalize | 0.01                | O0      |None        |  0.9841  |
|**7**|  se_resnext101_32x4(layer4 weight init) |  Gridmask (3), Normalize | 0.01  | O0  |None        |  0.9857  |
|**8**|  1 eff + 2 seresnext |  Gridmask (3), Normalize | 0.01  | O0  |True        |  **0.9867**  |


# Main Contributors  
HeeChul Jung, Chaehyeon Lee, Jaehyeop Choi  

# Reference  
[1] Efficientnet : https://github.com/lukemelas/EfficientNet-PyTorch.git  
[2] se_resnext101_32x4d : https://github.com/Cadene/pretrained-models.pytorch  
[3] Ghostnet : https://github.com/huawei-noah/ghostnet  

