# Bengali.AI Handwritten Grapheme Classification  
<img width="290" alt="image" src="https://user-images.githubusercontent.com/52495256/75094244-be24e600-55cc-11ea-9e11-a4b915964226.png">  

Kaggle Competetion (Dec 20, 2019  ~ Mar 17,2020)  
Top 3% Result (33/1536)  
https://www.kaggle.com/c/bengaliai-cv19  

## Main Contributers  
HeeChul Jung, Chaehyeon Lee, Jaehyeop Choi, Yoonju Oh  

## Install  
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

## Run  
```bash
$ sh run.sh
```  



##  Results  

|   | Model            |Augmentation      | learning_rate       | amp opt_level |esemble  | Accuracy |
|:--:|:----------------:|:----------------:|:----------------:|:-------------:|:--------:|:-------:|
|**1**|  efficientnet-b5 |  Rotate, Normalize | 0.01                | O1      |None        |  0.9736  |
|**2**|  efficientnet-b5 |  Gridmask (1), Normalize| 0.01         | O1      |None        |  0.9815  |
|**3**|  efficientnet-b5 |  Gridmask (3), Normalize| 0.01       | O1      |None        |  0.9831  |
|**4**|  efficientnet-b5 |  Gridmask (3), Normalize| 0.005    | O1      |None        |  0.9839  |
|**5**|  efficientnet-b5 |  Gridmask (3), Normalize            | 0.005          | O1      |3+4      | **0.9841**   |  


## Reference  
[1] Efficientnet : https://github.com/lukemelas/EfficientNet-PyTorch.git  

