# Dogs and cats classification using FastAI
## Connect with me

<a href="https://www.linkedin.com/in/piyapadech/">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
</a>

## About
I use dataset from dogs vs cats kaggle datasets to explore fastai features. This project using ResNet18 with FastAI to classify images between dog and cat. I straightforward to use FastAI, so there are no preprocess or explore data in this project. Remember that FastAI only work with Pytorch, don't forget to use Pytorch while training model with FastAI. 
## Let's start! Setup
Firstly, we have to install FastAI package with this command
```
pip install -qq fastai --upgrade
```
Then import FastAI vision library for image classification
```
import fastai
from fastai.vision.all import *
from fastai.vision.augment import *
```
Check available of GPU and check which device we're using. If we're using GPU this code will print True 
```
use_cuda = torch.cuda.is_available()
print(use_cuda)
```
This code will print cuda if we're using GPU
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```
## Prepare Data
Create label list from images file name
```
label = []
for i in train:
    if 'cat' in i:
        label.append(0)
    elif 'dog' in i:
        label.append(1)
```
Then we have to create dataframe for training with FastAI
```
label_df = pd.DataFrame(label, columns=['labels'])
train_df = pd.DataFrame(train, columns=['img_name'])
df = pd.concat([train_df, label_df],axis=1)
```
Use DataLoader to prepare datasets
```
dls = ImageDataLoaders.from_df(df, path_train, folder=None, device = 'cuda', item_tfms=Resize(224, 224))
```
random show images and thier labels by show_batch 
```
dls.show_batch()
```
<div align="center"'>
    <img src="https://user-images.githubusercontent.com/59856773/163974404-a10ec85a-c35c-45bd-b3ea-2c54848a3713.png" alt="Dog"/>
</div>
                    
## Train
Define data for training model(dls), architecture(this project using ResNet18) and metrics for evaluation.
```
learn = cnn_learner(dls, 
                    resnet18, 
                    metrics=[accuracy,error_rate])
learn = learn.to_fp16()
```
Now training time! Define epochs and learning rate (I trained 2 epochs with learning rate = 1e-3) to learn.fine_tune and use learn.show_results() to view the result
```
learn.fine_tune(2, 1e-3)
learn.show_results()
```
<div align="center"'>
    <img width="485" alt="Screen Shot 2565-04-19 at 16 27 22" src="https://user-images.githubusercontent.com/59856773/163974473-77d100b8-b90e-44fe-ab19-649dd75a1c4d.png">
</div>
<div align="center"'>
    <img src="https://user-images.githubusercontent.com/59856773/163974516-562e0961-4986-4aae-ba39-93ca6bd89c32.png"/>
</div>

The result is great! Accuracy are around 0.98 and 0.99 and random show images with predictions are correct.

## Predict
Create testset from test path by get_image_files
```
fnames = get_image_files('/content/drive/Shareddrives/SuperAI/Kaggle/dogs vs cats/test')
```
Predict all of testset saving in prediction dataframe.
```root_dir = '/content/drive/Shareddrives/SuperAI/Kaggle/dogs vs cats/test'
prediction = {'id': [], 'label': []}
for idx, file in tqdm(enumerate(os.listdir(root_dir))):
    prediction['id'].append(idx+1)
    prediction['label'].append(learn.predict(fnames[idx])[0])
```
### Warning, long prediction time!! This FastAI project takes almost 4 hours to predict 12,500 images.
                      
Thanks to Koravich Sangkaew for sharing FastAI knowledge.

