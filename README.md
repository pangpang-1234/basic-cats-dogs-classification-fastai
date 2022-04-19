# Dogs and cats classification using FastAI
## About
I use dataset from dogs vs cats kaggle datasets to explore fastai features. This project using ResNet18 with FastAI to classify images between dog and cat. I straightforward to use FastAI, so there are no preprocess or explore data present in this project. Remember that FastAI only work with Pytorch, don't forget to use Pytorch while training model with FastAI. 
## Let's start! Setup
Firstly, we have to install FastAI packages with this command
```
pip install -qq fastai --upgrade
```
Then import FastAI vision library for image classification
```
import fastai
from fastai.vision.all import *
from fastai.vision.augment import *
```
Check availabel of GPU and check what device that we're using. If we're using GPU this code will print True 
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
random show images
```
dls.show_batch()
```
## Train
Define data to train (dls), which model to use (in this project I use ResNet18) and which metrics to evaluate.
```
learn = cnn_learner(dls, 
                    resnet18, 
                    metrics=[accuracy,error_rate])
learn = learn.to_fp16()
```
Now training time! Define epochs and learning rate (I trained 2 epochs with learning rate = 1e-3)
```
learn.fine_tune(2, 1e-3)
learn.show_results()
```
This is the results

## Predict
Create test set from test path
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
