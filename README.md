# Image-Captioning

In this project, a neural network architecture to automatically generate captions from images.

After using the Microsoft Common Objects in COntext (MS COCO) dataset to train  the network, new captions will be generated based on new images. 

## Project Files

The project includes the following files:
*  [model](https://github.com/BrunoEduardoCSantos/Image-Captioning/blob/master/model.py): containing the model architecture.
* [training](https://github.com/BrunoEduardoCSantos/Image-Captioning/blob/master/2_Training.ipynb): data pre-processing and training pipeline .
* [inference](https://github.com/BrunoEduardoCSantos/Image-Captioning/blob/master/3_Inference.ipynb): generate captions on test dataset using the trained model.

## Understanding LSTMs 
Long-Short-Term-Memory network is a sequential architecture that allows to solve long-term dependency problems. Remembering information for long periods of time is practically their default behavior. 

[image_0]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png
![LSTMs architecture][image_0] 
In order to achieve this long term behaviour LSTMs use 4 stages/gates as follows:
* Forget gate
* Learn gate
* Remember gate
* Use gate

The learn gate is a combination of current events with parts of long-term memory that weren't ignored by pass-through factor. 
Mathematically, the expression is the following:

[image_1]: https://d17h27t6h515a5.cloudfront.net/topher/2017/November/5a0e2cc3_screen-shot-2017-11-16-at-4.26.22-pm/screen-shot-2017-11-16-at-4.26.22-pm.png 
![Learn gate][image_1] 



## Dataset  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

## References
* [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Chris Olah


## TODO
* Use the validation set to guide your search for appropriate hyperparameters.
* Implement beam search to generate captions on new images.
* Tinker the model with attention to get research paper results
* Use YOLO to object detection
