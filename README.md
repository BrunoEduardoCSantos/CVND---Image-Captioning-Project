# Image-Captioning

In this project, a neural network architecture to automatically generate captions from images.

After using the Microsoft Common Objects in COntext (MS COCO) dataset to train  the network, new captions will be generated based on new images. 

## Project Files

The project includes the following files:
*  [model](https://github.com/BrunoEduardoCSantos/Image-Captioning/blob/master/model.py): containing the model architecture.
* [training](https://github.com/BrunoEduardoCSantos/Image-Captioning/blob/master/2_Training%20.ipynb): data pre-processing and training pipeline .
* [inference](https://github.com/BrunoEduardoCSantos/Image-Captioning/blob/master/3_Inference%20.ipynb): generate captions on test dataset using the trained model.

## Understanding LSTMs 
Long-Short-Term-Memory network is a sequential architecture that allows to solve long-term dependency problems. Remembering information for long periods of time is practically their default behavior. 

[image_0]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png
![LSTMs architecture][image_0] 
In order to achieve this long term behaviour LSTMs use 4 stages/gates as follows:
* Forget gate
* Learn gate
* Remember gate
* Use gate

The learn gate is a combination of current events with parts of short-term memory that weren't ignored by pass-through factor. 
Mathematically, the expression is the following:

[image_1]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png
![Learn gate][image_1] 

where *i* is the ignoring factor given by a sigmoide between 0 and 1.

The forget gate is simply using long-term memories and forget part of it, creating a new memory.

[image_2]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png 
![Forget gate][image_2] 

The remember gate is just combining the forget and learning gate generating a new long-term memory.

[image_3]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png
![Remeber gate][image_3] 


Finally, we need to decide what we’re going to output, i.e., use gate aka new short-term memory. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

[image_4]:  http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png
![Remeber gate][image_4] 

## Methodology

For the representation of images, it was used a Convolutional Neural Network (CNN). They have been widely used and
studied for image tasks, and are currently state-of-the art for object recognition and detection. The particular choice of CNN uses a novel approach to batch normalization and yields the current best performance on the ILSVRC 2014 classification competition. For a particular choice of CNN architecture it was used ResNet due to this [performance on object classification on ImageNet](https://github.com/jcjohnson/cnn-benchmarks).

Regarding the decoder, the choice of sequence generator LSTM  is governed by its ability to dealwith vanishing and exploding gradients  the most common challenge in designing and training RNNs. To select the embed and hidden size (=512) I used 
(this)[https://arxiv.org/pdf/1411.4555.pdf] paper. In addition,  Dropout was used to avoid overfitting. 

[image_5]: https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/03/explain_2.png
![General architecture][image_5] 




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
* [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf) by  Oriol Vinyals et al


## TODO
* Use the validation set to guide your search for appropriate hyperparameters.
* Implement beam search to generate captions on new images.
* Tinker the model with attention to get research paper results
* Use YOLO to object detection
* Implementation of beam search
* Use attention model in text generation
