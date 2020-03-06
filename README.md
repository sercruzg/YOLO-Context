# YOLO-Context
This is part of the code used on the [EgoDaily](https://github.com/sercruzg/EgoDaily) dataset and was used for our [paper](https://doi.org/10.1016/j.imavis.2019.06.002). This code is based on the [YOLO](https://github.com/pjreddie/darknet) code.

This code masks the ground truth out after the convolutions on what we call the "Feature extraction" section, putting all the information inside the bounding box to 0. This forces the Neural Network to use the surrounding objects for detection.

To use the masking in a layer you can set ``mask=1`` like this:

```
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky
mask=1
```
The annotations are the same as YOLO, to train you need a file ``egoDailyDisamTrain.txt`` with each line pointing to the images 

```
<Path-to-images>/frame-0001.jpg
<Path-to-images>/frame-0002.jpg
<Path-to-images>/frame-0003.jpg
...
```

And you need the annotations to be in the same path, however, if the path has the word ``images`` in any portion of it, it will be replaced to ``labels``, so the program will look for the annotations like 

```
<Path-to-labels>/frame-0001.txt
<Path-to-labels>/frame-0002.txt
<Path-to-labels>/frame-0003.txt
...
```

We include a simple matlab program to annotate the images based on the EgoDaily dataset.
To start training you can use the following command

```
./darknet detector train egoDailyDisamObj.data yoloEgoDailyDisamMask-obj.cfg darknet19_448.conv.23 -gpus 0 -clear -dont_show
```

The file ``yoloEgoDailyDisamMask-obj.cfg`` contains the Neural Network layers. The file ``darknet19_448.conv.23`` contains the prelearned weights. The ``-clear`` parameter resets the training. The ``-dont_show`` parameter doesn't show a grpah with the Neural Network training performance, but instead prints it out on the command line. 

During training the weights will be saved every 5,000 iterations under the backup folder.

After training you can test the YOLO detector using the following command

```
./darknet detector test egoDailyDisamObj.data yoloEgoDailyDisam-obj.cfg ./backup/yoloEgoDailyDisam-obj_final.weights -imWidth 1920 -imHeight 1080 < egoDailyTest.txt > yoloEgoDailyDisam.txt
```

The parameter ``-imWidth`` and ``-imHeight`` define the image sizes for the final detection outputs. The file ``egoDailyTest.txt`` contains the list of images for testing, with each line being
The file ``yoloEgoDailyDisam.txt`` will contain the final detections, with the 3 first lines being some Network outputs (not important) and then a series of lines for each image as follows:

```
Enter Image Path: 1egoDailyDatabase/images/subject1/bike/bike1/frame10032.jpg: Predicted in 0.319138 seconds.
845
20 0 129 76 0.000320 0
0 0 210 168 0.000195 0
0 0 601 531 0.000160 1
0 0 756 263 0.000186 1
```

The first line being the image being tested on. The second line being the number of detections YOLO generated. Finally a series of lines, each line having a single detection with the format ``x1 y1 x2 y2 score label``, in this example the file would contain 845 lines with the format.

Having trained two YOLO networks you can train their joint architecture by using the following command:

```
./darknet detector joint egoDailyDisamObj.data yoloEgoDailyDisam-obj.cfg ./backup/yoloEgoDailyDisam-obj_final.weights -secondW ./backup/yoloEgoDailyDisamMask-obj_final.weights -jointNet yoloEgoHandsJointLateDisam-obj.cfg -gpus 0 -clear -dont_show 
```

The parameter ``-secondW`` represents the second stream in the joint architecture, as both streams have the save architecture we only need to give one file for it. The parameter ``-jointNet`` represents the joint architecture, where the concatenation occurs and the final layers for the detection. In this file we find the definition of a new layer called ``concat`` that takes both streams and concatenates them, as follows:

```
[concat]
height=13
width=13
channels=1024
stopbackward=1
layer=-2
```

Where ``height``, ``width`` and ``channels`` are the streams output grid sizes. ``stopbackward`` means the back propagation stops here. ``layer=-2`` represents which layer is gonna be taken to concatenate from both YOLO streams architecture, being the final layer number ``net.num_layers + layer``. If the network has 30 layers in total the layer taken from the concat layer would be ``30-2 = 28``.

Finally, for testing the joint architecture you can use the following command:

```
./darknet detector jointTest egoDailyDisamObj.data yoloEgoDailyDisam-obj.cfg ./backup/yoloEgoDailyDisam-obj_final.weights -secondW ./backup/yoloEgoDailyDisamMask-obj_final.weights -secondNet yoloEgoDailyDisamMask-obj.cfg -gpus 0 -dont_show -clear -jointNet yoloEgoDailyJointLateDisam-obj.cfg -jointW ./backup/yoloEgoDailyJointLateDisam-obj_final.weights -fileName egoDailyTest.txt -imWidth 1280 -imHeight 720 > yoloEgoDailyJointLateDisam.txt
```

### Citing EgoDaily
If you find this code useful in your research, please consider citing:
```
@article{CRUZ2019131,
title = "Is that my hand? An egocentric dataset for hand disambiguation",
journal = "Image and Vision Computing",
volume = "89",
pages = "131 - 143",
year = "2019",
issn = "0262-8856",
doi = "https://doi.org/10.1016/j.imavis.2019.06.002",
url = "http://www.sciencedirect.com/science/article/pii/S026288561930085X",
author = "Sergio Cruz and Antoni Chan"
}
```