# YOLO-Context
This is part of the code used on the [EgoDaily](https://github.com/sercruzg/EgoDaily) dataset. This code is based on the [YOLO](https://github.com/pjreddie/darknet) code.

This code masks the ground truth out after the convolutions, leaving no information left. This forces the Neural Network to use the surrounding objects to learn.

To start training you can use the following command

```
./darknet detector train egoDailyDisamObj.data yoloContext-obj.cfg darknet19_448.conv.23 -gpus 0 -clear -dont_show
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
