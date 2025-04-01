# Image Processing 
A bunch of problem statements and experiments involving digital image processing and simple classification, done as a part of Assignment 1 of the Computer Vision course (IIIT-Hyderabad, Spring '25). 

## Setup 
The conda environment file is available at `docs/env.yml`.  
```sh 
conda env create -f env.yml
```

Alternatively, install the dependencies by referring to those in `docs/env-hist.yml`. 

## Problems  

### Cloverleaf Bridge 

The initial image is available in `./Cloverleaf Bridge/data/cloverleaf_interchange.png`. 

<img src="Cloverleaf Bridge/data/cloverleaf_interchange.png" alt="initial cloverleaf interchange" width="300">

#### Results 
The final image with contour borders marked is as below: 

<img src="./Cloverleaf Bridge/res/contour borders.png" alt="contour borders" width="300">

The circles within the contours are marked below:

<img src="./Cloverleaf Bridge/res/marked circles.png" alt="contour circles" width="300">

#### Run 
To see the images along with all intermediate pre-processing, run as follows: 
```sh 
cd Cloverleaf\ Bridge
python -m main
```

#### Further Details 
See the [report](./Cloverleaf%20Bridge/report.pdf).

### Line Segmentation in Historical Documents 

The initial image is available in `./Line Segmentation in Historical Document/data/historical-doc.png`. 

<img src="./Line Segmentation in Historical Document/data/historical-doc.png" alt="initial historical doc" width="300">

#### Results 
The final image with text segmentation is as below:

<img src="./Line Segmentation in Historical Document/res/bounding boxes.png" alt="initial historical doc" width="300">

The final image with line wise segmentation is as below:

<img src="./Line Segmentation in Historical Document/res/line-wise bounding boxes.png" alt="initial historical doc" width="300">

The final image with polygonal text boundaries is: 


<img src="./Line Segmentation in Historical Document/res/polygons.png" alt="initial historical doc" width="300">

#### Run 

```sh
cd "Line Segmentation in Historical Document"
python -m main 
```

#### Further Details 
See the [report](./Line%20Segmentation%20in%20Historical%20Document/report.pdf).

### MLP Classification on Image Features 
to run mlp classification:
```sh 
python -m src.main --transform <transform> 
```

before that process the data using:
```sh 
python scripts/process.py 
```
