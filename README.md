# Image Processing

A bunch of problem statements and experiments involving digital image processing and simple classification, done as a part of Assignment 1 of the Computer Vision course (IIIT-Hyderabad, Spring '25). The assignment details can be found [in the assignment document](./docs/CV_S25_A1.pdf).

## Problems

### Cloverleaf Bridge

#### Results

| Initial Image                                                                                                    | Contour Borders                                                                           | Marked Circles                                                                  |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| <img src="./Cloverleaf Bridge/data/cloverleaf_interchange.png" alt="initial cloverleaf interchange" width="300"> | <img src="./Cloverleaf Bridge/res/contour borders.png" alt="contour borders" width="300"> | <img src="./Cloverleaf Bridge/res/radii.png" alt="contour circles" width="300"> |

#### Run

To see the images along with all intermediate pre-processing, run as follows:

```sh
cd Cloverleaf\ Bridge
python -m main
```

#### Further Details

See the [report](./Cloverleaf%20Bridge/report.pdf).

### Line Segmentation in Historical Documents

#### Results

| Initial Image                                                                                                           | Text Segmentation                                                                                                 | Line-wise Segmentation                                                                                                           | Polygonal Boundaries                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| <img src="./Line Segmentation in Historical Document/data/historical-doc.png" alt="initial historical doc" width="300"> | <img src="./Line Segmentation in Historical Document/res/bounding boxes.png" alt="text segmentation" width="300"> | <img src="./Line Segmentation in Historical Document/res/line-wise bounding boxes.png" alt="line-wise segmentation" width="300"> | <img src="./Line Segmentation in Historical Document/res/polygons.png" alt="polygonal boundaries" width="300"> |

#### Run

```sh
cd "Line Segmentation in Historical Document"
python -m main
```

#### Further Details

See the [report](./Line%20Segmentation%20in%20Historical%20Document/report.pdf).

### MLP Classification on Image Features

#### Setup

Download the data from [here](https://drive.google.com/drive/folders/1iMyT9emFeoJjuqYdDJkRb07TbbsQuGQE?usp=sharing) and place the files in `./MLP Classification on Image Features/raw/`.

Process the data as below:

```sh
cd "MLP Classification on Image Features"
python scripts/process.py
```

The above script stores the data in the form of 28x28 images and splits the train data into train and val sets.

#### Run

To run the model, select a transform from among:

- **raw**: raw images
- **edge_detect**: edge detection features
- **blur_equal**: blurred and equalised images
- **hog_feat**: hog features
  The model then performs classification on these features

Other arguments such as `batch_size`, `epochs`, `lr`, and `dropout` can also be set.

```sh
cd "MLP Classification on Image Features"
python -m src.main --transform <transform>
```

#### Results

The model results on blurred and equalised images with a dropout of 0.2 can be seen below:

![blur_equal 0.2](./MLP%20Classification%20on%20Image%20Features/res/dr0.2/model-blurred_equalised.png)

Other details such as the test loss, and a number of ablations can be found in the [report](./MLP%20Classification%20on%20Image%20Features/report.pdf) and in the [res dir](./MLP%20Classification%20on%20Image%20Features/res/).

## Setup

The conda environment file is available at `docs/env.yml`.

```sh
conda env create -f env.yml
```

Alternatively, install the dependencies by referring to those in `docs/env-hist.yml`.
