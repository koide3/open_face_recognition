# open_face_recognition

## Description
This is a wrapper ROS package for face embedding. As a face embedding backend, you can choose either of [Facenet](https://github.com/davidsandberg/facenet) or [OpenFace](https://github.com/cmusatyalab/openface); they implement a face embedding network based on the triplet loss function. The network takes a face image and outputs a 128D or 512D real vector which represents the input face.

## Usage
```bash
rosrun open_face_recognition open_face_recognition.py
# To use openface
# rosrun open_face_recognition open_face_recognition.py _embedding_framework:=openface
```

```bash
roscd open_face_recognition/thirdparty/facenet/data/images
rosrun open_face_recognition compare_images.py Anthony_Hopkins_0001.jpg Anthony_Hopkins_0002.jpg
```

## ROS services
- CalcImageFeatures takes a face image and returns the embedded face feature vector
- CompareImages takes two face images and then calculates the distance between them and returns the distance
Usually, the distance between a positive face pair is less than 0.75.

## Installation

```bash
cd catkin_ws/script
git clone https://github.com/koide3/open_face_recognition --recursive
```

```bash
cd open_face_recognition/install_scripts
./install_dlib.sh
./download_face_models.sh
```

### [Facenet](https://github.com/davidsandberg/facenet)

```bash
cd open_face_recognition/install_scripts
./install_facenet.sh
source ~/.bashrc
```

### [OpenFace](https://github.com/cmusatyalab/openface)

```bash
cd open_face_recognition/install_scripts
./install_torch.sh
./install_openface.sh
```

The above script installs torch at ~/torch. It also installs dlib openface on python.

If you cannot use dpnn on torch,
```bash
cd ~/torch/install/bin
sudo ./luarocks install torch
sudo ./luarocks install nn
sudo ./luarocks install dpnn
```

