# open_face_recognition

## Description
This is a wrapper node of OpenFace, a state-of-the-art face recognition framework.
OpenFace provides an implementation of FaceNet, a deep convolutional neural network for face embedding.
The network takes a face image and outputs a 128D real vector which represents the input face.
[OpenFace](https://cmusatyalab.github.io/openface/)

## Usage
```
rosrun open_face_recognition open_face_recognition.py
rosrun open_face_recognition compare_images.py (for demo)
```

## ROS services
- CalcImageFeatures takes a face image and returns the embedded face feature vector
- CompareImages takes two face images and then calculates the distance between them and returns the distance
Usually, the distance between a positive face pair is less than 0.75.

## Install
```
cd open_face_recognition/install_scripts
./install_all.sh
```
The above script installs torch at ~/torch. It also installs dlib openface on python.

If you cannot use dpnn on torch,
```
cd ~/torch/install/bin
sudo ./luarocks install torch
sudo ./luarocks install nn
sudo ./luarocks install dpnn
```