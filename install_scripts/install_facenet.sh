#!/bin/bash

cd ..
echo "PYTHONPATH=\$PYTHONPATH:$(pwd)/thirdparty/facenet/src" >> ~/.bashrc

sudo pip install -r $(pwd)/thirdparty/facenet/requirements.txt

cd data
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-" -o facemodel.zip

unzip facemodel.zip && mv 20180402-114759 facenet
rm facemodel.zip

