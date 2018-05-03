# TensorFlow implementation of Pose Guided Person Image Generation (NIPS 2017 paper)
Link to original paper: https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf

This is an unofficial implementation by Zhou He, Shaopeng Guo, Ziyu Wang and Xinyuan Yu from HKUST.

# Clone this repository
```
git clone https://github.com/samuelzhouhe/poseGuidedImgGeneration.git
cd poseGuidedImgGeneration
```

# Install pip3 and required libraries
```
sudo apt-get install python3-pip python3-dev
pip3 install -r requirements.txt
```


# Download dataset and keypoints (Please observe all requirements set by the dataset owner)
Download the dataset DeepFashion from http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html. (The dataset downloaded will be password-protected, and please kindly contact the owner of this dataset for password)       

Then put the folder ```In-shop Clothes Retrieval Benchmark/``` inside the project directory (```poseGuidedImgGeneration/```), and then rename ```In-shop Clothes Retrieval Benchmark/``` to ```dataset/```. Then extract the ```img.zip``` file in the ```dataset/Img/``` directory.      

Then download the keypoint locations file ```img-keypoints.zip``` prepared by us (using OpenPose, CVPR2017) from https://drive.google.com/file/d/1DwRPXCyVYBmtGa0hO3JlYkrD709s6zca/view?usp=sharing, put it inside the directory ```dataset/```, and extract it.  

Your file directory should now look like this:    

|--poseGuidedImageGeneration     
&nbsp;&nbsp;&nbsp;|--dataset     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--Anno    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--Eval   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--Img      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--img  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--img-keypoints  


# Use pre-trained model
Download ```model.tar.gz``` from https://drive.google.com/file/d/1z0mtWRSy_ObQ5NfXwXwcT9mIipIPxBsY/view?usp=sharing to project folder, then:
```
tar -xvzf model.tar.gz
rm -rf logs
mv model logs
python3 demo.py
```
# Train from scratch
```
rm -rf logs
python3 trainall.py
```

# Descriptions of source files

```config.py```: hyperparameters used for our network

```dataset_reader.py```: load training image data batches and process them for training

```model_all.py```: build G1, G2 & D in TensorFlow

```network.py```: helper class for building complicated networks

```read_keypoint.py```: adopt keras realtime multi-person pose estimation model to produce heatmap of human poses (has been done by us)

```trainall.py```: the main training procedure including data preprocessing

```demo.py```: use the pre-trained model to demo our result

```dataset/```: the directory which will contain the dataset after you finish downloading from both DeepFashion and our Google Drive link.

System tested: Linux (Ubuntu 16.04)
