#Загрузка библиотеки OpenPose на колаб и обработка видео с YouYube.
import os
from os.path import exists, join, basename, splitext
import json
import numpy as np
#import io

git_repo_url = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  	!wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
  	!tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local
  	!git clone -q --depth 1 $git_repo_url
  	!sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt
  	!apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev
  	!pip install -q youtube-dl
  	!cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`
  
from IPython.display import YouTubeVideo

YOUTUBE_ID = "rkdkHqch57g" #"JlapgV9i8v0"#'RXABo9hm8B8'
YouTubeVideo(YOUTUBE_ID)

!rm -rf youtube.mp4
!youtube-dl -f 'bestvideo[ext=mp4]' --output "youtube.%(ext)s" https://www.youtube.com/watch?v=$YOUTUBE_ID #получение видео
!ffmpeg -y -loglevel info -i youtube.mp4 -ss 00:02:42 -t 7 video.mp4 #обработка видео
!rm openpose.avi
!cd openpose && ./build/examples/openpose/openpose.bin --video ../video.mp4 --write_json ../outputs --display 0  --write_video ../openpose.avi #детектирование позы
!ffmpeg -y -loglevel info -i openpose.avi output.mp4 # конвертирование видео

def file_name(num) :
    if (num <= 9) :
        file_name = "outputs/video_00000000000" + str(num) + "_keypoints.json"
    elif (num >= 10 and num <= 99) :
        file_name = "outputs/video_0000000000" + str(num) + "_keypoints.json"
    elif (num >= 100 & num <= 999) :
        file_name = "outputs/video_000000000" + str(num) + "_keypoints.json"
    return file_name

person_1 = []
#person_2 = []
#person_3 = []
number = 119 #размер json, зависит от длины видео
for num in range(number):
    with open(file_name(num)) as f:
        templates = json.load(f)
        person_1.append(np.array(templates["people"][0]["pose_keypoints_2d"]).reshape((25, 3))[:15])
        #person_2.append(np.array(templates["people"][1]["pose_keypoints_2d"]).reshape((25, 3))[:15])
        #person_3.append(np.array(templates["people"][2]["pose_keypoints_2d"]).reshape((25, 3))[:15])
print(person_1)
print()
#print(person_2)
#print(person_2)