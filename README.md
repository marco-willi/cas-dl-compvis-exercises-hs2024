# Exercises for CAS Deep Learning - Module Computer Vision with Deep Learning (Part 1)

This repository is used for the development and distribution of exercises for the CAS Deep Learning - Module Computer Vision with Deep Learning (Part 1).


There are several ways to work on the assignments:

- Google Colab (easiest)
- local - pip install (not tested)


## Google Colab

Use Google Colab by clicking on the links below.


### Exercise 00 - Environment Check

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-willi/cas-dl-compvis-exercises-hs2024/blob/main/assignments/00_env_check/env_check.ipynb)


### Exercise 01 - Machine Learning Recap

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-willi/cas-dl-compvis-exercises-hs2024/blob/main/assignments/01_ml_recap/machine_learning_recap.ipynb)


### Exercise 02 - PyTorch & Machine Learning

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-willi/cas-dl-compvis-exercises-hs2024/blob/main/assignments/02_pytorch/pytorch.ipynb)


### Exercise 03 - Neuronale Netzwerke

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-willi/cas-dl-compvis-exercises-hs2024/blob/main/assignments/03_neural_networks/neural_networks.ipynb)


### Exercise 04 - Convolutional Neural Networks

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-willi/cas-dl-compvis-exercises-hs2024/blob/main/assignments/04_cnns/cnns.ipynb)


### Exercise 05 - Image Classification

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-willi/cas-dl-compvis-exercises-hs2024/blob/main/assignments/05_classification/classification.ipynb)



## Local - pip (not tested)

```
pip install -r requirements.txt
```


## Local - Docker

### 1. Install Docker on your computer

Depending on your operating system you have to install docker in different ways.  

You'll find detailed instructions here: https://docs.docker.com/get-docker


### 2. Pull the Docker image

```
# pull the image
docker pull mwilli13/cas-dl-compvis-exercises-hs2024:latest
```

### 3. Fork this repository

Fork this repository by pressing the fork button on the upper right.

### 4. Clone your fork to your computer. 

Clone into your ml directory (`MY_ML_DIR`) using:

```
git clone MY_REPO_FORK_HTTPS_ADDRESS
```

### 5. Start a ml container on your machine

```
# Replace 'MY_ML_DIR' with your local code directory
$ docker run -d \
    -p 8877:8888 \
    --user root \
    -v MY_ML_DIR:/home/jovyan/work/ \
    --name=cas_dl_computer_vision_part1 \
    mwilli13/cas-dl-compvis-exercises-hs2024:latest start.sh jupyter lab --LabApp.token=''
```

### 6. Check that your container is running

```
docker ps -a
```

### 7. Connect to your container through your browser

Enter `http://localhost:8877/lab` in your browser.

