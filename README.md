# Exercises for CAS Deep Learning - Module Computer Vision with Deep Learning (Part 1)

This repository is used for the distribution of exercises for the CAS Deep Learning - Module Computer Vision with Deep Learning (Part 1).


There are several ways to work on the assignments:

- Google Colab (easiest)
- pip  (local install)
- Docker


## Google Colab

Use Google Colab by clicking on the links below.


### Exercise 00 - Environment Check

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marco-willi/cas-dl-compvis-exercises-hs2024/blob/main/notebooks/00_env_check/env_check.ipynb)


## Local

```
pip install .
```

## Docker

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
docker run -it -d \
    -p 8880:8880 \
    -p 6006:6006 \
    --gpus=all \
    --shm-size 12G \
    -v MY_ML_DIR:/workspace/code \
    --name=cas_dl_computer_vision_part1 \
    mwilli13/cas-dl-compvis-exercises-hs2024:latest
```

### 6. Check that your container is running

```
docker ps -a
```

### 7. Connect to your container through your browser

Enter `http://localhost:8880/lab` in your browser.

