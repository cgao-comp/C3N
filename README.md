# **C3N**

The code related to the paper below: C3N: A Cross-modal Content Correlation Network for Enhancing Multimodal Fake News Detection

## **Data**

The two datasets used for this project are publicly accessible, and their links are provided below.

[weibo](https://github.com/yaqingwang/EANN-KDD18)

[twitter](https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2015)

In this project, due to upload size limitations for some data, the files in `/weibo/processed/crops/` used can be downloaded via [Google Drive](https://drive.google.com/file/d/1HyXQhoeHKEe_MILui--mra9fKA449c2S/view?usp=sharing).

# Requirements

We train our model on Python 3.7.0 and Pytorch 1.8.0. And your environment should have some packages as follows:

```
clip==1.0
cn_clip==1.5.1
importlib_metadata==6.0.0
langdetect==1.0.9
matplotlib==2.2.4
numpy==1.21.6
opencv_python==4.7.0.68
pandas==1.1.5
Pillow==9.5.0
scikit_learn==1.0.2
seaborn==0.12.2
torchtext==0.15.2
tqdm==4.64.1
transformers==4.25.1
```

# Run

After installing the environment, in the code directory, modify the save paths in `main.py`, `process_image_weibo.py`, and `process_text_weibo.py`.

First, execute `process_text_weibo.py` to obtain the complete text preprocessing results.

Next, run `process_image_weibo.py` to obtain the image preprocessing results.

Run `main.py` to perform the training process.
