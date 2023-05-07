# DeepRaccess
This repository includes the implementation of "DeepRaccess: High-speed RNA accessibility prediction using deep learning". Please cite this paper if you use our code or system output.

In this package, we provides resources including: source codes of the DeepRaccess model, pretrained weights, train/test/predict module.

## 1. Environment setup
It is recommended to install in a virtual environment such as conda, docker or otherwise. We have confirmed that it works up to python 3.6.8 and python 3.8.12. Please refer to requiremets.txt for the modules our code requires. Also, at least one NVIDIA GPU is required for higher speeds.

1.1 Install the package and other requiremments
(Required)
```
git clone https://github.com//DeepRaccess
cd DeepRaccess
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Train (Skip this section if you only want to make predictions)
2.1 Data preparation
Training data can be artificially created.
create fasta files like the sequence directory of sample_data.
Use an existing accessibility prediction tool (e.g. Raccess) to create a csv file like the accessiblity directory of sample_data.

2.2 Model training
```
export SEQ_DIR=sample_data/train/sequence/
export ACC_DIR=sample_data/train/accessibility/

python3 train.py \
    --seqdir ${SEQ_DIR} \
    --accdir ${ACC_DIR} \
    --epoch 10 \
    --model FCN \
```
You can check the options with `python train.py --help`.
After successful training, trained weights (.pth), scatter plot (.png) and logs (.log) are created.

## 3. Test (Skip this section if you only want to make predictions)
3.1 Preparation of learned weights
You can test the performance of DeepRaccess using the learned weights.
Download the pre-trained weights [here](https://drive.google.com/drive/folders/1xJOV2vIoVYCx6i9YY70CWwEGacQw8jTP?usp=sharing) (FCN_***.pth is the highest accuracy)
Then place them in the path directory.

3.2 Model test
```
export SEQ_FILE=sample_data/sequence/RF01000.fa
export ACC_FILE=sample_data/accessibility/RF01000.csv
export OUT_FILE=output.csv

python3 test.py \
    --seqfile ${SEQ_FILE} \
    --accfile ${ACC_FILE} \
    --outfile ${OUT_FILE} \
    --model FCN \
    --pretrain path/FCN_structured.pth \
```
You can check the options with `python test.py --help`.
After successful test, output accessibility(OUT_FILE/output.csv) and scatter plot (.png) are created.

## 4. Prediction
4.1 Preparation of learned weights (same as 3.1)
You can test the performance of DeepRaccess using the learned weights.
Download the pre-trained weights [here](https://drive.google.com/drive/folders/1xJOV2vIoVYCx6i9YY70CWwEGacQw8jTP?usp=sharing) (FCN_***.pth is the highest accuracy)
Then place them in the path directory.

4.2 Predicting Accessibility

```
export SEQ_FILE=sample_data/sequence/RF01000.fa
export OUT_FILE=output.csv

python3 predict.py \
    --seqfile ${SEQ_FILE} \
    --outfile ${OUT_FILE} \
    --model FCN \
    --pretrain path/FCN_structured.pth
```
You can check the options with `python predict.py --help`.
After successful test, output accessibility file(OUT_FILE/output.csv) is created.
