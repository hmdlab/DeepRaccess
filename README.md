# DeepRaccess
This repository includes the implementation of "DeepRaccess: High-speed RNA accessibility prediction using deep learning". Please cite this paper if you use our code or system output.

In this package, we provides resources including: source codes of the DeepRaccess model, pretrained weights, train/test/predict module.

## 1. Environment setup
Our code is written with Python3.8.12. Please refer to requiremets.txt for the modules our code requires.
```
git clone https://github.com//DeepRaccess
cd DeepRaccess
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. train (Skip this section if you only want to make predictions)
```
export SEQ_DIR=sample_data/train/sequence/
export ACC_DIR=sample_data/train/accessibility/

python3 train.py \
    --seqdir ${SEQ_DIR} \
    --accdir ${ACC_DIR} \
    --epoch 10 \
    --model FCN \
```

## 3. test (Skip this section if you only want to make predictions)
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

## 4. prediction
```
export SEQ_FILE=sample_data/sequence/RF01000.fa
export OUT_FILE=output.csv

python3 predict.py \
    --seqfile ${SEQ_FILE} \
    --outfile ${OUT_FILE} \
    --model FCN \
    --pretrain path/FCN_structured.pth
```