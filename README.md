# DeepRaccess
This repository includes the implementation of DeepRaccess, high-speed RNA accessibility prediction on GPU environmentsusing deep learning. Please cite the following paper if you use our code or system output.

Kaisei Hara, Natuki Iwano, Tsukasa Fukunaga and Michiaki Hamada. "DeepRaccess: High-speed RNA accessibility prediction using deep learning." (under submission)

In this package, we provide the source code of the DeepRaccess software, pre-trained models, and modules for training, test and prediction.　For instructions on how to use it, please refer to the explanation below, as well as the `demo.ipynb` file which provides an end-to-end workflow demonstration.

## 1. Environment setup
We recommend the use of a virtual environment such as Conda or Docker for installation. We have tested compatibility with Python versions 3.6.8 and 3.8.12. Please refer to `requirements.txt` for the additional modules required by DeepRaccess. Note that, for faster computation of accessibility, at least one NVIDIA GPU is required.

#### 1.1 Installation of the package and the other requirements
(Required)
```
git clone https://github.com/hmdlab/DeepRaccess.git
cd DeepRaccess
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Training (You can skip this section when you only use the pre-trained DeepRaccess model)
#### 2.1 Data preparation
Please prepare an input sequence file as a fasta file such as `sample_data/sequence/***.fa`.  
In addition, please prepare the accessibility file as a csv file like `sample_data/accessibility/***.csv`., by using an existing accessibility prediction tool (e.g. Raccess).

#### 2.2 Model training
```
export SEQ_DIR=sample_data/train/sequence/
export ACC_DIR=sample_data/train/accessibility/

python3 train.py \
    --seqdir ${SEQ_DIR} \
    --accdir ${ACC_DIR} \
    --epoch 10 \
    --model FCN \
```
You can check the software options with `python train.py --help`.

Successful training produces trained weights (`.pth`), scatter plots (`.png`) and logs of the learning (`.log`) for the model.

## 3. Test (You can skip this section when you only use the pre-trained DeepRaccess model)
#### 3.1 Preparation of trained weights

You can test the performance of the trained weights of DeepRaccess.  Please place the learned weights in the `path/` folder. You can use pre-trained weights in the `path/` folder.

#### 3.2 Model test
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
You can check the software options with `python test.py --help`. Successful test produces the predicted accessibility (OUT_FILE/`output.csv`) and scatter plot (`scatter.png`).

## 4. Prediction
#### 4.1 Preparation of trained weights (as in 3.1)
Please place the learned weights in the `path/` folder. You can use pre-trained weights in the `path/` folder.

#### 4.2 Predicting Accessibility

```
export SEQ_FILE=sample_data/sequence/RF01000.fa
export OUT_FILE=output.csv

python3 predict.py \
    --seqfile ${SEQ_FILE} \
    --outfile ${OUT_FILE} \
    --model FCN \
    --pretrain path/FCN_structured.pth
```
You can check the software options with `python test.py --help`.

Successful test produces the predicted accessibility (OUT_FILE/`output.csv`).

## 5. Brief explanation of interpretation
We used the fully convolutional network (FCN) for the network architecture. FCN does not use fully connected layers and is composed only of convolutional layers. We used a network of 40 convolutional layers with constant channel and unit sizes. The input RNA sequences are embedded into numerical vectors and fed into the neural networks. Our embedding first randomly generates six 120-dimensional numerical vectors corresponding to each of the six states: four RNA bases (A, C, G, U), one undetermined nucleotide (N) and padding. Please see our paper for implementation details.