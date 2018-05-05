## Calligraphy Evaluation
Evaluating Brush Movements for Chinese Calligraphy: A Computer Vision Based Approach

### environment
Python 3.6

tensorflow 1.4.1

Keras 2.1.2

sklearn 0.19.1

image size is 1920*1080

### Install
git clone --recursive https://github.com/excitingx/Calligraphy-Evaluation

### Usage
1. preprocessing:
- framing: split video into frames
- image2vec: convert the images to a vector and label it
2. recognization:
MCNN-LSTM_WriteStateReg.py is the main the process of recognization.
3. evaluation:
- similar_feature_data.csv is the similarity data of the writing traces between the strokes of the trainee and the teacher.
- artificial_score.csv is the scorec by teachers.
- score_regression.py is the regression model used to predict the score.