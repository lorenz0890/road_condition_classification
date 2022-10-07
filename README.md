# Classification of Road Conditions and Types Based on Mobile Devices Telemetry Data

Implementation of [BSc Thesis](https://github.com/lorenz0890/road_condition_classification/blob/master/BSc_Kummer.pdf).

## Abstract

While efficient and accurate solutions for detecting known patterns in
time series exist for at least for two decades, the task of labeling
real world data as required by applications found in insurance companies still leaves many potentially promising options unexplored.
Based on the Sussex-Huawei Locomotion and Transportation Dataset,
this work compares the quality of statistically extracted features,
grammar and similarity based motif discov-
ery with one another as well as the results found during the Sussex-Huawei
Locomotion Challenge. In a machine learning approach carefully en-
gineered based on existing solutions to similar problems, we were
able show that the classification quality of the features extracted by the
motif based approach are at least equal to the statistical approach and
more robust to the change point detection problem, achieving a clas-
sification rate of up to 85.2% in a binary road type classification problem
(City, Countryside) without change point detection, while the features
engineered by the statistical approach only resulted in a classification
accuracy of 79.0% without and 85.2% with simulated change point de-
tection. In a first step, the data is preprocessed and normalized, then
the features are extracted and backwards selected using wrapper or filter
methods. Finally, the classification quality if evaluated for a num-
ber of different hyper-parameter optimized classifiers (Random Forests,
CART-Trees, SVM).
