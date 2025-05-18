# Team 9 - CPE 313 Final Project
This is a repository containing the final project of Team 9 for their CPE 313 - Advanced Machine Learning and Deep Learning course

# Members
- Belocora, John Rome - qjrabelocora@tip.edu.ph
- De Guzman, Jemuel Endrew - qjedeguzman03@tip.edu.ph

# Description
- This app is for deploying our 2-stage model used for our research paper, Anomalous Packet Detection for Public WiFi Services.
- The data we are using is a cleaned version of the CICDDos2019 dataset, where the Benign samples are increased and the irrelevant columns are dropped beforehand
- Once an anomaly is detected, the stream of packets are then converted into an image for classification
- It consists of 2 models: A Simple Autoencoder for anomaly detection and a CNN for image classification
