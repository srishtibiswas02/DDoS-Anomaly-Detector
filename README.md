# DDoS-Anomaly-Detector
**Project Description**

This project is a Distributed Denial of Service (DDoS) Anomaly Detection System designed to identify unusual network traffic patterns indicative of DDoS attacks in real-time. Leveraging machine learning techniques, the system uses Random Forest model to classify network traffic as either "Benign" or "Attack." The model was trained on a wide dataset, ensuring that the system can effectively distinguish between normal and anomalous traffic patterns.

The system is deployed as a web-based application using Flask. Users can upload a .pcapng file containing network traffic data, which is then processed on the backend. The data goes through feature extraction and preprocessing, followed by predictions from the trained models. Results are displayed on a dedicated results page, featuring visualizations (pie charts and bar graphs) that summarize the distribution of benign vs. attack traffic. This helps users quickly understand the nature and extent of anomalies detected.

**Key Features**

**Machine Learning Models:** _Utilizes an Autoencoder and Random Forest model for anomaly detection, with features extracted from real network traffic data._
Feature Extraction and Preprocessing: Processes network traffic captured in .pcapng files, extracting relevant features for model input._

**Web Interface:** _Built using Flask, providing an intuitive interface for users to upload files and view results._

**Data Visualization:** _Displays results in the form of pie charts and bar graphs to help users interpret the predictions and network traffic composition._

**Real-Time Detection:** _Aims to provide immediate feedback on potential DDoS attacks based on uploaded traffic data.
Project Structure_
**Frontend:** _HTML, CSS, and JavaScript for a user-friendly web interface, including upload functionality and visualization._

**Backend:** _Flask for routing, feature extraction, and model inference, using test.py for feature extraction and test_pred.py for prediction._

**Model:** _Random Forest Classifier was used due to its high accuracy and speed while working with large datasets._
