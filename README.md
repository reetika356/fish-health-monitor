# 🐟 Fish Health Monitoring System

A Streamlit web app that classifies a fish photo as **Healthy** or **Sick** using a CNN, with an optional Gmail alert when a sick fish is detected.

## Features
- Upload a fish image and get an instant prediction with a confidence score
- Optional email alert via Gmail SMTP (uses an App Password, nothing is hardcoded)

## Model
- CNN built from scratch (3 Conv + BatchNorm + MaxPool blocks, GlobalAveragePooling, Dropout)
- Input: 128x128 RGB, about 26K parameters
- Trained on 305 images (163 healthy, 142 sick) from a public Kaggle fish disease dataset
- Held-out test accuracy: about 70% (varies between runs, roughly 62-70%, because the test set is small)
- Limitation: recall on sick fish is low (about 0.41), so it misses many sick fish. This is a prototype, not a diagnostic tool.

## Run locally
1. Install Python and download this repository
2. Open a terminal in the project folder and run:
