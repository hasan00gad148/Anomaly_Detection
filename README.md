# Real-time Anomaly Detection System

This project implements a real-time anomaly detection system using Python. It generates synthetic time series data, detects anomalies, and visualizes the results in real-time.

## Features

- Real-time data generation with daily seasonality and random anomalies
- Anomaly detection using a sliding window and z-score thresholding
- Live plotting of normal and anomalous data points
- User-controlled program termination

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/realtime-anomaly-detection.git
   cd realtime-anomaly-detection
   ```

2. Install the required packages:
   ```
   pip install numpy matplotlib
   ```

## Usage

Run the main script:

```
python anomaly_detection.py
```

The program will start generating data and displaying a live plot. To stop the program, type 'quit' in the console.


To Quit  the program, type 'quit' in the terminal
and close the plots

## Code Structure

- `AnomalyDetector` class: Implements the anomaly detection algorithm
- `generate_data` function: Creates synthetic time series data
- `input_thread_function`: Handles user input for program termination
- `main` function: Orchestrates the data generation, anomaly detection, and visualization

## How It Works

1. The program generates synthetic data that includes:
   - A base sine wave with a period of 1 hour
   - Daily seasonality
   - Random noise
   - Occasional anomalies

2. in the stat module The `AnomalyDetector` class uses a sliding window approach to detect anomalies:
   - It maintains a fixed-size window of recent values
   - For each new value, it computes the z-score based on the window's mean and standard deviation
   - If the z-score exceeds a threshold, the value is flagged as an anomaly

3. in the ML module The `AnomalyDetector` class uses a sliding window approach to detect anomalies:
   - It maintains a fixed-size window of recent values
   - IsolationForest model is trained on this window of data
   - IsolationForest predict if the new value is anomaly or not

4. The main loop:
   - Generates a new data point
   - Checks if it's an anomaly
   - Updates the plot in real-time

5. The program runs until the user types 'quit' in the console

6. After termination, it displays a final plot showing all data points

## Customization

You can modify the following parameters in the code:

- `window_size` in `AnomalyDetector.__init__`: Changes the size of the sliding window
- `threshold` in `AnomalyDetector.__init__`: Adjusts the sensitivity of anomaly detection
- Parameters in `generate_data`: Modify the characteristics of the generated data

