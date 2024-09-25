import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from collections import deque
import time
import threading




class MLAnomalyDetector:
    def __init__(self, window_size=100, contamination=0.05):
        self.window_size = window_size
        self.contamination = contamination
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def update(self, value, timestamp):
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        if len(self.values) == self.window_size:
            self.model.fit(np.array(self.values).reshape(-1, 1))
        
    def is_anomaly(self, value):
        if len(self.values) < self.window_size:
            return False
        
        prediction = self.model.predict(np.array([[value]]))
        return prediction[0] == -1  # -1 indicates an anomaly




def generate_data(t):
    # Base signal: sine wave with period of 1 hour
    base = 10 * np.sin(2 * np.pi * t / 3600)
    
    # Add daily seasonality
    daily = 5 * np.sin(2 * np.pi * t / 72000)
    
    # Add noise
    noise = np.random.normal(0, 1)
    
    # Combine components
    value = base + daily + noise
    
    # Randomly introduce anomalies (about 5% of the time)
    if np.random.random() < 0.05:
        value += np.random.choice([-1, 1]) * np.random.uniform(15, 25)
    
    return value





def input_thread_function(stop_event):
    while True:
        if input().lower() == 'quit':
            stop_event.set()
            break



def main():
    detector = MLAnomalyDetector()
    
    stop_event = threading.Event()

    # Start the input thread
    input_thread = threading.Thread(target=input_thread_function, args=(stop_event,))
    input_thread.daemon = True
    input_thread.start()



    # Set up the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line_normal, = ax.plot([], [], 'b-', label='Normal')
    line_anomaly, = ax.plot([], [], 'r*', markersize=10, label='Anomaly')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('Real-time ML-based Anomaly Detection')
    ax.legend()
    
    normal_x, normal_y = [], []
    anomaly_x, anomaly_y = [], []
    
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            current_time = time.time() - start_time
            value = generate_data(current_time)
            
            if detector.is_anomaly(value):
                anomaly_x.append(current_time)
                anomaly_y.append(value)
            else:
                normal_x.append(current_time)
                normal_y.append(value)
            
            detector.update(value, current_time)
            
            # Update the plot
            line_normal.set_data(normal_x, normal_y)
            line_anomaly.set_data(anomaly_x, anomaly_y)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            
            time.sleep(0.1)  # Simulate real-time data stream
        print("Stopping the data stream...")    
    except Exception:
        print("An Error accurd Stopping the data stream...")
    finally:
        plt.ioff()
        plt.plot(normal_x, normal_y, 'b-', label='Normal')
        plt.scatter(anomaly_x, anomaly_y, c='r', marker='*', label='Anomaly')
        plt.legend()
        plt.show()
        plt.plot(normal_x, normal_y)
        plt.scatter(anomaly_x,anomaly_y,c="r")
        plt.draw()
        plt.show()


if __name__ == "__main__":
    main()