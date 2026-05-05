import time
import os
import numpy as np

class Logger:
    def __init__(self, log_file):
        # Initialize the logger with a given file to save logs
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
    def __init__(self, log_path, run_name):
        # Create a log file with the run name and current timestamp
        self.log_file = f"{log_path}/{int(time.time())}_{run_name}.txt"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def onlylog(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    def logandprint(self, message):
        print(message)
        self.onlylog(message)
