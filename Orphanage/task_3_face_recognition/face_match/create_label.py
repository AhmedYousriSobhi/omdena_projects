"""
    Author: Ahmed Sobhi
    Department: Data Science
    Created_at: 2023-09-10
    Objective: Generate Label CSV file.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Generate Label.')

# Define command-line arguments with default values
parser.add_argument('--label_path', default='data/label/', help='Label directory path (default: /data/label)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
label_path = args.label_path

if __name__ == '__main__':
    for dir_name in os.listdir(label_path):
        print(dir_name)
