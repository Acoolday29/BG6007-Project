import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Directories
gt_dir = Path("data/gt")
left_dir = Path("data/ppg-csv/Left")
right_dir = Path("data/ppg-csv/Right")

# Patient numbers
patient_ids = [f"10000{i}" for i in range(1,7)]

def load_and_process_data():    
    all_data = []
    for patient_id in patient_ids:
        print(f"Processing {patient_id}...")
        
        # Load gt data
        gt_file = gt_dir / f"{patient_id}.csv"
        if not gt_file.exists():
            print(f" gt file doesn't exist {gt_file}")
            continue
            
        gt_data = pd.read_csv(gt_file)
        
        # Load ppg data
        left_file = left_dir / f"{patient_id}.csv"
        right_file = right_dir / f"{patient_id}.csv"
        
        if not left_file.exists() or not right_file.exists():
            print(f"  ppg file doesn't exist")
            continue
            
        left_ppg = pd.read_csv(left_file)
        right_ppg = pd.read_csv(right_file)
        
        # Work on left hand ppg data
        left_processed = process_ppg_data(left_ppg, gt_data, patient_id, "left", ["SpO2 1", "SpO2 2"])
        all_data.extend(left_processed)
        
        # Work on right hand ppg data
        right_processed = process_ppg_data(right_ppg, gt_data, patient_id, "right", ["SpO2 4", "SpO2 5"])
        all_data.extend(right_processed)
    
    return pd.DataFrame(all_data)

