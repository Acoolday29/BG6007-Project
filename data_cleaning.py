import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Copied from data_cleaning because got confused by the file format
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

def process_ppg_data(ppg_data, gt_data, patient_id, hand, spo2_columns):
    processed_data = []

    # Align gt and ppg data 
    min_length = min(len(ppg_data), len(gt_data) * 30)
    num_seconds = min_length//30

    for second in range(num_seconds):
        start_frame = second * 30
        end_frame = start_point + 30
        if end_frame > len(ppg_data):
            break

        # Calculate the average RGB values for 30 frames
        frame_data = ppg_data.iloc[start_frame:end_frame]
        avg_r = frame_data['R'].mean()
        avg_g = frame_data['G'].mean()
        avg_b = frame_data['B'].mean()

        # Get corresponding gt data
        if second < len(gt_data):
            gt_row = gt_data.iloc[second]

            # For each SpO2 reading, create a summary line. Also, exclude N/A values
            for spo2_col in spo2_columns:
                if spo2_col in gt_row and pd.notna(gt_row[spo2_col]) and gt_row[spo2_col] > 0:
                    processed_data.append({
                        'patient_id': patient_id, 'R': avg_r, 'G': avg_g, 'B': avg_b, 'which_hand': hand, 'corresponding_SpO2': gt_row[spo2_col]})
    
    return processed_data

def main():
    df = load_and_process_data()

    if df.empty:
        print("Error, not receiving any info")
        return

    output_file = "cleaned_data.csv"
    df.to_csv(output_file, index = False)
    
    return df

if __name__ == "__main__":
    df = main()
