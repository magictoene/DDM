# Posturography Stability Analysis: IMU to CoM Exceedance

This project utilizes IMU sensor data to calculate **Center of Mass (CoM) displacement** and assess postural stability by measuring the percentage of time the CoM trajectory exceeds statistically derived benchmark ellipses.

The pipeline is split into two scripts:
1.  **`pipeline.py`**: Handles core data processing, orientation estimation (EKF), and displacement calculation.
2.  **`ellipse_exceedance.py`**: Manages the multi-trial workflow, computes benchmark ellipses, and calculates the final stability metric (**threshold exceedance**).

## ⚙️ Setup and Installation

### Prerequisites

You need **Python 3.x** installed, along with the following required libraries:

* `numpy` (for numerical operations)
* `pandas` (for data handling)
* `scipy` (for signal filtering)
* `matplotlib` (for plotting/visualization)
* `ahrs` (for the Extended Kalman Filter - EKF)

### Installation

Install the necessary Python packages using `pip`:

```bash
pip install numpy pandas scipy matplotlib ahrs
```

## 📂 Data Structure

The scripts assume your raw data files (CSV or TXT) are organized within a specified folder path. The example assumes the following structure:

Certainly, here is the folder structure section in raw Markdown code.

Markdown

## 📂 Data Structure

The scripts assume your raw data files (CSV or TXT) are organized within a specified folder path.


🚀 Execution GuideThe main analysis is executed by running ellipse_exceedance.py.1. Configure SettingsBefore running, you must configure the parameters inside the if __name__ == "__main__": block of ellipse_exceedance.py:Pythonif __name__ == "__main__":
    folder = "measurement/lab_day_22/wxdumps" # <-- Update this path
    
    # Filenames for baseline Quiet Standing (QS) trials
    baseline_trials = ["1_QS.csv", "2_QS.csv"]  
    
    # Filenames for Limits of Stability (LoS) trials
    los_trials = ["4_LoS.csv", "5_LoS.csv"]  
    
    # Filenames for the trials you want to evaluate against the benchmarks
    test_trials = ["3_QS.csv"]  
    
    # IMU-specific settings:
    sensor_id = 2             # The ID used in the column headers (e.g., Imu_2_ImuAcc:X(g))
    sensor_height = 1140      # Height of the sensor from the ground (in mm)
    
    results_df = analyze_trials(folder, baseline_trials, los_trials, test_trials, sensor_id=sensor_id, sensor_height=sensor_height)
    # ... print results ...
    
2. Run the AnalysisExecute the main script from your terminal:
   ```
   python ellipse_exceedance.py
   ```
🔬 Analysis Output
The script will perform the following steps and provide both console output and visual plots.

Console Output
The console output will summarize the areas of the computed benchmarks and present the final stability metric in a table (Pandas DataFrame).

| Metric | Description |
| :--- | :--- |
| **QSth ellipse area** | Area of the Quiet Standing benchmark ellipse (based on $k=2$ standard deviations). |
| **LoSth ellipse area** | Area of the Limits of Stability benchmark ellipse. |
| **Prop\_Exceeded\_QS\_%** | **Percentage of time** the test trial trajectory falls *outside* the Quiet Standing benchmark. |
| **Prop\_Exceeded\_LoS\_%** | **Percentage of time** the test trial trajectory falls *outside* the Limits of Stability benchmark. |

## Plots
For each trial in test_trials, plots showing angles, displacement over time, and the CoM trajectory with the QS and LoS benchmark ellipses overlaid will be generated.
