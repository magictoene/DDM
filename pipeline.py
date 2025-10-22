import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from ahrs.filters import EKF
from ahrs.common.orientation import q2euler
import os, glob
from io import StringIO
import matplotlib.pyplot as plt

# ==============================================================
# 1. LOAD DATA
# ==============================================================
def load_data(folder_path):
    """
    Loads all .txt and .csv files from a given folder into a dictionary.
    Automatically detects encoding and separator.
    """
    def load_txt(file):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="latin1")
            except pd.errors.ParserError:
                df = pd.read_csv(file, sep=";", encoding="latin1")
        else:
            with open(file, "r", encoding="utf-8", errors="replace") as f:
                lines = [line for line in f.readlines() if line.strip()]
            header_line = lines[1].strip().split("\t")
            data_lines = lines[2:]
            df = pd.read_csv(StringIO("".join(data_lines)), sep="\t", names=header_line)

        df.columns = (df.columns.str.replace("�", "u", regex=False)
                                  .str.replace("µ", "u", regex=False)
                                  .str.strip())
        return df

    txt_files = glob.glob(os.path.join(folder_path, "*.txt")) + \
                glob.glob(os.path.join(folder_path, "*.csv"))

    data = {os.path.basename(file): load_txt(file) for file in txt_files}
    print(f"✅ Loaded {len(data)} files from {folder_path}")
    return data


# ==============================================================
# 2. BASIC PROCESSING
# ==============================================================
def get_sampling_rate(df):
    """Estimate sampling rate (Hz) from the time column."""
    time = df["Time(s)"].to_numpy()
    dt = np.diff(time)
    return 1 / np.median(dt)


def trim_initial_data(df, fs, duration=0.1):
    """Trim the first <duration> seconds of data."""
    cut_samples = int(duration * fs)
    df = df.iloc[cut_samples:].reset_index(drop=True)
    df["Time(s)"] = df["Time(s)"] - df["Time(s)"].iloc[0]
    return df


def scale_units(df, acc_cols, gyr_cols, mag_cols):
    """Convert sensor units to SI units."""
    g = 9.80665
    df[acc_cols] *= g
    df[gyr_cols] *= np.pi / 180
    df[mag_cols] *= 1e3  # µT → nT
    return df


def downsample(df, fs, target_fs=100):
    """Downsample data by integer factor."""
    factor = int(fs / target_fs)
    df = df.iloc[::factor, :].reset_index(drop=True)
    return df, target_fs


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, data)


def filter_data(df, acc_cols, gyr_cols, fs, cutoff=15):
    """Filter accelerometer and gyroscope signals."""
    df_filt = df.copy()
    for col in acc_cols + gyr_cols:
        df_filt[col] = butter_lowpass_filter(df[col], cutoff, fs)
    return df_filt


# ==============================================================
# 3. EXTENDED KALMAN FILTER
# ==============================================================
def run_ekf(df, acc_cols, gyr_cols, mag_cols, fs, use_mag=False):
    """Run EKF and return Euler angles (radians)."""
    acc = df[acc_cols].to_numpy()
    gyr = df[gyr_cols].to_numpy()

    if use_mag:
        mag = df[mag_cols].to_numpy()
        ekf = EKF(gyr=gyr, acc=acc, mag=mag, frequency=fs)
    else:
        ekf = EKF(gyr=gyr, acc=acc, frequency=fs)

    euler_angles = np.array([q2euler(q_) for q_ in ekf.Q])
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    return roll, pitch, yaw


def normalize_angles(roll, pitch):
    """Unwrap and set initial angles to zero."""
    roll = np.unwrap(roll) - roll[0]
    pitch = np.unwrap(pitch) - pitch[0]
    return roll, pitch


# ==============================================================
# 4. POST-PROCESSING: DISPLACEMENT AND METRICS
# ==============================================================
def compute_displacement(roll, pitch, sensor_height_mm):
    """Compute displacement (mm) from roll and pitch."""
    x_disp = -sensor_height_mm * np.sin(roll)
    y_disp = -sensor_height_mm * np.sin(pitch)
    return x_disp, y_disp


def get_min_max_displacement(x_disp, y_disp):
    """Return min/max values of displacement."""
    return (np.min(x_disp), np.max(x_disp)), (np.min(y_disp), np.max(y_disp))


# ==============================================================
# 5. PLOTTING FUNCTIONS
# ==============================================================
def plot_displacement_over_time(time, x_disp, y_disp):
    plt.figure(figsize=(12, 6))
    plt.plot(time, x_disp, label="X displacement (mm)")
    plt.plot(time, y_disp, label="Y displacement (mm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (mm)")
    plt.title("Center of Mass Displacement Over Time")
    plt.legend()
    plt.grid()
    plt.show()


def plot_angles_over_time(time, roll, pitch):
    plt.figure(figsize=(12, 6))
    plt.plot(time, np.degrees(roll), label="Roll (°)")
    plt.plot(time, np.degrees(pitch), label="Pitch (°)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (°)")
    plt.title("Roll and Pitch Angles Over Time")
    plt.legend()
    plt.grid()
    plt.show()


def plot_displacement_plane(x_disp, y_disp):
    plt.figure(figsize=(8, 8))
    plt.plot(x_disp, y_disp, label="Center of Mass Trajectory")
    plt.xlabel("X displacement (mm)")
    plt.ylabel("Y displacement (mm)")
    plt.title("Center of Mass Displacement Plane (Top View)")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.grid()
    plt.show()


# ==============================================================
# 6. MAIN WORKFLOW
# ==============================================================
def process_trial(folder_path, trial_index=0, sensor_id=6, sensor_height=1140, use_mag=False):
    """Full pipeline for one trial."""

    # --- Load data ---
    data = load_data(folder_path)
    file_name = list(data.keys())[trial_index]
    df = data[file_name]
    print(f"📂 Using file: {file_name}")

    # --- Define columns ---
    acc_cols = [f"Imu_{sensor_id}_ImuAcc:X(g)", f"Imu_{sensor_id}_ImuAcc:Y(g)", f"Imu_{sensor_id}_ImuAcc:Z(g)"]
    gyr_cols = [f"Imu_{sensor_id}_ImuGyro:X(D/s)", f"Imu_{sensor_id}_ImuGyro:Y(D/s)", f"Imu_{sensor_id}_ImuGyro:Z(D/s)"]
    mag_cols = [f"Imu_{sensor_id}_ImuMag:X(uT)", f"Imu_{sensor_id}_ImuMag:Y(uT)", f"Imu_{sensor_id}_ImuMag:Z(uT)"]

    # --- Processing steps ---
    fs = get_sampling_rate(df)
    df = trim_initial_data(df, fs)
    df = scale_units(df, acc_cols, gyr_cols, mag_cols)
    df, fs = downsample(df, fs, target_fs=100)
    df_filt = filter_data(df, acc_cols, gyr_cols, fs, cutoff=15)

    # --- EKF ---
    roll, pitch, yaw = run_ekf(df_filt, acc_cols, gyr_cols, mag_cols, fs, use_mag=use_mag)
    roll, pitch = normalize_angles(roll, pitch)

    # --- Displacement ---
    x_disp, y_disp = compute_displacement(roll, pitch, sensor_height)
    (x_min, x_max), (y_min, y_max) = get_min_max_displacement(x_disp, y_disp)

    # --- Results ---
    print(f"X displacement: {x_min:.2f} mm to {x_max:.2f} mm")
    print(f"Y displacement: {y_min:.2f} mm to {y_max:.2f} mm")

    # --- Plots ---
    time = df_filt["Time(s)"].to_numpy()
    plot_angles_over_time(time, roll, pitch)
    plot_displacement_over_time(time, x_disp, y_disp)
    plot_displacement_plane(x_disp, y_disp)

    return {
        "file": file_name,
        "fs": fs,
        "roll": roll,
        "pitch": pitch,
        "x_disp": x_disp,
        "y_disp": y_disp,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max)
    }


# ==============================================================
# 7. RUN PIPELINE
# ==============================================================
if __name__ == "__main__":
    folder = "measurement/20_10"
    results = process_trial(folder_path=folder, trial_index=0, sensor_id=6, sensor_height=1140, use_mag=False)