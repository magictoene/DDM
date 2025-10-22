import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import process_trial, load_data


# assume process_trial() is imported from your existing script
# returns: { "file", "fs", "roll", "pitch", "x_disp", "y_disp", "x_range", "y_range" }

# =====================================
# 1. THRESHOLD / ELLIPSE FUNCTIONS
# =====================================
def compute_threshold_ellipse(AP, ML, k=2):
    """Compute benchmark ellipse from AP/ML arrays."""
    mu_AP, mu_ML = np.mean(AP), np.mean(ML)
    sigma_AP, sigma_ML = np.std(AP), np.std(ML)
    a = k * sigma_ML  # ML radius
    b = k * sigma_AP  # AP radius
    area = np.pi * a * b
    return {"center": (mu_ML, mu_AP), "radii": (a, b), "area": area}


def normalize_to_benchmark(AP, ML, benchmark):
    """Shift displacement so ellipse center is at origin for comparison."""
    cx, cy = benchmark["center"]
    AP_norm = AP - cy
    ML_norm = ML - cx
    return AP_norm, ML_norm


def compute_exceedance(AP, ML, benchmark):
    """Compute % of time trajectory is outside the benchmark ellipse."""
    cx, cy = benchmark["center"]
    a, b = benchmark["radii"]
    norm_sq = ((ML - cx) ** 2 / a ** 2) + ((AP - cy) ** 2 / b ** 2)
    exceeded = norm_sq > 1.0
    prop_exceeded = 100.0 * np.mean(exceeded)
    return exceeded, prop_exceeded


# =====================================
# 2. MAIN WRAPPER FOR MULTI-TRIAL ANALYSIS
# =====================================
def analyze_trials(folder_path, baseline_trials, los_trials, test_trials, sensor_id, sensor_height=1140, use_mag=False, k=2, plot_examples=True):
    all_results = []

    # --- Step 0: Load all files once ---
    all_data = load_data(folder_path)

    # --- Step 1: Process trials ---
    trial_data = {}

    for trial_list in [baseline_trials, los_trials, test_trials]:
        for trial_idx, trial_file in enumerate(trial_list):
            df = all_data[trial_file]
            res = process_trial(df, file_name=trial_file,
                                sensor_id=sensor_id, sensor_height=sensor_height,
                                use_mag=use_mag, plot=True)
            trial_data[trial_file] = res

    # --- Step 2: Compute benchmark ellipses ---
    # Quiet standing (QS)
    AP_qs = np.concatenate([trial_data[t]["y_disp"] for t in baseline_trials])
    ML_qs = np.concatenate([trial_data[t]["x_disp"] for t in baseline_trials])
    qs_ellipse = compute_threshold_ellipse(AP_qs, ML_qs, k=k)

    # Limits of stability (LoS)
    AP_los = np.concatenate([trial_data[t]["y_disp"] for t in los_trials])
    ML_los = np.concatenate([trial_data[t]["x_disp"] for t in los_trials])
    los_ellipse = compute_threshold_ellipse(AP_los, ML_los, k=k)

    print(f"QSth ellipse area = {qs_ellipse['area']:.3e} mm²")
    print(f"LoSth ellipse area = {los_ellipse['area']:.3e} mm²")

    # --- Step 3: Compare test trials to benchmarks ---
    for test in test_trials:
        AP_test = trial_data[test]["y_disp"]
        ML_test = trial_data[test]["x_disp"]

        # Normalize relative to ellipse center (optional)
        AP_qs_norm, ML_qs_norm = normalize_to_benchmark(AP_test, ML_test, qs_ellipse)
        AP_los_norm, ML_los_norm = normalize_to_benchmark(AP_test, ML_test, los_ellipse)

        # Compute exceedance
        _, prop_exceeded_qs = compute_exceedance(AP_qs_norm, ML_qs_norm,
                                                 {"center": (0, 0), "radii": qs_ellipse["radii"]})
        _, prop_exceeded_los = compute_exceedance(AP_los_norm, ML_los_norm,
                                                  {"center": (0, 0), "radii": los_ellipse["radii"]})

        all_results.append({
            "Trial": test,
            "Prop_Exceeded_QS_%": prop_exceeded_qs,
            "Prop_Exceeded_LoS_%": prop_exceeded_los
        })

        # Optional plot
        if plot_examples:
            plt.figure(figsize=(6, 6))
            plt.plot(ML_test, AP_test, 'k.', alpha=0.25, label="Trajectory")
            theta = np.linspace(0, 2 * np.pi, 300)
            # QS ellipse at origin
            plt.plot(qs_ellipse["radii"][0] * np.cos(theta), qs_ellipse["radii"][1] * np.sin(theta),
                     'b--', label=f"QSth ellipse")
            # LoS ellipse at origin
            plt.plot(los_ellipse["radii"][0] * np.cos(theta), los_ellipse["radii"][1] * np.sin(theta),
                     'r-', label=f"LoSth ellipse")
            plt.xlabel("ML (mm)")
            plt.ylabel("AP (mm)")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.title(f"{test} | QS {prop_exceeded_qs:.1f}% | LoS {prop_exceeded_los:.1f}%")
            plt.show()

    return pd.DataFrame(all_results)


# =====================================
# 3. EXAMPLE __MAIN__ USAGE
# =====================================
if __name__ == "__main__":
    folder = "measurement/lab_day_22/wxdumps"

    baseline_trials = ["1_QS.csv", "2_QS.csv"]  # quiet standing filenames
    los_trials = ["4_LoS.csv", "5_LoS.csv"]  # limits of stability filenames
    test_trials = ["3_QS.csv"]  # trial to evaluate

    results_df = analyze_trials(folder, baseline_trials, los_trials, test_trials, sensor_id=2)
    print("\n=== Threshold Exceedance Results ===")
    print(results_df)
