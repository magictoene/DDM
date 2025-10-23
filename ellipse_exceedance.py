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

# --- new helper functions ---
def compute_bsi(prop_exceeded_qs):
    """BSI_QSth = 1 - (Prop_Exceeded_QSth / 100)"""
    return 1.0 - (prop_exceeded_qs / 100.0)


def compute_los_margin_index(prop_exceeded_los):
    """LoS_Margin_Index = 1 - (Prop_Exceeded_LoSth / 100)"""
    return 1.0 - (prop_exceeded_los / 100.0)


def compute_composite_balance_score(bsi_qs, los_margin):
    """Balance_Score = 100 * (0.7 * BSI_QSth + 0.3 * LoS_Margin_Index)"""
    score = 100.0 * (0.7 * bsi_qs + 0.3 * los_margin)
    return max(0.0, score)


def classify_score(score):
    """Return categorical interpretation for a numeric score."""
    if score >= 90.0:
        return "Excellent"
    if score >= 75.0:
        return "Good"
    if score >= 60.0:
        return "Moderate"
    return "Poor"

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
                                use_mag=use_mag, plot=False)
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

        # compute indices and composite score
        bsi_qs = compute_bsi(prop_exceeded_qs)
        los_margin = compute_los_margin_index(prop_exceeded_los)
        balance_score = compute_composite_balance_score(bsi_qs, los_margin)
        category = classify_score(balance_score)

        all_results.append({
            "Trial": test,
            "Prop_Exceeded_QS_%": prop_exceeded_qs,
            "Prop_Exceeded_LoS_%": prop_exceeded_los,
            "BSI_QSth": bsi_qs,
            "LoS_Margin_Index": los_margin,
            "Balance_Score": balance_score,
            "Score_Category": category
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

    # summary bar chart for all test trials (optional)
    if plot_examples and len(all_results) > 0:
        df_results = pd.DataFrame(all_results)
        plt.figure(figsize=(max(6, 0.6 * len(df_results)), 4))
        # colors = df_results["Score_Category"].map({
        #     "Excellent": "#2ca02c",
        #     "Good": "#1f77b4",
        #     "Moderate": "#ff7f0e",
        #     "Poor": "#d62728"
        # }).fillna("#7f7f7f")
        plt.bar(df_results["Trial"], df_results["Balance_Score"], color='steelblue') #, color=colors)
        plt.axhline(90, color="green", linestyle="--", label="Excellent (≥90)")
        plt.axhline(75, color="yellow", linestyle="--", label="Good (≥75)")
        plt.axhline(60, color="orange", linestyle="--", label="Moderate (≥60)")
        plt.axhline(59, color="red", linestyle=":", label="Poor (<60)")
        for idx, row in df_results.iterrows():
            plt.text(idx, row["Balance_Score"] + 1.0, f"{row['Balance_Score']:.1f}", ha='center', va='bottom',
                     fontsize=8)
        plt.ylim(0, 110)
        plt.ylabel("Balance Score (0-100)")
        plt.title("Composite Balance Scores by Trial")
        plt.legend()
        # plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(f"balance_score.png")
        plt.show()


    return pd.DataFrame(all_results)


# =====================================
# 3. EXAMPLE __MAIN__ USAGE
# =====================================
if __name__ == "__main__":
    folder = "measurement/lab_day_22/wxdumps"

    baseline_trials = ["1_QS.csv"]  # quiet standing filenames
    los_trials = ["4_LoS.csv"]  # limits of stability filenames
    test_trials = ["3_QS.csv", "6_LoS.csv"]  # trial to evaluate

    results_df = analyze_trials(folder, baseline_trials, los_trials, test_trials, sensor_id=2)
    print("\n=== Threshold Exceedance Results ===")
    print(results_df)
