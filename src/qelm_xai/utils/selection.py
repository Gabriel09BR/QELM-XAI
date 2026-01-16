
# --------- Multi-objective settings (COMPOSITE ONLY) ---------
W_F1 = 0.7  # weight for F1 in the composite score (0..1)
df = dataset

def best_composite_idx(df, f1_col="F1", tt_col="TT_sec", w=0.7):
    """
    Selection by normalized composite score:
        score = w * F1_norm + (1 - w) * (1 - TT_norm)
    (higher is better)

    Returns:
        best_index : index of the selected configuration
        scores     : composite score for all configurations
    """
    f1 = df[f1_col].to_numpy(dtype=float)
    tt = df[tt_col].to_numpy(dtype=float)

    # Min-max normalization
    f1n = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-12)
    ttn = (tt - tt.min()) / (tt.max() - tt.min() + 1e-12)

    score = w * f1n + (1 - w) * (1 - ttn)
    best_pos = int(np.argmax(score))

    return df.index[best_pos], score


# =========================
# PyCaret setup
# =========================
setup(
    data=df,
    target="class",
    session_id=42,
    fold=20,
    verbose=False
)

# =========================
# Search space
# =========================
kernels = ["relu", "sigmoid", "tanh", "linear", "sine", "hardlim", "tribas", "poly", "radbas"]

# =========================
# Tuning loop per kernel
# =========================
results = []  # will store: {kernel, F1, TT_sec, params, model}

for k in kernels:
    print(f"\n====================== Kernel: {k} ======================")

    # Base model (custom estimator)
    base_model = ELMClassifier(activation=k)

    # Hyperparameter grid
    custom_grid = {
        "n_hidden":  np.linspace(50, 1500, 40, dtype=int).tolist(),
        "C":         np.linspace(0.01, 100.0, 100),
        "generator": ["new"],
    }

    # OPTIONAL: if you want to tune activation_params for poly / radbas,
    # uncomment and adjust the following block.
    # WARNING: This may lead to a combinatorial explosion.
    """
    from itertools import product

    if k == "poly":
        deg_list   = [2, 3]
        gamma_list = np.linspace(0.01, 10.0, 400)
        coef0_list = np.linspace(0.01, 10.0, 400)

        custom_grid["activation_params"] = [
            {"degree": int(d), "gamma": float(g), "coef0": float(c)}
            for d, g, c in product(deg_list, gamma_list, coef0_list)
        ]

    elif k == "radbas":
        gamma_list = np.linspace(0.01, 10.0, 400)
        sigma_list = np.linspace(0.01, 10.0, 400)

        custom_grid["activation_params"] = [
            {"gamma": float(g), "sigma": float(s)}
            for g, s in product(gamma_list, sigma_list)
        ]
    """

    # ---------- Tuning (random search) ----------
    tuned, tuner = tune_model(
        base_model,
        optimize="F1",
        custom_grid=custom_grid,
        search_library="scikit-learn",
        search_algorithm="random",
        n_iter=500,
        choose_better=True,
        fold=25,
        verbose=False,
        return_tuner=True
    )

    # PyCaret summary table
    results_df = pull()
    print("\nPyCaret table (tuning summary):")
    print(results_df.head())

    print("\nBest hyperparameters (PyCaret - optimizing F1 only):")
    print(tuned.get_params())

    # =============================================
    # Composite multi-objective analysis (F1 + time)
    # =============================================
    try:
        cv = pd.DataFrame(tuner.cv_results_)

        # Identify the n_hidden column (param_*n_hidden*)
        nh_cols = [c for c in cv.columns if c.startswith("param_") and "n_hidden" in c]
        if not nh_cols:
            raise RuntimeError(
                "Could not find param_* column for n_hidden in cv_results_."
            )
        col_nh = nh_cols[0]

        # F1 metric used in tuning
        if "mean_test_score" not in cv.columns:
            raise RuntimeError("cv_results_ does not contain mean_test_score.")

        # Training time: prefer mean_fit_time, fallback if needed
        if "mean_fit_time" in cv.columns:
            tt_series = cv["mean_fit_time"].astype(float)
        else:
            split_fit_cols = [c for c in cv.columns if c.startswith("split") and c.endswith("_fit_time")]
            if split_fit_cols:
                tt_series = cv[split_fit_cols].astype(float).mean(axis=1)
            elif "mean_score_time" in cv.columns:
                tt_series = cv["mean_score_time"].astype(float)
            else:
                raise RuntimeError(
                    "No training-time columns found in cv_results_."
                )

        # Build dataframe for composite selection
        df_mo = pd.DataFrame({
            "F1": cv["mean_test_score"].astype(float),
            "TT_sec": tt_series,
            "n_hidden": pd.to_numeric(cv[col_nh], errors="coerce"),
        }).dropna()

        if df_mo.empty:
            raise RuntimeError("df_mo is empty after dropna().")

        # ------- Composite selection -------
        best_idx, scores = best_composite_idx(df_mo, "F1", "TT_sec", w=W_F1)
        chosen_row = df_mo.loc[best_idx]

        print("\n[Multi-objective | Composite] Best selected point:")
        print(chosen_row.to_frame().T)

     
    except Exception as e:
        print(f"\n[WARNING] Composite multi-objective analysis failed for kernel {k}: {e}")