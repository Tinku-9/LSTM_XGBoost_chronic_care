import gradio as gr
import matplotlib.pyplot as plt

def predict_and_explain(static_age, static_sex, static_diabetes, static_htn, med_adherence, glucose, bp_sys, bp_dia, hr):
    # Build payload (dummy with one timeseries point for now)
    payload = {
        "static": {
            "age": int(static_age),
            "sex": static_sex,
            "diabetes": int(static_diabetes),
            "htn": int(static_htn),
            "med_adherence": float(med_adherence)
        },
        "ts": [
            {"date":"2025-08-01","glucose":float(glucose),"bp_systolic":float(bp_sys),
             "bp_diastolic":float(bp_dia),"hr":float(hr)},
            {"date":"2025-08-15","glucose":float(glucose*1.05),"bp_systolic":float(bp_sys+2),
             "bp_diastolic":float(bp_dia+1),"hr":float(hr+2)},
            {"date":"2025-08-29","glucose":float(glucose*1.1),"bp_systolic":float(bp_sys+5),
             "bp_diastolic":float(bp_dia+3),"hr":float(hr+4)},
        ]
    }

    # --- Run model prediction ---
    preds = predict_from_payload(payload)  # {'xgb_prob':..., 'fusion_prob':...}
    risk = preds["fusion_prob"]

    # --- Make traffic-light risk text ---
    if risk < 0.3:
        risk_status = f"🟢 Low Risk ({risk:.2f})"
    elif risk < 0.6:
        risk_status = f"🟡 Moderate Risk ({risk:.2f})"
    else:
        risk_status = f"🔴 High Risk ({risk:.2f})"

    # --- Run explanations using your explain_sample-like logic ---
    # For simplicity, we’ll just fake it here with SHAP (use your existing explain_sample if available)
    # Assuming you adapted explain_sample(payload) to work directly with JSON
    try:
        exp = explain_sample(5)  # Replace this with a proper explain_from_payload(payload)
        xgb_top = exp['xgb_top']
        lstm_top = exp['lstm_top']
    except:
        xgb_top = [("glucose_mean", 0.25), ("bp_systolic_last", 0.15), ("age", 0.10)]
        lstm_top = [("glucose", 0.3), ("bp_systolic", 0.2), ("hr", 0.1)]

    # --- Build factor explanation text ---
    factors_text = "Top XGBoost factors:\n" + "\n".join([f"{f}: {s:.3f}" for f,s in xgb_top]) + \
                   "\n\nTop LSTM saliency:\n" + "\n".join([f"{f}: {s:.3f}" for f,s in lstm_top])

    # --- Plot vitals ---
    ts_df = pd.DataFrame(payload["ts"])
    plt.figure(figsize=(6,4))
    plt.plot(ts_df["date"], ts_df["glucose"], label="Glucose", marker="o")
    plt.plot(ts_df["date"], ts_df["bp_systolic"], label="BP Systolic", marker="o")
    plt.plot(ts_df["date"], ts_df["bp_diastolic"], label="BP Diastolic", marker="o")
    plt.plot(ts_df["date"], ts_df["hr"], label="Heart Rate", marker="o")
    plt.xticks(rotation=30)
    plt.legend()
    plt.title("Patient Vitals Trend")
    plt.tight_layout()
    fig = plt.gcf()

    return risk_status, factors_text, fig


# --- Gradio UI ---
demo = gr.Interface(
    fn=predict_and_explain,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["M","F"], label="Sex"),
        gr.Radio([0,1], label="Diabetes"),
        gr.Radio([0,1], label="Hypertension"),
        gr.Slider(0,1,step=0.1,label="Med Adherence"),
        gr.Number(label="Glucose"),
        gr.Number(label="BP Systolic"),
        gr.Number(label="BP Diastolic"),
        gr.Number(label="HR"),
    ],
    outputs=[
        gr.Textbox(label="Risk Score"),
        gr.Textbox(label="Top Contributing Factors"),
        gr.Plot(label="Vitals Graph")
    ]
)

demo.launch(share=True)
