"""Anchored CASAS activity-error demonstration; scripted research data only."""
from pathlib import Path
import json
import warnings
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).parent
st.set_page_config(page_title="Anchored | Activity ML", page_icon="🌳", layout="wide")

@st.cache_resource(show_spinner="Loading trained CASAS model …")
def load_artifact():
    return joblib.load(ROOT / "anchored_casas_mini_model.joblib")

@st.cache_data
def load_results():
    return json.loads((ROOT / "anchored_ml_results.json").read_text(encoding="utf-8"))

st.title("🌳 Anchored | Activity Error ML")
st.caption("Preliminary research demonstration • real CASAS extracted features • no patient data")
st.info("This model classifies error types in **scripted research tasks**. It does not detect Alzheimer's disease, diagnose impairment, or monitor patients.")

results=load_results()
a,b,c=st.columns(3)
a.metric("Random Forest balanced accuracy", f"{100*results['model_balanced_accuracy']:.2f}%")
b.metric("Baseline balanced accuracy", f"{100*results['baseline_balanced_accuracy']:.2f}%")
c.metric("Research feature rows", f"{results['source_rows']}")
st.caption("The reported evaluation used a row-level split, **not** a participant-independent split. Test-set leakage and performance inflation remain possible.")
st.subheader("Measured baseline comparison")
st.bar_chart(pd.DataFrame({"Balanced accuracy (%)": [100*results['baseline_balanced_accuracy'],100*results['model_balanced_accuracy']]},index=["Majority-class baseline","Random Forest"]))

try:
    artifact=load_artifact()
    model=artifact["model"]
    features=list(artifact["feature_names"])
except Exception as exc:
    st.error("The saved model could not be loaded. Confirm requirements.txt uses scikit-learn==1.6.1 and Python 3.11.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

st.subheader("Run a genuine CASAS example")
st.markdown("Upload `anchored_casas_demo_rows.csv`, exported from the **same Colab notebook after training**. The file contains actual held-out research feature rows and their recorded labels. No artificial input values are generated.")

source = ROOT / "anchored_casas_demo_rows.csv"
if source.is_file():
    st.success("Genuine CASAS demonstration examples are included in this deployment.")
    data=pd.read_csv(source)
else:
    st.warning("No CASAS demo rows have been added yet. Upload the CSV from Colab using the file picker below, or commit it to GitHub for a ready-to-record demo.")
    upload=st.file_uploader("Upload real CASAS demo rows (CSV)",type=["csv"])
    data=pd.read_csv(upload) if upload else None

if data is not None:
    if not set(features).issubset(set(data.columns)):
        missing=sorted(set(features)-set(data.columns))
        st.error(f"CSV is missing {len(missing)} required model feature columns. These include: {missing[:12]}")
        st.stop()
    if len(data)==0:
        st.error("The uploaded CSV has no rows.")
        st.stop()
    idx=st.selectbox("Choose a recorded research observation",range(len(data)),format_func=lambda i: f"CASAS record {i+1} / {len(data)}")
    row=data.iloc[[idx]][features]
    left,right=st.columns(2)
    with left:
        st.markdown("**Recorded experimental observation**")
        if "activity" in data.columns:
            st.write("Activity:",str(data.iloc[idx]["activity"]))
        if "actual_class" in data.columns:
            st.write("Research annotation:",str(data.iloc[idx]["actual_class"]))
        st.caption("Data originate from experimentally scripted CASAS tasks; observations are not clinical patient records.")
    with right:
        st.markdown("**Model inference**")
        if st.button("Predict activity-error category",type="primary",use_container_width=True):
            try:
                prediction=model.predict(row)[0]
                st.success(f"Predicted category: {prediction}")
                if "actual_class" in data.columns:
                    actual=str(data.iloc[idx]["actual_class"])
                    st.write("Matches research annotation:","Yes" if str(prediction)==actual else "No")
                if hasattr(model,"predict_proba"):
                    prob=model.predict_proba(row)[0]
                    labels=model.named_steps["clf"].classes_
                    st.dataframe(pd.DataFrame({"Error type":labels,"Model score":prob}).sort_values("Model score",ascending=False).reset_index(drop=True),hide_index=True,use_container_width=True)
                    st.caption("Model scores are not clinically calibrated probabilities.")
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
    with st.expander("Inspect actual research feature values"):
        st.dataframe(row.T.rename(columns={row.index[0]:"Value"}),use_container_width=True)
        st.caption(f"The saved model expects {len(features)} features.")
else:
    st.caption("For the first demo, complete the CSV export step shown in README.md.")

st.divider()
st.subheader("Evidence and responsible use")
st.markdown("""
- **Data:** CASAS Cognitive Activity Errors dataset, Zenodo record [15716889](https://zenodo.org/records/15716889); licensed CC BY 4.0. Errors were deliberately introduced in research activities.
- **Prediction task:** four annotated activity-error categories: inefficient action, omission, irrelevant action, and substitution.
- **Validation limitation:** the current split is row-level; participant-separated generalization is not established. The minority classes are very small.
- **No medical claims:** this research demo does not predict, diagnose or treat Alzheimer's disease.
- **Privacy:** only non-identifiable, publicly available research features should be uploaded. Do **not** upload patient information.
""")
st.caption("Anchored — early technical research prototype. A separate LifeGraph demo shows fictional personal-memory retrieval.")
