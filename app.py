import sys
from pathlib import Path
import pandas as pd
import streamlit as st
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from validation import REQUIRED_COLUMNS, NUMERIC_COLUMNS, validate_csv_columns, validate_records
st.set_page_config(page_title="Machine Monitoring",layout="wide")
st.title("Machine Monitoring")
st.caption("Milestone 2 — Data Input, Validation and Machine Monitoring")
mode=st.sidebar.radio("Input method",["CSV Upload","Manual Machine Input"])
raw=pd.DataFrame()
if mode=="CSV Upload":
    f=st.file_uploader("Upload machine data (CSV)",type=["csv"])
    if f:
        try:
            raw=pd.read_csv(f)
            st.success(f"Loaded {len(raw)} record(s), {len(raw.columns)} column(s).")
            missing=validate_csv_columns(raw)
            if missing: st.error("Required columns missing: "+", ".join(missing))
        except (pd.errors.ParserError,UnicodeDecodeError,ValueError) as e: st.error(f"Invalid CSV format: {e}")
        except Exception as e: st.error(f"Could not process file: {e}")
else:
    with st.form("manual"):
        typ=st.selectbox("Product Type",["L","M","H"])
        a,b,c=st.columns(3)
        air=a.number_input("Air Temperature [K]",150.0,500.0,300.0)
        speed=a.number_input("Rotational Speed [rpm]",0,100000,1500)
        proc=b.number_input("Process Temperature [K]",150.0,600.0,310.0)
        torque=b.number_input("Torque [Nm]",0.0,10000.0,40.0)
        wear=c.number_input("Tool Wear [min]",0,100000,0)
        submitted=st.form_submit_button("Validate Machine Input")
    if submitted:
        raw=pd.DataFrame([[typ,air,proc,speed,torque,wear]],columns=REQUIRED_COLUMNS)
if not raw.empty:
    st.subheader("Uploaded / Entered Records"); st.dataframe(raw,use_container_width=True,hide_index=True)
    missing=validate_csv_columns(raw)
    if missing: st.error("Cannot validate: missing columns — "+", ".join(missing))
    else:
        valid,errors=validate_records(raw)
        for e in errors: st.warning(e)
        if len(valid)==len(raw): st.success(f"Validation passed: {len(valid)} record(s).")
        else: st.error(f"{len(raw)-len(valid)} invalid; {len(valid)} valid record(s) retained.")
        st.subheader("Validated Machine Parameters"); st.dataframe(valid,use_container_width=True,hide_index=True)
        st.subheader("Prediction Input DataFrame"); st.dataframe(valid[REQUIRED_COLUMNS],use_container_width=True,hide_index=True)
        st.download_button("Download validated CSV",valid.to_csv(index=False).encode(), "validated_machine_inputs.csv","text/csv",disabled=valid.empty)
