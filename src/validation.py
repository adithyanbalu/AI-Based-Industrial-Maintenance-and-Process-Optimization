import pandas as pd
REQUIRED_COLUMNS = ["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
ALLOWED_TYPES = {"L","M","H"}
def validate_csv_columns(df):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]
def validate_records(df):
    missing_cols=validate_csv_columns(df)
    if missing_cols: return pd.DataFrame(columns=REQUIRED_COLUMNS), ["Missing required columns: "+", ".join(missing_cols)]
    work=df[REQUIRED_COLUMNS].copy()
    work["Type"]=work["Type"].astype("string").str.strip().str.upper()
    for c in NUMERIC_COLUMNS: work[c]=pd.to_numeric(work[c],errors="coerce")
    errors=[]
    missing=work[REQUIRED_COLUMNS].isna().any(axis=1)
    if missing.any(): errors.append(f"{int(missing.sum())} row(s) have missing/non-numeric required values.")
    typ=work["Type"].isin(ALLOWED_TYPES)
    if (~typ).any(): errors.append(f"{int((~typ).sum())} row(s) have invalid Product Type; allowed: L, M, H.")
    ranges={"Air temperature [K]":(150,500),"Process temperature [K]":(150,600),"Rotational speed [rpm]":(0,100000),"Torque [Nm]":(0,10000),"Tool wear [min]":(0,100000)}
    ok=pd.Series(True,index=work.index)
    for c,(lo,hi) in ranges.items(): ok &= work[c].between(lo,hi)
    if (~ok & ~missing).any(): errors.append(f"{int((~ok & ~missing).sum())} row(s) outside configured sanity ranges.")
    valid=work.loc[(~missing)&typ&ok,REQUIRED_COLUMNS].reset_index(drop=True)
    return valid,errors
