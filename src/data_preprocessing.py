from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Dataset.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
lines = [
    "DATA PREPROCESSING INSPECTION REPORT",
    "=" * 40,
    f"Source file: {DATA_PATH.relative_to(ROOT)}",
    f"Dataset shape (rows, columns): {df.shape}",
    "",
    "FIRST FIVE ROWS",
    "-" * 40,
    df.head().to_string(index=False),
    "",
    "COLUMN NAMES AND DATA TYPES",
    "-" * 40,
]
lines.extend(f"{col}: {dtype}" for col, dtype in df.dtypes.items())
lines += ["", "MISSING VALUES BY COLUMN", "-" * 40]
lines.extend(f"{col}: {count}" for col, count in df.isna().sum().items())
lines += [
    f"Total missing cells: {int(df.isna().sum().sum())}",
    "",
    "DUPLICATE ROWS",
    "-" * 40,
    f"Duplicate rows: {int(df.duplicated().sum())}",
]
report = "\n".join(lines) + "\n"
print(report)
(OUT / "preprocessing_report.txt").write_text(report, encoding="utf-8")

preview = df.head()
fig, ax = plt.subplots(figsize=(18, 3.5))
ax.axis("off")
ax.set_title(f"Dataset.csv — First Five Rows | Shape: {df.shape}", fontsize=14, pad=16)
table = ax.table(
    cellText=preview.astype(str).values.tolist(),
    colLabels=[str(c) for c in preview.columns],
    cellLoc="center", colLoc="center", loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.6)
fig.tight_layout()
fig.savefig(OUT / "first_five_rows.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved report: {OUT / 'preprocessing_report.txt'}")
print(f"Saved preview image: {OUT / 'first_five_rows.png'}")
