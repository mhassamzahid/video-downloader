import pandas as pd

CSV_FILE = "youtube_video_metadata.csv"

df = pd.read_csv(CSV_FILE, dtype=str).fillna("")

cols_to_reset = [
    "ai_check_done",
    "ai_decision",
    "ai_confidence",
    "ai_reason",
    "classification_error",
]

for col in cols_to_reset:
    if col in df.columns:
        df[col] = ""

df.to_csv(CSV_FILE, index=False)

print("AI classification columns reset.")