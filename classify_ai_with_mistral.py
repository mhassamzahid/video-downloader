import os
import json
import time
import shutil
import pandas as pd
from pathlib import Path
from mistralai import Mistral
from tqdm import tqdm


CSV_FILE = "youtube_video_metadata.csv"

REVIEW_FOLDER = Path("mp3_needs_review_or_delete")
MODEL = "mistral-small-latest"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("Missing MISTRAL_API_KEY environment variable")

client = Mistral(api_key=MISTRAL_API_KEY)


def ask_mistral(title, description):
    prompt = f"""
You classify YouTube videos.

Determine if this video is likely AI-generated, AI voice, AI avatar, synthetic story narration, AI news narration, faceless AI content, or low-quality generated content.

Only use the title and description.

Return ONLY valid JSON:
{{
  "decision": "keep" or "delete",
  "new_label": "real" or "Delete",
  "confidence": 0.0 to 1.0,
  "reason": "short reason"
}}

Rules:
- Use "Delete" if the video is likely AI-generated or synthetic.
- Use "real" if it looks like real human footage, interviews, podcasts, vlogs, real events, real people speaking, or normal YouTube content.
- If unsure, use "keep".

TITLE:
{title}

DESCRIPTION:
{description}
""".strip()

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "decision": "keep",
            "new_label": "real",
            "confidence": 0,
            "reason": f"Invalid JSON from Mistral: {text[:200]}"
        }


def move_mp3(row):
    mp3_path = str(row.get("mp3_path", "")).strip()

    if not mp3_path:
        return ""

    src = Path(mp3_path)

    if not src.exists():
        return ""

    REVIEW_FOLDER.mkdir(exist_ok=True)

    dst = REVIEW_FOLDER / src.name

    if dst.exists():
        dst = REVIEW_FOLDER / f"{src.stem}_{int(time.time())}{src.suffix}"

    shutil.move(str(src), str(dst))

    return str(dst)


def main():
    df = pd.read_csv(CSV_FILE, dtype=str).fillna("")

    extra_cols = [
        "ai_check_done",
        "ai_decision",
        "ai_confidence",
        "ai_reason",
        "old_label",
        "moved_mp3_path"
    ]

    for col in extra_cols:
        if col not in df.columns:
            df[col] = ""

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if str(row.get("ai_check_done", "")).lower() == "true":
            continue

        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()

        if not title and not description:
            df.at[i, "ai_check_done"] = "true"
            df.at[i, "ai_decision"] = "keep"
            df.at[i, "ai_confidence"] = "0"
            df.at[i, "ai_reason"] = "Missing title and description"
            df.to_csv(CSV_FILE, index=False)
            continue

        try:
            result = ask_mistral(title, description)

            decision = result.get("decision", "keep")
            new_label = result.get("new_label", "real")
            confidence = result.get("confidence", 0)
            reason = result.get("reason", "")

            df.at[i, "ai_check_done"] = "true"
            df.at[i, "ai_decision"] = decision
            df.at[i, "ai_confidence"] = str(confidence)
            df.at[i, "ai_reason"] = reason

            if decision == "delete" or new_label == "Delete":
                df.at[i, "old_label"] = row.get("label", "")
                df.at[i, "label"] = "Delete"

                moved_path = move_mp3(row)

                if moved_path:
                    df.at[i, "moved_mp3_path"] = moved_path
                    df.at[i, "mp3_path"] = moved_path

            df.to_csv(CSV_FILE, index=False)

            time.sleep(0.5)

        except KeyboardInterrupt:
            df.to_csv(CSV_FILE, index=False)
            print("Stopped. Progress saved.")
            return

        except Exception as e:
            df.at[i, "ai_check_done"] = ""
            df.at[i, "ai_decision"] = ""
            df.at[i, "ai_reason"] = str(e)[:500]
            df.to_csv(CSV_FILE, index=False)

            time.sleep(2)

    print("Done.")
    print(f"Updated CSV: {CSV_FILE}")
    print(f"Moved Delete MP3s to: {REVIEW_FOLDER}")


if __name__ == "__main__":
    main()