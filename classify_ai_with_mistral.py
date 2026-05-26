import os
import json
import time
import shutil
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


CSV_FILE = "youtube_video_metadata.csv"

MP3_BASE_DIR = Path("mp3_dataset")
DELETE_FOLDER = Path("mp3_delete_review")

MODEL = "mistral-small-latest"
SLEEP_BETWEEN_CALLS = 0.5

FORCE_RECHECK = True
MOVE_DELETE_MP3 = True
MIN_CONFIDENCE = 0.70


load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY missing in .env")


def call_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "response_format": {"type": "json_object"},
    }

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def clean_json(text):
    text = str(text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


def normalize_label(label):
    label = str(label or "").strip().lower()

    if label == "human":
        return "human"

    if label == "ai":
        return "ai"

    return "Delete"


def keyword_prefilter(title, description):
    text = f"{title} {description}".lower()

    delete_phrases = [
        "how to make ai voice",
        "how to create ai voice",
        "how to generate ai voice",
        "how to use ai voice",
        "how to clone voice",
        "ai voice tutorial",
        "text to speech tutorial",
        "tts tutorial",
        "elevenlabs tutorial",
        "voice cloning tutorial",
        "ai voice guide",
        "ai voice generator tutorial",
        "best ai voice generator",
        "ai tools",
        "ai tool",
        "tutorial",
        "step by step",
        "explained",
        "guide to",
        "review of",
        "comparison",
        "human vs ai",
        "ai vs human",
        "reaction",
        "reacts to",
        "compilation",
        "funny moments",
        "meme",
        "remix",
        "edit",
        "lyrics",
        "music video",
        "sound effect",
        "sfx",
        "no commentary",
        "gameplay",
        "movie clip",
        "trailer",
        "anime clip",
        "cartoon clip",
        "scene from",
        "behind the scenes",
    ]

    ai_phrases = [
        "ai voice",
        "ai generated voice",
        "ai-generated voice",
        "ai generated narration",
        "ai-generated narration",
        "ai narration",
        "ai narrator",
        "text to speech",
        "text-to-speech",
        "tts",
        "elevenlabs",
        "synthetic voice",
        "synthetic narration",
        "voice generated",
        "generated voice",
        "ai avatar",
        "ai podcast",
        "ai song",
        "ai cover",
        "made with ai",
        "created with ai",
        "artificial intelligence voice",
        "voice cloning",
        "cloned voice",
        "faceless ai",
        "reddit story ai",
        "ai reddit story",
        "ai story",
        "ai generated story",
    ]

    human_phrases = [
        "podcast",
        "interview",
        "lecture",
        "sermon",
        "debate",
        "speech",
        "press conference",
        "classroom",
        "meeting",
        "conversation",
        "live stream",
        "livestream",
        "vlog",
        "stand up",
        "standup",
        "news report",
        "real interview",
        "keynote",
        "talk show",
    ]

    for phrase in delete_phrases:
        if phrase in text:
            return {
                "label": "Delete",
                "confidence": 0.95,
                "reason": f"Delete phrase matched: {phrase}",
            }

    ai_hits = [phrase for phrase in ai_phrases if phrase in text]
    human_hits = [phrase for phrase in human_phrases if phrase in text]

    if ai_hits and human_hits:
        return {
            "label": "Delete",
            "confidence": 0.95,
            "reason": f"Mixed AI and human signals found. AI={ai_hits[:3]}, Human={human_hits[:3]}",
        }

    return None


def classify_video(title, description):
    prefilter = keyword_prefilter(title, description)
    if prefilter:
        return prefilter

    prompt = f"""
You are classifying YouTube videos for a clean audio dataset.

The dataset must contain ONLY:
1. Pure real human speech
2. Pure AI/synthetic speech

Everything else must be deleted.

You must choose exactly one label:
- human
- ai
- Delete

Use ONLY the title and description.

LABEL DEFINITIONS:

human:
Choose "human" only when the video is clearly pure real human speech.
Good examples:
- real podcast episode
- real interview
- real vlog with a person speaking
- real lecture
- real sermon
- real debate
- real news report
- real speech
- real classroom recording
- real meeting or conversation
- real standup comedy
- real livestream with human speaking

Only choose human if there are no AI signals and no mixed-source signals.

ai:
Choose "ai" only when the video is clearly pure AI or synthetic speech.
Good examples:
- AI voice narration
- text-to-speech narration
- TTS video
- ElevenLabs voice narration
- synthetic voiceover
- AI-generated story narration
- AI Reddit story narration
- AI motivational narration
- AI facts narration
- AI history narration
- AI podcast made with synthetic voices
- AI avatar speaking with generated voice
- AI song or AI cover
- clearly generated celebrity voice

Only choose ai if the final audio itself appears to be AI/synthetic.

Delete:
Choose "Delete" for anything that is not cleanly pure human or pure AI.

Delete all of these:
- mixed human and AI
- human intro with AI narration
- AI clip with human commentary
- guides about AI
- tutorials about AI voice tools
- videos explaining how to make AI voices
- AI voice generator reviews
- tool comparisons
- human talking about AI
- videos that mention AI but do not clearly contain pure AI audio
- reaction videos
- compilations
- memes
- edits
- remixes
- music-only videos
- lyrics videos
- movie clips
- anime/cartoon clips
- gameplay
- no commentary videos
- sound effects
- trailers
- ads
- product demos
- unclear shorts
- unclear source
- title/description not enough to know
- anything misplaced for a clean voice dataset

Important decision logic:
- If clearly pure human speech, return human.
- If clearly pure AI speech, return ai.
- If mixed, return Delete.
- If it is about AI tools or teaches AI, return Delete.
- If uncertain, return Delete.
- Do not over-delete clear AI videos just because they are faceless.
- Do not label guides/tutorials as ai.
- Do not label human commentary about AI as ai.
- Do not label unclear videos as human.

Return ONLY valid JSON in this exact format:

{{
  "label": "human",
  "confidence": 0.0,
  "reason": "short reason"
}}

TITLE:
{title}

DESCRIPTION:
{description}
""".strip()

    try:
        raw = call_mistral(prompt)
        parsed = clean_json(raw)

        label = normalize_label(parsed.get("label", "Delete"))
        confidence = float(parsed.get("confidence", 0) or 0)
        reason = str(parsed.get("reason", ""))[:500]

        if confidence < MIN_CONFIDENCE:
            label = "Delete"
            reason = f"Low confidence. {reason}"

        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
        }

    except Exception as e:
        return {
            "label": "Delete",
            "confidence": 0,
            "reason": f"Classification error: {str(e)[:400]}",
        }


def ensure_columns(df):
    needed = {
        "label": "",
        "title": "",
        "description": "",
        "video_id": "",
        "mp3_path": "",
        "ai_check_done": "",
        "ai_decision": "",
        "ai_confidence": "",
        "ai_reason": "",
        "old_label": "",
        "moved_mp3_path": "",
        "classification_error": "",
    }

    for col, default in needed.items():
        if col not in df.columns:
            df[col] = default

    return df


def find_mp3_by_video_id(video_id):
    video_id = str(video_id or "").strip()

    if not video_id:
        return ""

    matches = list(MP3_BASE_DIR.rglob(f"*{video_id}*.mp3"))

    if matches:
        return str(matches[0])

    return ""


def move_mp3_to_delete(row):
    mp3_path = str(row.get("mp3_path", "")).strip()
    video_id = str(row.get("video_id", "")).strip()

    if not mp3_path or not Path(mp3_path).exists():
        mp3_path = find_mp3_by_video_id(video_id)

    if not mp3_path:
        return ""

    src = Path(mp3_path)

    if not src.exists():
        return ""

    DELETE_FOLDER.mkdir(parents=True, exist_ok=True)

    dst = DELETE_FOLDER / src.name

    counter = 1
    while dst.exists():
        dst = DELETE_FOLDER / f"{src.stem}_{counter}{src.suffix}"
        counter += 1

    shutil.move(str(src), str(dst))
    return str(dst)


def main():
    csv_path = Path(CSV_FILE)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = ensure_columns(df)

    total_human = 0
    total_ai = 0
    total_delete = 0
    total_skipped = 0
    total_errors = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        already_done = str(row.get("ai_check_done", "")).lower() == "true"

        if already_done and not FORCE_RECHECK:
            total_skipped += 1
            continue

        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()

        if not title and not description:
            result = {
                "label": "Delete",
                "confidence": 0,
                "reason": "Missing title and description",
            }
        else:
            result = classify_video(title, description)

        label = result["label"]

        try:
            df.at[i, "old_label"] = row.get("label", "")
            df.at[i, "label"] = label
            df.at[i, "ai_check_done"] = "true"
            df.at[i, "ai_decision"] = label
            df.at[i, "ai_confidence"] = str(result["confidence"])
            df.at[i, "ai_reason"] = result["reason"]
            df.at[i, "classification_error"] = ""

            if label == "Delete":
                if MOVE_DELETE_MP3:
                    moved_path = move_mp3_to_delete(row)

                    if moved_path:
                        df.at[i, "moved_mp3_path"] = moved_path
                        df.at[i, "mp3_path"] = moved_path

                total_delete += 1

            elif label == "human":
                total_human += 1

            elif label == "ai":
                total_ai += 1

            else:
                df.at[i, "label"] = "Delete"
                df.at[i, "ai_decision"] = "Delete"
                df.at[i, "ai_reason"] = "Invalid label returned, forced to Delete"
                total_delete += 1

        except KeyboardInterrupt:
            df.to_csv(csv_path, index=False)
            print("\nStopped manually. Progress saved.")
            return

        except Exception as e:
            df.at[i, "ai_check_done"] = ""
            df.at[i, "classification_error"] = str(e)[:500]
            total_errors += 1

        df.to_csv(csv_path, index=False)
        time.sleep(SLEEP_BETWEEN_CALLS)

    df.to_csv(csv_path, index=False)

    print("\nDone.")
    print(f"Updated CSV: {CSV_FILE}")
    print(f"Delete/review folder: {DELETE_FOLDER}")
    print(f"Human: {total_human}")
    print(f"AI: {total_ai}")
    print(f"Delete: {total_delete}")
    print(f"Skipped: {total_skipped}")
    print(f"Errors: {total_errors}")


if __name__ == "__main__":
    main()