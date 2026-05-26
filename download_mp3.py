import os
import re
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from yt_dlp import YoutubeDL


CSV_FILE = "youtube_video_metadata.csv"
OUT_DIR = Path("mp3_dataset")

NODE_PATH = r"C:\Program Files\nodejs\node.exe"
FFMPEG_PATH = r"C:\ffmpeg\bin"

MAX_ATTEMPTS_PER_VIDEO = 3


def safe_filename(text):
    text = str(text or "")
    text = re.sub(r'[<>:"/\\|?*#]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:80] or "video"


def normalize_youtube_url(row):
    url = str(row.get("url", "")).strip()
    video_id = str(row.get("video_id", "")).strip()

    if url:
        return url

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"

    return ""


def find_created_mp3(folder, video_id):
    matches = list(folder.glob(f"*{video_id}*.mp3"))
    if matches:
        return str(matches[0])
    return ""


def download_mp3(url, output_base_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_base_path) + ".%(ext)s",
        "noplaylist": True,

        # Important: don't let yt-dlp silently continue internally
        "ignoreerrors": False,

        # More reliable network behavior
        "socket_timeout": 60,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "file_access_retries": 5,
        "continuedl": True,

        # Avoid some YouTube throttling issues
        "concurrent_fragment_downloads": 1,

        # Paths
        "ffmpeg_location": FFMPEG_PATH,

        # Node JS for YouTube player JS
        "js_runtimes": {
            "node": {
                "path": NODE_PATH
            }
        },

        # Convert to mp3
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],

        # Safer Windows filenames
        "restrictfilenames": False,
        "windowsfilenames": True,

        # Show actual errors
        "quiet": False,
        "no_warnings": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def load_csv():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE, dtype=str).fillna("")

    for col in ["mp3_done", "mp3_path", "mp3_error", "mp3_attempts"]:
        if col not in df.columns:
            df[col] = ""

    return df


def reset_failed_rows(df):
    """
    This is important.
    Failed rows should be retried every run instead of skipped forever.
    """
    for i, row in df.iterrows():
        done = str(row.get("mp3_done", "")).lower() == "true"
        path = str(row.get("mp3_path", ""))

        if done and path and os.path.exists(path):
            continue

        if done and (not path or not os.path.exists(path)):
            df.at[i, "mp3_done"] = ""
            df.at[i, "mp3_path"] = ""
            df.at[i, "mp3_error"] = "Reset because MP3 file was missing"

        if str(row.get("mp3_done", "")).lower() == "false":
            df.at[i, "mp3_done"] = ""

    return df


def save(df):
    df.to_csv(CSV_FILE, index=False)


def main():
    df = load_csv()
    df = reset_failed_rows(df)

    OUT_DIR.mkdir(exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        existing_path = str(row.get("mp3_path", ""))

        if (
            str(row.get("mp3_done", "")).lower() == "true"
            and existing_path
            and os.path.exists(existing_path)
        ):
            continue

        label = safe_filename(row.get("label", "unknown"))
        video_id = str(row.get("video_id", "")).strip()
        title = safe_filename(row.get("title", "video"))
        url = normalize_youtube_url(row)

        if not video_id:
            df.at[i, "mp3_done"] = ""
            df.at[i, "mp3_path"] = ""
            df.at[i, "mp3_error"] = "Missing video_id"
            save(df)
            continue

        if not url or ("youtube.com" not in url and "youtu.be" not in url):
            df.at[i, "mp3_done"] = ""
            df.at[i, "mp3_path"] = ""
            df.at[i, "mp3_error"] = "Missing or invalid YouTube URL"
            save(df)
            continue

        label_dir = OUT_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)

        output_base = label_dir / f"{label}_{video_id}_{title}"
        expected_mp3 = str(output_base) + ".mp3"

        existing_found = find_created_mp3(label_dir, video_id)
        if os.path.exists(expected_mp3) or existing_found:
            df.at[i, "mp3_done"] = "true"
            df.at[i, "mp3_path"] = expected_mp3 if os.path.exists(expected_mp3) else existing_found
            df.at[i, "mp3_error"] = ""
            save(df)
            continue

        success = False
        last_error = ""

        for attempt in range(1, MAX_ATTEMPTS_PER_VIDEO + 1):
            try:
                print(f"\nDownloading attempt {attempt}/{MAX_ATTEMPTS_PER_VIDEO}: {url}")

                download_mp3(url, output_base)

                found_mp3 = expected_mp3 if os.path.exists(expected_mp3) else find_created_mp3(label_dir, video_id)

                if found_mp3 and os.path.exists(found_mp3):
                    df.at[i, "mp3_done"] = "true"
                    df.at[i, "mp3_path"] = found_mp3
                    df.at[i, "mp3_error"] = ""
                    df.at[i, "mp3_attempts"] = str(attempt)
                    success = True
                    break

                last_error = "Download finished but MP3 file was not found"

            except KeyboardInterrupt:
                df.at[i, "mp3_done"] = ""
                df.at[i, "mp3_error"] = "Stopped manually"
                save(df)
                print("\nStopped. Progress saved.")
                return

            except Exception as e:
                last_error = str(e)[:500]
                print(f"Error attempt {attempt}: {last_error}")

                time.sleep(3)

        if not success:
            # Important: leave mp3_done empty so the row retries next run
            df.at[i, "mp3_done"] = ""
            df.at[i, "mp3_path"] = ""
            df.at[i, "mp3_error"] = last_error
            df.at[i, "mp3_attempts"] = str(MAX_ATTEMPTS_PER_VIDEO)

        save(df)

    save(df)

    print("Done.")
    print(f"Updated CSV: {CSV_FILE}")
    print(f"MP3 folder: {OUT_DIR}")


if __name__ == "__main__":
    main()