import os
import csv
import time
import json
import math
import hashlib
import requests
import numpy as np
import webrtcvad

from pathlib import Path
from tqdm import tqdm
from moviepy import VideoFileClip


TARGET_PER_LABEL = 300
MIN_SECONDS = 4
MAX_SECONDS = 90

OUT_DIR = Path("video_dataset")
META_FILE = OUT_DIR / "metadata.csv"

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

REAL_KEYWORDS = [
    "interview",
    "people talking",
    "podcast",
    "news reporter",
    "teacher speaking",
    "street interview",
    "vlog talking",
    "presentation speech",
    "public speaking",
    "conversation"
]

AI_KEYWORDS = [
    "ai generated talking",
    "digital human talking",
    "virtual influencer talking",
    "ai avatar speaking",
    "synthetic person speaking",
    "cgi character talking",
    "3d avatar talking",
    "robot voice video",
    "virtual presenter",
    "animated character speaking"
]


def ensure_dirs():
    (OUT_DIR / "real").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ai").mkdir(parents=True, exist_ok=True)


def count_label(label):
    return len(list((OUT_DIR / label).glob("*.mp4")))


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def metadata_exists(video_id, source):
    if not META_FILE.exists():
        return False

    with open(META_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] == source and row["source_video_id"] == str(video_id):
                return True

    return False


def save_metadata(row):
    file_exists = META_FILE.exists()

    fields = [
        "label",
        "source",
        "source_video_id",
        "keyword",
        "duration",
        "width",
        "height",
        "page_url",
        "download_url",
        "local_path",
        "sha256",
        "voice_ratio"
    ]

    with open(META_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def download_file(url, path):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)


def get_video_duration(path):
    try:
        clip = VideoFileClip(str(path))
        duration = clip.duration or 0
        clip.close()
        return duration
    except Exception:
        return 0


def detect_voice(path, sample_seconds=15, min_voice_ratio=0.08):
    """
    Uses WebRTC VAD to detect human-like speech.
    This is library-only, no direct ffmpeg command.
    """

    try:
        clip = VideoFileClip(str(path))

        if clip.audio is None:
            clip.close()
            return False, 0.0

        duration = min(float(clip.duration or 0), sample_seconds)

        if duration <= 0:
            clip.close()
            return False, 0.0

        audio = clip.audio.subclipped(0, duration).to_soundarray(fps=16000)
        clip.close()

        if audio.size == 0:
            return False, 0.0

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = np.nan_to_num(audio)

        max_abs = np.max(np.abs(audio))
        if max_abs < 0.003:
            return False, 0.0

        audio_int16 = np.clip(audio, -1, 1)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)

        vad = webrtcvad.Vad(2)

        sample_rate = 16000
        frame_ms = 30
        frame_size = int(sample_rate * frame_ms / 1000)

        total_frames = 0
        speech_frames = 0

        for start in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[start:start + frame_size]
            frame_bytes = frame.tobytes()

            total_frames += 1

            try:
                if vad.is_speech(frame_bytes, sample_rate):
                    speech_frames += 1
            except Exception:
                continue

        if total_frames == 0:
            return False, 0.0

        voice_ratio = speech_frames / total_frames
        return voice_ratio >= min_voice_ratio, round(voice_ratio, 4)

    except Exception:
        return False, 0.0


def search_pexels(keyword, page):
    if not PEXELS_API_KEY:
        return []

    url = "https://api.pexels.com/videos/search"

    headers = {
        "Authorization": PEXELS_API_KEY
    }

    params = {
        "query": keyword,
        "per_page": 80,
        "page": page,
        "orientation": "portrait"
    }

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()

    return r.json().get("videos", [])


def search_pixabay(keyword, page):
    if not PIXABAY_API_KEY:
        return []

    url = "https://pixabay.com/api/videos/"

    params = {
        "key": PIXABAY_API_KEY,
        "q": keyword,
        "per_page": 200,
        "page": page,
        "safesearch": "true"
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    return r.json().get("hits", [])


def pick_pexels_file(video_files):
    files = [
        f for f in video_files
        if f.get("file_type") == "video/mp4" and f.get("link")
    ]

    if not files:
        return None

    files = sorted(
        files,
        key=lambda x: x.get("width", 0) * x.get("height", 0),
        reverse=True
    )

    return files[0]


def process_pexels(video, label, keyword):
    video_id = video.get("id")

    if not video_id:
        return False

    if metadata_exists(video_id, "pexels"):
        return False

    duration = video.get("duration", 0)

    if duration < MIN_SECONDS or duration > MAX_SECONDS:
        return False

    chosen = pick_pexels_file(video.get("video_files", []))

    if not chosen:
        return False

    filename = f"pexels_{video_id}.mp4"
    local_path = OUT_DIR / label / filename

    if local_path.exists():
        return False

    try:
        download_file(chosen["link"], local_path)

        has_voice, voice_ratio = detect_voice(local_path)

        if not has_voice:
            print(f"Deleted no-voice video: {local_path}")
            local_path.unlink(missing_ok=True)
            return False

        sha = sha256_file(local_path)

        save_metadata({
            "label": label,
            "source": "pexels",
            "source_video_id": video_id,
            "keyword": keyword,
            "duration": duration,
            "width": chosen.get("width", ""),
            "height": chosen.get("height", ""),
            "page_url": video.get("url", ""),
            "download_url": chosen.get("link", ""),
            "local_path": str(local_path),
            "sha256": sha,
            "voice_ratio": voice_ratio
        })

        return True

    except Exception as e:
        print(f"Pexels failed {video_id}: {e}")
        local_path.unlink(missing_ok=True)
        return False


def process_pixabay(video, label, keyword):
    video_id = video.get("id")

    if not video_id:
        return False

    if metadata_exists(video_id, "pixabay"):
        return False

    duration = video.get("duration", 0)

    if duration < MIN_SECONDS or duration > MAX_SECONDS:
        return False

    videos = video.get("videos", {})

    chosen = (
        videos.get("large")
        or videos.get("medium")
        or videos.get("small")
        or videos.get("tiny")
    )

    if not chosen or not chosen.get("url"):
        return False

    filename = f"pixabay_{video_id}.mp4"
    local_path = OUT_DIR / label / filename

    if local_path.exists():
        return False

    try:
        download_file(chosen["url"], local_path)

        has_voice, voice_ratio = detect_voice(local_path)

        if not has_voice:
            print(f"Deleted no-voice video: {local_path}")
            local_path.unlink(missing_ok=True)
            return False

        sha = sha256_file(local_path)

        save_metadata({
            "label": label,
            "source": "pixabay",
            "source_video_id": video_id,
            "keyword": keyword,
            "duration": duration,
            "width": chosen.get("width", ""),
            "height": chosen.get("height", ""),
            "page_url": video.get("pageURL", ""),
            "download_url": chosen.get("url", ""),
            "local_path": str(local_path),
            "sha256": sha,
            "voice_ratio": voice_ratio
        })

        return True

    except Exception as e:
        print(f"Pixabay failed {video_id}: {e}")
        local_path.unlink(missing_ok=True)
        return False


def collect_label(label, keywords):
    print(f"\nCollecting {label} videos with voice...")

    page = 1

    while count_label(label) < TARGET_PER_LABEL:
        before = count_label(label)

        for keyword in keywords:
            if count_label(label) >= TARGET_PER_LABEL:
                break

            print(f"\n[{label}] keyword='{keyword}' page={page}")

            try:
                pexels_results = search_pexels(keyword, page)

                for video in tqdm(pexels_results, desc="Pexels"):
                    if count_label(label) >= TARGET_PER_LABEL:
                        break

                    process_pexels(video, label, keyword)
                    time.sleep(0.15)

            except Exception as e:
                print(f"Pexels search error: {e}")

            try:
                pixabay_results = search_pixabay(keyword, page)

                for video in tqdm(pixabay_results, desc="Pixabay"):
                    if count_label(label) >= TARGET_PER_LABEL:
                        break

                    process_pixabay(video, label, keyword)
                    time.sleep(0.15)

            except Exception as e:
                print(f"Pixabay search error: {e}")

        after = count_label(label)

        print(f"{label}: {after}/{TARGET_PER_LABEL}")

        page += 1

        if page > 25:
            print(f"Stopped {label}. Page limit reached.")
            break

        if after == before:
            print("No new accepted videos on this page. Continuing...")


def main():
    ensure_dirs()

    if not PEXELS_API_KEY and not PIXABAY_API_KEY:
        raise RuntimeError(
            "Missing API keys. Set PEXELS_API_KEY and/or PIXABAY_API_KEY."
        )

    collect_label("real", REAL_KEYWORDS)
    collect_label("ai", AI_KEYWORDS)

    print("\nDone.")
    print(f"Real videos: {count_label('real')}")
    print(f"AI videos: {count_label('ai')}")
    print(f"Metadata saved to: {META_FILE}")


if __name__ == "__main__":
    main()