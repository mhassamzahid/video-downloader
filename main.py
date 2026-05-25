import os
import csv
import time
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

TARGET_PER_LABEL = 300
MIN_SECONDS = 4
MAX_SECONDS = 90

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

OUT_DIR = Path("video_dataset")
META_FILE = OUT_DIR / "metadata.csv"

REAL_KEYWORDS = [
    "street interview", "people walking", "office meeting", "city traffic",
    "family dinner", "students classroom", "sports game", "nature hiking",
    "news reporter", "restaurant cooking", "wedding guests", "gym workout",
    "market street", "doctor patient", "construction workers"
]

AI_KEYWORDS = [
    "ai generated", "digital human", "3d animation", "cgi face",
    "virtual influencer", "robot person", "futuristic avatar",
    "synthetic video", "surreal animation", "3d character",
    "metaverse avatar", "animated portrait", "ai art video",
    "digital face", "cyberpunk character"
]


def ensure_dirs():
    (OUT_DIR / "real").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ai").mkdir(parents=True, exist_ok=True)


def file_hash(content):
    return hashlib.sha256(content).hexdigest()


def metadata_exists(video_id, source):
    if not META_FILE.exists():
        return False

    with open(META_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] == source and row["source_video_id"] == str(video_id):
                return True
    return False


def count_label(label):
    folder = OUT_DIR / label
    return len(list(folder.glob("*.mp4")))


def save_metadata(row):
    file_exists = META_FILE.exists()

    with open(META_FILE, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
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
            "sha256"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def download_file(url, path):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    chunks = []
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 512):
            if chunk:
                chunks.append(chunk)
                f.write(chunk)

    return file_hash(b"".join(chunks))


def pick_pexels_file(video_files):
    if not video_files:
        return None

    mp4s = [v for v in video_files if v.get("file_type") == "video/mp4"]
    if not mp4s:
        mp4s = video_files

    return sorted(
        mp4s,
        key=lambda x: x.get("width", 0) * x.get("height", 0),
        reverse=True
    )[0]


def search_pexels(keyword, page=1):
    if not PEXELS_API_KEY:
        return []

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": keyword,
        "per_page": 80,
        "page": page,
        "orientation": "portrait"
    }

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("videos", [])


def search_pixabay(keyword, page=1):
    if not PIXABAY_API_KEY:
        return []

    url = "https://pixabay.com/api/videos/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": keyword,
        "per_page": 200,
        "page": page,
        "safesearch": "true",
        "video_type": "film"
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("hits", [])


def process_pexels_video(video, label, keyword):
    video_id = video["id"]

    if metadata_exists(video_id, "pexels"):
        return False

    duration = video.get("duration", 0)
    if duration < MIN_SECONDS or duration > MAX_SECONDS:
        return False

    chosen = pick_pexels_file(video.get("video_files", []))
    if not chosen:
        return False

    download_url = chosen["link"]
    width = chosen.get("width", "")
    height = chosen.get("height", "")

    filename = f"pexels_{video_id}.mp4"
    local_path = OUT_DIR / label / filename

    if local_path.exists():
        return False

    try:
        sha = download_file(download_url, local_path)
    except Exception as e:
        print(f"Download failed Pexels {video_id}: {e}")
        return False

    save_metadata({
        "label": label,
        "source": "pexels",
        "source_video_id": video_id,
        "keyword": keyword,
        "duration": duration,
        "width": width,
        "height": height,
        "page_url": video.get("url", ""),
        "download_url": download_url,
        "local_path": str(local_path),
        "sha256": sha
    })

    return True


def process_pixabay_video(video, label, keyword):
    video_id = video["id"]

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

    if not chosen:
        return False

    download_url = chosen["url"]
    width = chosen.get("width", "")
    height = chosen.get("height", "")

    filename = f"pixabay_{video_id}.mp4"
    local_path = OUT_DIR / label / filename

    if local_path.exists():
        return False

    try:
        sha = download_file(download_url, local_path)
    except Exception as e:
        print(f"Download failed Pixabay {video_id}: {e}")
        return False

    save_metadata({
        "label": label,
        "source": "pixabay",
        "source_video_id": video_id,
        "keyword": keyword,
        "duration": duration,
        "width": width,
        "height": height,
        "page_url": video.get("pageURL", ""),
        "download_url": download_url,
        "local_path": str(local_path),
        "sha256": sha
    })

    return True


def collect_label(label, keywords):
    print(f"\nCollecting {label} videos...")

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
                    process_pexels_video(video, label, keyword)
                    time.sleep(0.2)
            except Exception as e:
                print(f"Pexels error for {keyword}: {e}")

            try:
                pixabay_results = search_pixabay(keyword, page)
                for video in tqdm(pixabay_results, desc="Pixabay"):
                    if count_label(label) >= TARGET_PER_LABEL:
                        break
                    process_pixabay_video(video, label, keyword)
                    time.sleep(0.2)
            except Exception as e:
                print(f"Pixabay error for {keyword}: {e}")

        after = count_label(label)

        if after == before:
            print(f"No new {label} videos found on page {page}. Trying next page...")

        page += 1

        if page > 20:
            print(f"Stopped {label}: reached page limit.")
            break

    print(f"Finished {label}: {count_label(label)} videos")


def main():
    ensure_dirs()

    if not PEXELS_API_KEY and not PIXABAY_API_KEY:
        raise RuntimeError("Set at least one API key: PEXELS_API_KEY or PIXABAY_API_KEY")

    collect_label("real", REAL_KEYWORDS)
    collect_label("ai", AI_KEYWORDS)

    print("\nDone.")
    print(f"Real videos: {count_label('real')}")
    print(f"AI videos: {count_label('ai')}")
    print(f"Metadata: {META_FILE}")


if __name__ == "__main__":
    main()