import csv
import time
from pathlib import Path
import instaloader
from tqdm import tqdm

USERNAME = "fullstackraju"
OUT_DIR = Path("instagram_videos") / USERNAME
CSV_FILE = OUT_DIR / "metadata.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

L = instaloader.Instaloader(
    dirname_pattern=str(OUT_DIR),
    download_pictures=False,
    download_videos=True,
    download_video_thumbnails=False,
    download_geotags=False,
    download_comments=False,
    save_metadata=True,
    compress_json=False,
    post_metadata_txt_pattern="",
    max_connection_attempts=3,
)

# Optional but recommended if Instagram blocks public requests:
# L.login("YOUR_IG_USERNAME", "YOUR_IG_PASSWORD")

profile = instaloader.Profile.from_username(L.context, USERNAME)

existing_shortcodes = set()
if CSV_FILE.exists():
    with open(CSV_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing_shortcodes = {row["shortcode"] for row in reader if row.get("shortcode")}

write_header = not CSV_FILE.exists()

with open(CSV_FILE, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "shortcode",
            "url",
            "caption",
            "date_utc",
            "is_video",
            "video_view_count",
            "likes",
            "comments",
        ],
    )

    if write_header:
        writer.writeheader()

    posts = list(profile.get_posts())

    for post in tqdm(posts, desc=f"Downloading videos from @{USERNAME}"):
        try:
            if not post.is_video:
                continue

            if post.shortcode in existing_shortcodes:
                continue

            L.download_post(post, target=USERNAME)

            writer.writerow({
                "shortcode": post.shortcode,
                "url": f"https://www.instagram.com/p/{post.shortcode}/",
                "caption": (post.caption or "").replace("\n", " ").strip(),
                "date_utc": post.date_utc.isoformat(),
                "is_video": post.is_video,
                "video_view_count": post.video_view_count,
                "likes": post.likes,
                "comments": post.comments,
            })

            f.flush()
            existing_shortcodes.add(post.shortcode)

            time.sleep(3)

        except Exception as e:
            print(f"Failed on {post.shortcode}: {e}")
            time.sleep(10)

print("Done.")
print(f"Videos saved in: {OUT_DIR}")
print(f"Metadata CSV: {CSV_FILE}")