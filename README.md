# Video Downloader - Audio Dataset Builder

A Python-based pipeline to build a clean audio dataset by collecting, downloading, and classifying YouTube and Instagram videos as either **real human speech** or **AI-generated speech**.

## 📋 Project Overview

This project automates the process of:

1. **Collecting videos** from YouTube using keyword searches filtered for real human and AI-generated content
2. **Downloading audio** (MP3) from collected videos
3. **Classifying videos** using Mistral AI API to determine if audio is human or AI-generated
4. **Organizing datasets** into labeled folders (`ai/` and `real/`) for machine learning use
5. **Managing deletions** by moving videos that don't fit clean dataset criteria to a review folder

The goal is to create high-quality, curated datasets for training speech recognition, AI detection, or audio analysis models.

## 🗂️ Project Structure

```
video-downloader/
├── main.py                              # YouTube video collection script
├── download_mp3.py                      # MP3 download from YouTube URLs
├── classify_ai_with_mistral.py         # Mistral AI classification service
├── instagram_video_downloader.py       # Instagram video/audio downloader
├── reset.py                             # Reset classification columns for re-checking
├── requirements.txt                     # Python dependencies
├── youtube_video_metadata.csv          # CSV storing collected video metadata
├── mp3_dataset/                        # Directory containing downloaded MP3s
│   ├── ai/                             # AI-generated speech audio files
│   └── real/                           # Real human speech audio files
├── mp3_delete_review/                  # MP3s moved to deletion folder for review
├── instagram_videos/                   # Instagram video downloads
└── venv/                               # Python virtual environment
```

## 🔄 Workflow

### Phase 1: Video Collection (`main.py`)

Searches YouTube using keyword lists and collects video metadata:
- **Real keywords**: street interviews, podcasts, lectures, news interviews, vlogs, speeches
- **AI keywords**: AI avatars, digital humans, synthesia, heygen, etc.
- Applies pre-filtering to ensure videos match expected categories
- Stores results in `youtube_video_metadata.csv`

**CSV columns added:**
- `label`, `keyword`, `video_id`, `title`, `description`, `channel_title`
- `published_at`, `duration`, `view_count`, `like_count`, `comment_count`, `url`

### Phase 2: Audio Download (`download_mp3.py`)

Converts YouTube videos to MP3 audio:
- Uses `yt-dlp` to download best available audio quality
- Uses FFmpeg to convert audio to MP3 (192 kbps)
- Organizes files by label: `mp3_dataset/ai/` and `mp3_dataset/real/`
- Tracks download status, retries failed downloads (up to 3 attempts)
- Updates CSV with `mp3_done`, `mp3_path`, `mp3_error`, `mp3_attempts`

**Features:**
- Skips already-downloaded videos
- Resets and retries failed downloads on next run
- Handles network interruptions with retry logic

### Phase 3: AI Classification (`classify_ai_with_mistral.py`)

Uses Mistral Small LLM to classify video content as human, AI, or delete:
- **Pre-filters** with keyword matching for quick decisions
- **Calls Mistral API** for ambiguous cases with detailed classification prompt
- **Labels** videos as:
  - `human` - clearly pure real human speech
  - `ai` - clearly pure AI/synthetic speech
  - `Delete` - mixed content, tutorials, unclear, etc.
- Moves deleted MP3s to `mp3_delete_review/` for manual verification
- Stores classification results in CSV columns: `ai_check_done`, `ai_decision`, `ai_confidence`, `ai_reason`

**Pre-filter covers:**
- Tutorials, guides, reviews about AI tools
- Mixed human and AI content
- Non-speech content (music, gameplay, trailers)
- Unclear or low-confidence content

### Phase 4: Instagram Downloads (`instagram_video_downloader.py`)

Downloads videos from Instagram profiles:
- Uses `instaloader` library to fetch videos
- Saves metadata to CSV: shortcode, caption, date, engagement metrics
- Skips already-downloaded posts
- Organized by username in `instagram_videos/` directory

## 📦 Dependencies

```
annotated-types==0.7.0
anyio==4.13.0
certifi==2024.X.X
charset-normalizer==3.X.X
click==8.X.X
instaloader==4.X.X
pandas==2.X.X
python-dotenv==1.X.X
requests==2.X.X
yt-dlp==2024.X.X
tqdm==4.X.X
```

**Required external tools:**
- **FFmpeg**: For audio conversion (set in `download_mp3.py`: `C:\ffmpeg\bin`)
- **Node.js**: For YouTube player JavaScript handling (set in `download_mp3.py`: `C:\Program Files\nodejs\node.exe`)

**API Keys required:**
- `YOUTUBE_API_KEY` - Google YouTube Data API v3
- `MISTRAL_API_KEY` - Mistral AI API

## 🚀 Setup

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd video-downloader
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
venv\Scripts\activate.bat    # Windows CMD
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install External Tools

- **FFmpeg**: Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract to `C:\ffmpeg\`
- **Node.js**: Download from [nodejs.org](https://nodejs.org/), install to default location

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Getting API Keys:**

- **YouTube API**: 
  1. Go to [Google Cloud Console](https://console.cloud.google.com/)
  2. Create new project
  3. Enable YouTube Data API v3
  4. Create OAuth 2.0 credentials
  
- **Mistral API**:
  1. Sign up at [console.mistral.ai](https://console.mistral.ai/)
  2. Generate API key from account settings

## 📝 Usage

### Collect YouTube Videos

```bash
python main.py
```

Collects up to 300 videos per label (600 total: real + AI) from recent videos (last 30 days) and saves to `youtube_video_metadata.csv`.

**Configuration in `main.py`:**
- `MAX_PER_LABEL = 300` - Videos to collect per category
- `DAYS_BACK = 30` - Search recent videos from X days ago

### Download MP3 Files

```bash
python download_mp3.py
```

Downloads audio from all videos in CSV not yet downloaded. Creates MP3 files in `mp3_dataset/real/` and `mp3_dataset/ai/` folders.

**Configuration in `download_mp3.py`:**
- `MAX_ATTEMPTS_PER_VIDEO = 3` - Retry failed downloads
- `FFMPEG_PATH` - Adjust if FFmpeg installed elsewhere
- `NODE_PATH` - Adjust if Node.js installed elsewhere

### Classify Videos with Mistral AI

```bash
python classify_ai_with_mistral.py
```

Classifies all videos in CSV as human, AI, or delete. Moves deleted MP3s to `mp3_delete_review/` folder.

**Configuration in `classify_ai_with_mistral.py`:**
- `MIN_CONFIDENCE = 0.70` - Confidence threshold (lower = more deletions)
- `FORCE_RECHECK = True` - Re-classify already classified videos
- `MOVE_DELETE_MP3 = True` - Move deleted MP3s to review folder
- `SLEEP_BETWEEN_CALLS = 0.5` - Delay between Mistral API calls

### Reset Classification (Start Over)

```bash
python reset.py
```

Clears all classification results to re-run `classify_ai_with_mistral.py` with different settings.

### Download Instagram Videos

```bash
python instagram_video_downloader.py
```

Downloads videos from Instagram profile. Edit `USERNAME` variable in script to change profile.

**Supports:**
- Optional Instagram login (uncomment and add credentials)
- Skips already-downloaded posts
- Saves video metadata to CSV

## 📊 CSV Columns Explained

### After `main.py` (Collection):
| Column | Description |
|--------|-------------|
| `label` | "ai" or "real" (initial classification from keywords) |
| `video_id` | YouTube video ID |
| `title` | Video title |
| `description` | Video description |
| `url` | Full YouTube URL |
| `view_count`, `like_count`, `comment_count` | Video engagement metrics |
| `duration` | Video duration |

### After `download_mp3.py` (Audio Download):
| Column | Description |
|--------|-------------|
| `mp3_done` | "true" if MP3 downloaded successfully |
| `mp3_path` | Full path to downloaded MP3 file |
| `mp3_error` | Error message if download failed |
| `mp3_attempts` | Number of attempts made |

### After `classify_ai_with_mistral.py` (Classification):
| Column | Description |
|--------|-------------|
| `ai_check_done` | "true" if classification complete |
| `ai_decision` | Final label: "human", "ai", or "Delete" |
| `ai_confidence` | Mistral confidence score (0.0-1.0) |
| `ai_reason` | Explanation for classification |
| `moved_mp3_path` | Path if MP3 was moved to delete folder |
| `old_label` | Original label before Mistral reclassification |

## 🎯 Key Features

✅ **Automatic Retries**: Failed downloads retry up to 3 times  
✅ **Incremental Processing**: Resume from where you left off  
✅ **Error Tracking**: All failures logged with reasons  
✅ **Manual Review**: Deleted MP3s moved to folder for inspection  
✅ **API Efficiency**: Keyword pre-filtering reduces Mistral API calls  
✅ **Progress Tracking**: Uses `tqdm` for real-time progress bars  
✅ **Windows Compatible**: Paths and executables configured for Windows  

## ⚙️ Configuration Tips

### Speed Up Processing
```python
# In classify_ai_with_mistral.py
SLEEP_BETWEEN_CALLS = 0.1  # Reduce from 0.5 (faster API calls)
MIN_CONFIDENCE = 0.85      # Higher threshold = fewer API calls
```

### Save on API Costs
```python
# In classify_ai_with_mistral.py
FORCE_RECHECK = False      # Skip already-classified videos
MIN_CONFIDENCE = 0.95      # Higher = trust pre-filter more
```

### Collect More Videos
```python
# In main.py
MAX_PER_LABEL = 500        # Increase from 300
DAYS_BACK = 90             # Search further back in time
```

## 🔍 Classification Logic

**Pre-filter (keyword-based):**
- Delete: tutorials, how-to guides, comparisons, reactions, etc.
- AI: Text-to-speech, ElevenLabs, synthesia, digital human, etc.
- Human: podcasts, interviews, lectures, vlogs, speeches, etc.

**Mistral Classification (for unclear cases):**
- Uses detailed prompt to distinguish pure human vs pure AI speech
- Rejects mixed content (e.g., human intro + AI narration)
- Lower confidence scores automatically marked for deletion

## 📝 Troubleshooting

### "YOUTUBE_API_KEY missing in .env"
Make sure `.env` file exists in project root with your API key.

### "MISTRAL_API_KEY missing in .env"
Add your Mistral API key to `.env` file.

### "FFmpeg not found"
Ensure FFmpeg is installed and path in `download_mp3.py` is correct.

### "No module named 'yt_dlp'"
Run `pip install -r requirements.txt` to install all dependencies.

### YouTube blocks downloads
- Try using Instagram downloader for alternative sources
- Adjust retry logic in `download_mp3.py`
- Add Instagram login credentials in `instagram_video_downloader.py`

### Mistral API rate limit
- Increase `SLEEP_BETWEEN_CALLS` in `classify_ai_with_mistral.py`
- Check Mistral API pricing and rate limits

## 📈 Example Workflow

```bash
# 1. Collect 300 "real" + 300 "ai" videos from YouTube
python main.py

# 2. Download MP3s (may take a while - hundreds of files)
python download_mp3.py

# 3. Classify with AI and organize dataset
python classify_ai_with_mistral.py

# 4. Review and move any questionable files
# - Check mp3_delete_review/ folder
# - Move back to mp3_dataset/ if classification was wrong

# 5. Final dataset ready in:
# mp3_dataset/real/*.mp3    <- All real human speech
# mp3_dataset/ai/*.mp3      <- All AI-generated speech
```

## 🤝 Contributing

To improve classification:
1. Review videos in `mp3_delete_review/` folder
2. Check which ones were incorrectly deleted in CSV
3. Update keyword pre-filters or Mistral prompt in code
4. Run `reset.py` and re-classify with improved logic

## 📄 License

[Add your license here]

## 👤 Author

[Your information]

---

**Last Updated**: May 2026  
**Project Status**: Active Development
