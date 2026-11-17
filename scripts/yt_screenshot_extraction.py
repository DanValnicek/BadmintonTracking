import os
import sys
import json
import subprocess
from urllib.parse import urlparse, parse_qs


def get_video_id(url):
    """Extract YouTube video ID for clean file naming."""
    parsed = urlparse(url)
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        return parse_qs(parsed.query).get('v', ['unknown'])[0]
    elif parsed.hostname == 'youtu.be':
        return parsed.path.lstrip('/')
    return "video"


def get_video_info(url, max_height=1080):
    """
    Fetches video metadata (duration) and stream URL in a single fast call.
    """
    cmd = [
        "yt-dlp",
        "-j",  # Dump metadata as JSON
        "-f", f"bestvideo[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}]/best",
        url
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode('utf-8')
        info = json.loads(output)
        return info.get('duration'), info.get('url')
    except Exception as e:
        print(f"❌ Failed to fetch info for {url}: {e}")
        return None, None


def extract_single_frame(stream_url, timestamp_sec, output_path):
    """
    Directly seeks to timestamp_sec using HTTP Range Requests (-ss BEFORE -i).
    This takes ~0.2 seconds per frame instead of decoding the stream.
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-ss", str(timestamp_sec),  # Seeking BEFORE input is crucial for HTTP speed
        "-i", stream_url,
        "-vframes", "1",
        "-q:v", "2",
        output_path,
        "-y"
    ]
    res = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0


def process_youtube_video(url, output_dir, interval_sec=30):
    video_id = get_video_id(url)
    print(f"\n🌐 Fetching info for Video: {video_id}...")

    duration, stream_url = get_video_info(url)
    if not duration or not stream_url:
        print(f"  ❌ Skipping {url} due to extraction failure.")
        return []

    duration_min = duration / 60
    print(f"  ⏱️ Video Duration: {duration_min:.1f} mins. Extracting frames every {interval_sec}s...")

    manifest_entries = []

    # Generate timestamps: 0s, 30s, 60s, 90s... up to video end
    timestamps = list(range(0, int(duration), interval_sec))

    for idx, ts in enumerate(timestamps, 1):
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        ts_formatted = f"{minutes:02d}m{seconds:02d}s"

        filename = f"{video_id}_ts{ts_formatted}.jpg"
        output_path = os.path.join(output_dir, filename)

        # Fast HTTP Range Seek
        success = extract_single_frame(stream_url, ts, output_path)

        if success and os.path.exists(output_path):
            manifest_entries.append({
                "image_filename": filename,
                "youtube_url": url,
                "video_id": video_id,
                "timestamp_sec": ts,
                "timestamp_formatted": ts_formatted
            })
            sys.stdout.write(f"\r  └─ Progress: [{idx}/{len(timestamps)}] Extracted {filename}")
            sys.stdout.flush()

    print(f"\n  ✅ Completed {video_id}! Saved {len(manifest_entries)} frames.")
    return manifest_entries


def main(links_file, output_dir, interval_sec=30):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")

    if not os.path.exists(links_file):
        print(f"❌ Input links file '{links_file}' not found.")
        sys.exit(1)

    with open(links_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"📋 Loaded {len(urls)} YouTube links.")

    all_manifest = {}
    for i, url in enumerate(urls, 1):
        print(f"\n--- [{i}/{len(urls)}] ---")
        entries = process_youtube_video(url, output_dir, interval_sec=interval_sec)
        for entry in entries:
            all_manifest[entry["image_filename"]] = entry

    # Save combined manifest
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        json.dump(all_manifest, mf, indent=4)

    print(f"\n🎉 Done! Total frames extracted: {len(all_manifest)}")
    print(f"📄 Manifest updated at: {manifest_path}")


if __name__ == "__main__":
    LINKS_FILE = "../amateur_badminton_videos_yt.txt"
    OUTPUT_FOLDER = "dataset/unfiltered_frames2"
    SAMPLE_INTERVAL_SECONDS = 30

    main(LINKS_FILE, OUTPUT_FOLDER, interval_sec=SAMPLE_INTERVAL_SECONDS)