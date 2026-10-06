import os
import json
import uuid
import base64
import re
import tempfile
import subprocess
import yt_dlp
import google.generativeai as genai
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


# ---------------- DOWNLOAD ----------------

def download_video(url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)

    opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info), info


# ---------------- HELPERS ----------------

def get_duration(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video],
        capture_output=True,
        text=True
    )

    try:
        return float(result.stdout.strip())
    except ValueError:
        return 60.0


def extract_frames(video, count=12):
    duration = get_duration(video)
    frames = []

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, count + 1):
            timestamp = duration * i / (count + 1)
            path = os.path.join(tmp, f"{i}.jpg")

            subprocess.run([
                "ffmpeg", "-ss", str(timestamp), "-i", video,
                "-frames:v", "1", "-q:v", "5", path, "-y"
            ], capture_output=True)

            if os.path.exists(path):
                with open(path, "rb") as f:
                    frames.append({
                        "timestamp": timestamp,
                        "data": base64.b64encode(f.read()).decode()
                    })

    return frames, duration


def parse_json(text):
    text = text.strip()

    if "```" in text:
        text = text.split("```")[1]
        text = re.sub(r"^json", "", text, flags=re.I).strip()

    return json.loads(text)


# ---------------- GEMINI ANALYSIS ----------------

def analyze_video(video, clip_count=5):
    if not API_KEY:
        return []

    frames, duration = extract_frames(video)

    prompt = f"""
You are an expert short-form video editor.

Find the {clip_count} best viral moments from this video.

Rules:
- Start before the hook.
- End after the payoff.
- No overlapping clips.
- Each clip must be 20-60 seconds.
- The clip should make sense by itself.
- Rank by virality.

Return ONLY JSON:

[
  {{
    "start_time": 0,
    "end_time": 30,
    "description": "Why this clip is interesting",
    "hook": "Opening hook",
    "virality_score": 9,
    "clip_type": "funny"
  }}
]
"""

    content = [prompt]

    for frame in frames:
        content.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": frame["data"]
            }
        })

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(content)

    clips = parse_json(response.text)
    return sorted(
        clips,
        key=lambda x: x.get("virality_score", 0),
        reverse=True
    )


# ---------------- CAPTIONS ----------------

def generate_captions(video, output):
    try:
        import whisper
    except ImportError:
        return False

    model = whisper.load_model("base")
    result = model.transcribe(video, word_timestamps=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write(
            "Format: Name, Fontname, Fontsize, PrimaryColour,"
            " SecondaryColour, OutlineColour, BackColour, Bold,"
            " Italic, Underline, StrikeOut, ScaleX, ScaleY,"
            " Spacing, Angle, BorderStyle, Outline, Shadow,"
            " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )
        f.write(
            "Style: Default,Arial Black,60,"
            "&H00FFFFFF,&H000000FF,&H00000000,"
            "&H00000000,-1,0,0,0,100,100,0,0,1,3,2,2,40,40,300,1\n"
        )

        f.write("\n[Events]\n")
        f.write(
            "Format: Layer, Start, End, Style, Name,"
            " MarginL, MarginR, MarginV, Effect, Text\n"
        )

        for segment in result["segments"]:
            start = format_ass(segment["start"])
            end = format_ass(segment["end"])
            text = segment["text"].strip().upper()

            f.write(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
            )

    return True


def format_ass(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ---------------- RENDER ----------------

def render_clip(video, start, end, output, captions=True):
    duration = end - start

    raw = output.replace(".mp4", "_raw.mp4")

    vf = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=40,"
        "eq=brightness=-0.3[bg];"
        "[fg]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black@0[fg];"
        "[bg][fg]overlay=0:0"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video,
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac",
        raw
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:])

    if not captions:
        os.replace(raw, output)
        return output

    ass = output.replace(".mp4", ".ass")

    if not generate_captions(raw, ass):
        os.replace(raw, output)
        return output

    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw,
        "-vf", f"ass={ass}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "copy",
        output
    ], check=True)

    os.remove(raw)
    os.remove(ass)

    return output


# ---------------- CREATE CLIPS ----------------

def create_clips(video, clips, output_dir="output", captions=True):
    os.makedirs(output_dir, exist_ok=True)

    tasks = []

    for i, clip in enumerate(clips):
        start = max(0, float(clip["start_time"]))
        end = min(get_duration(video), float(clip["end_time"]))

        output = os.path.join(
            output_dir,
            f"clip_{i}_{uuid.uuid4().hex[:6]}.mp4"
        )

        tasks.append((video, start, end, output, captions))

    results = []

    with ProcessPoolExecutor(max_workers=min(4, len(tasks))) as executor:
        futures = [
            executor.submit(render_clip, *task)
            for task in tasks
        ]

        for future in as_completed(futures):
            try:
                results.append(future.result())
                print("✓ Clip created")
            except Exception as e:
                print("✗ Clip failed:", e)

    return results


# ---------------- MAIN ----------------

if __name__ == "__main__":
    url = input("Enter video URL: ").strip()

    video, info = download_video(url)

    print("Analyzing video...")
    clips = analyze_video(video, clip_count=5)

    print(f"Found {len(clips)} clips")

    create_clips(
        video,
        clips,
        output_dir="output",
        captions=True
    )

    print("Done! Check the output folder.")
