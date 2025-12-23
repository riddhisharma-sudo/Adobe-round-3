import os
import json
import requests
import re
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from pydub import AudioSegment


load_dotenv()

def _generate_azure_tts(text, output_file, voice=None):
    """Generate audio using Azure OpenAI TTS."""
    api_key = os.getenv("AZURE_TTS_KEY")
    endpoint = os.getenv("AZURE_TTS_ENDPOINT")
    deployment = os.getenv("AZURE_TTS_DEPLOYMENT", "tts")
    api_version = os.getenv("AZURE_TTS_API_VERSION", "2025-03-01-preview")
    
    if not api_key or not endpoint:
        raise ValueError("AZURE_TTS_KEY and AZURE_TTS_ENDPOINT must be set for Azure OpenAI TTS")
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": deployment,
        "input": text,
        "voice": voice,
    }
    
    try:
        response = requests.post(
            f"{endpoint}/openai/deployments/{deployment}/audio/speech?api-version={api_version}",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"[azure_tts] saved: {output_file}")
        return output_file
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Azure OpenAI TTS failed: {e}")


# ---------------- Helpers & Config ----------------

curr_dir = os.path.dirname(__file__)

JSON_FILE = os.path.join(curr_dir, "output", "podcast.json")

# Use outputs/podcast_voice directory
backend_dir = os.path.dirname(curr_dir)  
project_root = os.path.dirname(backend_dir)
OUTPUT_DIR = Path(os.path.join(project_root, "outputs", "podcast_voice"))
TEMP_DIR = OUTPUT_DIR / "temp"
FINAL_FILENAME = OUTPUT_DIR / "final_episode.mp3"

# Configurable via env
VOICE_A = "onyx"
VOICE_B = "nova"
MAX_CHARS_PER_REQ = int(os.getenv("MAX_CHARS_PER_REQ", "800"))
PAUSE_BETWEEN_TURNS_MS = int(os.getenv("PAUSE_BETWEEN_TURNS_MS", "1000"))  # ms

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", s).strip("_")

"""Naive chunk spilitting"""
def chunk_text(text: str, max_chars: int):
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        wlen = len(w) + (1 if cur else 0)
        if cur_len + wlen > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += wlen
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def generate_json_podcast(json_file):
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Expected {json_file} in current directory.")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    dialogue = data.get("dialogue", [])
    if not dialogue:
        raise ValueError("No 'dialogue' array found in podcast.json")

    # List of AudioSegment pieces to concat
    final_segments = []

    for i, turn in enumerate(dialogue, start=1):
        speaker = (turn.get("speaker") or "").strip()
        text = (turn.get("text") or "").strip()
        timestamp = (turn.get("timestamp") or "").strip().replace(":", "-") or f"{i:02d}"

        # determining voice
        if speaker.lower().startswith("host a") or speaker.lower().startswith("hosta") or speaker == "Alex":
            voice = VOICE_A
        elif speaker.lower().startswith("host b") or speaker.lower().startswith("hostb") or speaker == "Emma":
            voice = VOICE_B
        else:
            # alternate speaker based on index (even/odd)
            voice = VOICE_A if (i % 2 == 1) else VOICE_B

        if not text:
            continue

        chunks = chunk_text(text, MAX_CHARS_PER_REQ)

        for ci, chunk in enumerate(chunks, start=1):
            # temp file path per chunk
            safe_base = sanitize_filename(f"{i:02d}_{speaker or 'Speaker'}_{timestamp}_part{ci}")
            temp_file = TEMP_DIR / f"{safe_base}.mp3"
            try:
                _generate_azure_tts(chunk, str(temp_file), voice=voice)
            except Exception as e:
                print(f"[ERROR] TTS failed for turn {i} part {ci}: {e}")
                continue

            # load into pydub AudioSegment
            try:
                seg = AudioSegment.from_file(str(temp_file), format="mp3")
            except Exception as e:
                print(f"[ERROR] Failed to load generated mp3 {temp_file}: {e}")
                continue

            final_segments.append(seg)
            sleep(0.15)

        # add silence between turns 
        if PAUSE_BETWEEN_TURNS_MS > 0:
            final_segments.append(AudioSegment.silent(duration=PAUSE_BETWEEN_TURNS_MS))

    if not final_segments:
        raise RuntimeError("No audio segments were generated.")

    print(f"[INFO] Concatenating {len(final_segments)} segments into single mp3...")

    combined = final_segments[0]
    for seg in final_segments[1:]:
        combined += seg

    combined.export(str(FINAL_FILENAME), format="mp3", bitrate="192k")
    print(f"[DONE] Final episode written to: {FINAL_FILENAME}")

    # cleanup the temp files
    for f in TEMP_DIR.iterdir():
        try:
            f.unlink()
        except Exception:
            pass
    try:
        TEMP_DIR.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    generate_json_podcast("podcast.json")
