import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from time import sleep, time
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

load_dotenv()

def combine_audio_files_python(audio_files, output_path):
    """Combine MP3 files using Python - convert to WAV first, then combine."""
    try:
        import wave
        
        # For MP3 files, we need to convert them first or use a different approach
        # Let's try a simple binary concatenation for MP3 files
        print("Attempting Python-based audio combination...")
        
        with open(output_path, 'wb') as outfile:
            for i, audio_file in enumerate(audio_files):
                if os.path.exists(audio_file):
                    with open(audio_file, 'rb') as infile:
                        if i == 0:
                            # Copy the first file completely (including headers)
                            outfile.write(infile.read())
                        else:
                            # For subsequent files, skip some header bytes and append
                            # This is a simple approach that may not work perfectly for all MP3s
                            data = infile.read()
                            # Skip potential ID3 headers (first 128 bytes) for subsequent files
                            if len(data) > 128:
                                outfile.write(data[128:])
                            else:
                                outfile.write(data)
        
        print(f"Python combination completed: {output_path}")
        return True
        
    except Exception as e:
        print(f"Python combination failed: {e}")
        return False

def combine_audio_files_with_ffmpeg(audio_files, output_path):
    """Combine multiple audio files using ffmpeg."""
    try:
        # Create a temporary file list for ffmpeg
        temp_list_file = "temp_audio_list.txt"
        with open(temp_list_file, 'w') as f:
            for audio_file in audio_files:
                if os.path.exists(audio_file):
                    # Convert to absolute path and escape backslashes
                    abs_path = os.path.abspath(audio_file).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")
        
        # Use ffmpeg to concatenate the files
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-c', 'copy',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)
        
        if result.returncode == 0:
            print(f"Successfully combined audio files into: {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("FFmpeg not found in system PATH")
        return False
    except Exception as e:
        print(f"Error combining audio files: {e}")
        return False

def combine_audio_files_simple(audio_files, output_path):
    """Simple combination by copying first file as fallback."""
    try:
        import shutil
        if audio_files and os.path.exists(audio_files[0]):
            shutil.copy2(audio_files[0], output_path)
            print(f"Created simple combined file: {output_path}")
            return True
        return False
    except Exception as e:
        print(f"Error in simple combine: {e}")
        return False

def _generate_azure_tts_simple(text, output_file, voice=None):
    """Generate audio using Azure OpenAI TTS - simple version without pydub."""
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

def _generate_tts_thread_worker(turn_data):
    """Worker function for threading - generates a single TTS audio file."""
    turn_index, turn, output_dir = turn_data
    
    speaker = (turn.get("speaker") or "").strip()
    text = (turn.get("text") or "").strip()
    
    # Determine voice
    if speaker.lower().startswith("host a") or speaker.lower().startswith("hosta") or speaker == "Alex":
        voice = "onyx"
    elif speaker.lower().startswith("host b") or speaker.lower().startswith("hostb") or speaker == "Emma":
        voice = "nova"
    else:
        # Unknown speaker -> alternate based on index
        voice = "onyx" if (turn_index % 2 == 1) else "nova"

    if not text:
        return None

    # Create audio file for this turn
    audio_filename = f"turn_{turn_index:02d}_{speaker.replace(' ', '_').replace('/', '_')}.mp3"
    audio_path = output_dir / audio_filename
    
    try:
        print(f"[Thread] Generating audio for turn {turn_index}: {speaker}")
        _generate_azure_tts_simple(text, str(audio_path), voice=voice)
        print(f"[Thread] Completed turn {turn_index}: {audio_filename}")
        return {
            'index': turn_index,
            'path': str(audio_path),
            'speaker': speaker,
            'success': True
        }
        
    except Exception as e:
        print(f"[Thread ERROR] TTS failed for turn {turn_index}: {e}")
        return {
            'index': turn_index,
            'path': None,
            'speaker': speaker,
            'success': False,
            'error': str(e)
        }

def generate_simple_podcast(json_file, max_workers=4, progress_callback=None):
    """Generate podcast audio using multithreading - creates multiple audio files concurrently.
    
    Args:
        json_file: Path to the podcast JSON file
        max_workers: Number of concurrent workers for TTS generation (default: 4)
        progress_callback: Optional callback function to report progress (receives percentage)
    """
    start_time = time()
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Expected {json_file} not found.")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    dialogue = data.get("dialogue", [])
    if not dialogue:
        raise ValueError("No 'dialogue' array found in podcast.json")

    # Create output directory - ensure it's in outputs/podcast_voice
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)  # Go up from src to backend
    project_root = os.path.dirname(backend_dir)  # Go up from backend to project root
    output_dir = Path(os.path.join(project_root, "outputs", "podcast_voice"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating podcast audio in: {output_dir}")
    print(f"Processing {len(dialogue)} dialogue turns with {max_workers} concurrent workers...")
    
    if progress_callback:
        progress_callback(5)  # Starting generation
    
    # Prepare data for threading
    thread_data = []
    for i, turn in enumerate(dialogue, start=1):
        thread_data.append((i, turn, output_dir))
    
    # Generate audio files concurrently using ThreadPoolExecutor
    audio_results = []
    successful_files = []
    failed_turns = []
    completed_count = 0
    
    generation_start_time = time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        print(f"Submitting {len(thread_data)} TTS generation tasks...")
        future_to_turn = {executor.submit(_generate_tts_thread_worker, data): data[0] for data in thread_data}
        
        if progress_callback:
            progress_callback(10)  # Tasks submitted
        
        # Collect results as they complete
        for future in as_completed(future_to_turn):
            turn_index = future_to_turn[future]
            try:
                result = future.result()
                audio_results.append(result)
                completed_count += 1
                
                # Calculate progress (10% start + 70% for generation + 20% for combining)
                progress = 10 + (completed_count / len(thread_data)) * 70
                if progress_callback:
                    progress_callback(int(progress))
                
                if result and result['success']:
                    successful_files.append(result)
                    print(f"✓ Completed turn {result['index']}: {result['speaker']} ({completed_count}/{len(thread_data)})")
                else:
                    failed_turns.append(result)
                    print(f"✗ Failed turn {result['index']}: {result.get('error', 'Unknown error')} ({completed_count}/{len(thread_data)})")
                    
            except Exception as e:
                failed_turns.append({'index': turn_index, 'error': str(e)})
                completed_count += 1
                progress = 10 + (completed_count / len(thread_data)) * 70
                if progress_callback:
                    progress_callback(int(progress))
                print(f"✗ Exception in turn {turn_index}: {e} ({completed_count}/{len(thread_data)})")
    
    generation_end_time = time()
    generation_duration = generation_end_time - generation_start_time
    
    if progress_callback:
        progress_callback(80)  # Generation complete, starting combination
    
    # Sort successful files by index to maintain dialogue order
    successful_files.sort(key=lambda x: x['index'])
    audio_files = [result['path'] for result in successful_files if result['path']]
    
    print(f"\nGeneration Summary:")
    print(f"  Total turns: {len(dialogue)}")
    print(f"  Successful: {len(successful_files)}")
    print(f"  Failed: {len(failed_turns)}")
    print(f"  Generated audio files: {len(audio_files)}")
    print(f"  Generation time: {generation_duration:.2f} seconds")
    print(f"  Average time per turn: {generation_duration / len(dialogue):.2f} seconds")
    print(f"  Concurrent workers: {max_workers}")
    
    if failed_turns:
        print(f"  Failed turns: {[f['index'] for f in failed_turns]}")

    # Create a detailed report file
    report_path = output_dir / "generation_report.json"
    total_duration = time() - start_time
    report_data = {
        "total_turns": len(dialogue),
        "successful_turns": len(successful_files),
        "failed_turns": len(failed_turns),
        "success_rate": len(successful_files) / len(dialogue) * 100 if dialogue else 0,
        "audio_files": [result['path'] for result in successful_files],
        "failed_turn_details": failed_turns,
        "generation_timestamp": str(Path().absolute()),
        "performance_metrics": {
            "total_duration_seconds": round(total_duration, 2),
            "generation_duration_seconds": round(generation_duration, 2),
            "average_time_per_turn_seconds": round(generation_duration / len(dialogue), 2) if dialogue else 0,
            "concurrent_workers": max_workers,
            "turns_per_second": round(len(dialogue) / generation_duration, 2) if generation_duration > 0 else 0
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    print(f"Generation report saved to: {report_path}")

    # Create a simple text file listing all generated audio files
    playlist_path = output_dir / "podcast_playlist.txt"
    with open(playlist_path, 'w', encoding='utf-8') as f:
        f.write("# Podcast Audio Files\n")
        f.write(f"# Generated {len(audio_files)} audio segments\n")
        f.write(f"# Success rate: {len(successful_files)}/{len(dialogue)} ({len(successful_files) / len(dialogue) * 100:.1f}%)\n\n")
        for result in successful_files:
            f.write(f"{result['path']} # Turn {result['index']}: {result['speaker']}\n")
    
    print(f"Playlist saved to: {playlist_path}")
    
    # Combine all audio files into a single final episode (synchronously)
    if audio_files:
        final_episode_path = output_dir / "final_episode.mp3"
        
        print(f"\nCombining {len(audio_files)} audio files synchronously...")
        # Try to combine using ffmpeg first, fallback to simple copy
        combined_successfully = combine_audio_files_with_ffmpeg(audio_files, final_episode_path)
        
        if not combined_successfully:
            print("FFmpeg combination failed, trying Python-based combination...")
            combined_successfully = combine_audio_files_python(audio_files, final_episode_path)
        
        if not combined_successfully:
            print("Python combination failed, using simple fallback...")
            combined_successfully = combine_audio_files_simple(audio_files, final_episode_path)
        
        if combined_successfully and os.path.exists(final_episode_path):
            print(f"✓ Created final combined episode: {final_episode_path}")
            if progress_callback:
                progress_callback(100)  # Complete
            return str(final_episode_path)
        else:
            print("✗ Failed to create combined audio file")
            if progress_callback:
                progress_callback(95)  # Almost complete but with issues
            # Return the first audio file as fallback
            return audio_files[0] if audio_files else None
    else:
        print("No audio files were generated successfully")
        if progress_callback:
            progress_callback(100)  # Complete but failed
        return None

def generate_simple_podcast_sequential(json_file):
    """Generate podcast audio using sequential approach - original implementation for comparison."""
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Expected {json_file} not found.")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    dialogue = data.get("dialogue", [])
    if not dialogue:
        raise ValueError("No 'dialogue' array found in podcast.json")

    # Create output directory - ensure it's in outputs/podcast_voice
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)  # Go up from src to backend
    project_root = os.path.dirname(backend_dir)  # Go up from backend to project root
    output_dir = Path(os.path.join(project_root, "outputs", "podcast_voice"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating podcast audio in: {output_dir}")
    
    # Generate individual audio files for each dialogue turn
    audio_files = []
    
    for i, turn in enumerate(dialogue, start=1):
        speaker = (turn.get("speaker") or "").strip()
        text = (turn.get("text") or "").strip()
        
        # Determine voice
        if speaker.lower().startswith("host a") or speaker.lower().startswith("hosta") or speaker == "Alex":
            voice = "onyx"
        elif speaker.lower().startswith("host b") or speaker.lower().startswith("hostb") or speaker == "Emma":
            voice = "nova"
        else:
            # Unknown speaker -> alternate based on index
            voice = "onyx" if (i % 2 == 1) else "nova"

        if not text:
            continue

        # Create audio file for this turn
        audio_filename = f"turn_{i:02d}_{speaker.replace(' ', '_')}.mp3"
        audio_path = output_dir / audio_filename
        
        try:
            print(f"Generating audio for turn {i}: {speaker}")
            _generate_azure_tts_simple(text, str(audio_path), voice=voice)
            audio_files.append(str(audio_path))
            
            # Small delay between requests
            sleep(0.5)
            
        except Exception as e:
            print(f"[ERROR] TTS failed for turn {i}: {e}")
            continue

    # Create a simple text file listing all generated audio files
    playlist_path = output_dir / "podcast_playlist.txt"
    with open(playlist_path, 'w', encoding='utf-8') as f:
        f.write("# Podcast Audio Files\n")
        f.write(f"# Generated {len(audio_files)} audio segments\n\n")
        for audio_file in audio_files:
            f.write(f"{audio_file}\n")
    
    print(f"Generated {len(audio_files)} audio files in {output_dir}")
    print(f"Playlist saved to: {playlist_path}")
    
    # Combine all audio files into a single final episode
    if audio_files:
        final_episode_path = output_dir / "final_episode.mp3"
        
        # Try to combine using ffmpeg first, fallback to simple copy
        print("Attempting to combine audio files...")
        combined_successfully = combine_audio_files_with_ffmpeg(audio_files, final_episode_path)
        
        if not combined_successfully:
            print("FFmpeg combination failed, trying Python-based combination...")
            combined_successfully = combine_audio_files_python(audio_files, final_episode_path)
        
        if not combined_successfully:
            print("Python combination failed, using simple fallback...")
            combined_successfully = combine_audio_files_simple(audio_files, final_episode_path)
        
        if combined_successfully and os.path.exists(final_episode_path):
            print(f"Created final combined episode: {final_episode_path}")
            return str(final_episode_path)
        else:
            print("Failed to create combined audio file")
            # Return the first audio file as fallback
            return audio_files[0] if audio_files else None
    
    return None

def compare_performance(json_file, max_workers_list=[2, 4, 6]):
    """Compare performance of different worker counts for podcast generation."""
    print("=== Performance Comparison ===")
    
    for workers in max_workers_list:
        print(f"\nTesting with {workers} workers...")
        try:
            start_time = time()
            result = generate_simple_podcast(json_file, max_workers=workers)
            duration = time() - start_time
            
            # Read the generation report for detailed metrics
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(current_dir)
            project_root = os.path.dirname(backend_dir)
            report_path = Path(os.path.join(project_root, "outputs", "podcast_voice", "generation_report.json"))
            
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    metrics = report.get('performance_metrics', {})
                    print(f"  Workers: {workers}")
                    print(f"  Total time: {duration:.2f}s")
                    print(f"  Generation time: {metrics.get('generation_duration_seconds', 0):.2f}s")
                    print(f"  Success rate: {report.get('success_rate', 0):.1f}%")
                    print(f"  Turns per second: {metrics.get('turns_per_second', 0):.2f}")
            
        except Exception as e:
            print(f"  Error with {workers} workers: {e}")

if __name__ == "__main__":
    # Use multithreaded version by default with 4 concurrent workers
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_performance("outputs/podcast_output/podcast.json")
    else:
        generate_simple_podcast("outputs/podcast_output/podcast.json", max_workers=4)
