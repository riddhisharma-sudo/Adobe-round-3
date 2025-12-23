import os
import subprocess
import requests
from pathlib import Path
# from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv()

# Python libraries to be installed: requests, google-cloud-texttospeech, pydub(optional)

"""
Unified Text-to-Speech Interface with Multi-Provider Support

This module provides a unified interface for text-to-speech using various providers
including Azure OpenAI TTS, Google Cloud Text-to-Speech, and local espeak-ng.

SETUP:
Users are expected to set appropriate environment variables for their chosen TTS provider
before calling the generate_audio function.

Environment Variables:

TTS_PROVIDER (default: "local")
    - "azure": Azure OpenAI TTS
    - "gcp": Google Cloud Text-to-Speech
    - "local": Local TTS implementation (default, uses espeak-ng)

For Azure TTS:
    AZURE_TTS_KEY: Your Azure OpenAI API key
    AZURE_TTS_ENDPOINT: Azure OpenAI endpoint URL
    AZURE_TTS_VOICE (default: "alloy"): Voice to use (alloy, echo, fable, onyx, nova, shimmer)
    AZURE_TTS_DEPLOYMENT (default: "tts"): Deployment name
    AZURE_TTS_API_VERSION (default: "2025-03-01-preview"): API version

For Google Cloud TTS:
    GOOGLE_API_KEY: Your Google API key (recommended)
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON file (alternative)
    GCP_TTS_VOICE (default: "en-US-Neural2-F"): Voice to use
    GCP_TTS_LANGUAGE (default: "en-US"): Language code

For Local TTS (espeak-ng):
    ESPEAK_VOICE (default: "en"): Voice to use
    ESPEAK_SPEED (default: "150"): Speech rate (words per minute)
    Note: Participants can modify the local provider implementation to use any local TTS solution
    
    Installation:
        # Ubuntu/Debian
        sudo apt-get install espeak-ng
        
        # macOS
        brew install espeak
        
        # CentOS/RHEL
        sudo yum install espeak-ng

Usage:
    from tts import generate_audio
    
    # Basic usage with default provider (local)
    generate_audio("Hello, world!", "output.wav")
    
    # With specific provider
    generate_audio("Hello, world!", "output.mp3", provider="azure")
    
    # With custom voice
    generate_audio("Hello, world!", "output.wav", voice="alloy")
    

"""

def generate_audio(text, output_file, provider=None, voice=None):
    """
    Generate audio from text using the specified TTS provider.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    provider = provider or os.getenv("TTS_PROVIDER", "local").lower()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if provider == "azure":
        return _generate_azure_tts(text, output_file, voice)
    elif provider == "gcp": 
        return _generate_gcp_tts(text, output_file, voice)
    elif provider == "local":
        return _generate_local_tts(text, output_file, voice)
    else:
        raise ValueError(f"Unsupported TTS_PROVIDER: {provider}")

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
            timeout=30
        )
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"Azure OpenAI TTS audio saved to: {output_file}")
        return output_file
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Azure OpenAI TTS failed: {e}")

def _generate_gcp_tts(text, output_file, voice=None):
    """Generate audio using Google Cloud Text-to-Speech."""
    api_key = os.getenv("GOOGLE_API_KEY")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_voice = voice or os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")
    language = os.getenv("GCP_TTS_LANGUAGE", "en-US")
    
    if not api_key and not credentials_path:
        raise ValueError("Either GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set for Google Cloud TTS")
    
    try:
        # Use API key if available, otherwise use service account credentials
        if api_key:
            # For API key authentication, we need to use the REST API directly
            import requests
            
            url = "https://texttospeech.googleapis.com/v1/text:synthesize"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": {"text": text},
                "voice": {
                    "languageCode": language,
                    "name": gcp_voice
                },
                "audioConfig": {
                    "audioEncoding": "MP3"
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # Decode the base64 audio content
            import base64
            audio_content = base64.b64decode(response.json()["audioContent"])
            
            with open(output_file, "wb") as f:
                f.write(audio_content)
                
        else:
            # Use service account credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            client = texttospeech.TextToSpeechClient()
            
            input_text = texttospeech.SynthesisInput(text=text)
            
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language,
                name=gcp_voice
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=input_text,
                voice=voice_params,
                audio_config=audio_config
            )
            
            with open(output_file, "wb") as f:
                f.write(response.audio_content)
        
        print(f"Google Cloud TTS audio saved to: {output_file}")
        return output_file
        
    except Exception as e:
        raise RuntimeError(f"Google Cloud TTS failed: {e}")

def _generate_local_tts(text, output_file, voice=None):
    """Generate audio using local TTS implementation (espeak-ng command line).
    
    Note: Participants can modify this function to use any local TTS solution
    such as pyttsx3, say, or other local TTS tools.
    """
    # TODO: Participants can modify this implementation to use any local TTS solution
    # Examples: pyttsx3, say (macOS), gtts, or any other local TTS tool
    
    espeak_voice = voice or os.getenv("ESPEAK_VOICE", "en")
    espeak_speed = os.getenv("ESPEAK_SPEED", "150")
    
    # Create temporary WAV file for espeak-ng
    temp_wav_file = output_file.replace('.mp3', '.wav')
    
    try:
        # Use espeak-ng command line tool
        cmd = [
            'espeak-ng',
            '-v', espeak_voice,
            '-s', str(espeak_speed),
            '-w', temp_wav_file,
            text
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {result.stderr}")
        
        # Check if temporary WAV file was created
        if not os.path.exists(temp_wav_file):
            raise RuntimeError(f"espeak-ng did not create output file {temp_wav_file}")
        
        # Convert WAV to MP3 if output file is MP3
        if output_file.endswith('.mp3'):
            try:
                from pydub import AudioSegment
                
                # Load WAV file
                audio = AudioSegment.from_wav(temp_wav_file)
                
                # Export as MP3
                audio.export(output_file, format="mp3")
                
                # Remove temporary WAV file
                os.remove(temp_wav_file)
                
                print(f"Local TTS audio saved to: {output_file}")
                return output_file
                
            except ImportError:
                raise RuntimeError("pydub library not installed. Please install it with: pip install pydub")
            except Exception as e:
                raise RuntimeError(f"Failed to convert WAV to MP3: {e}")
        else:
            # If output is WAV, just rename the file
            os.rename(temp_wav_file, output_file)
            print(f"Local TTS audio saved to: {output_file}")
            return output_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("espeak-ng synthesis timed out")
    except FileNotFoundError:
        raise RuntimeError("espeak-ng is not installed. Please install it first:\nUbuntu/Debian: sudo apt-get install espeak-ng\nmacOS: brew install espeak\nCentOS/RHEL: sudo yum install espeak-ng")
    except Exception as e:
        raise RuntimeError(f"Local TTS synthesis error: {str(e)}")

def test_tts_providers():
    """Test all available TTS providers."""
    test_text = "Hello, this is a test of text to speech functionality!"
    test_file = "test_output"
    
    providers = ["local", "azure", "gcp"]
    
    for provider in providers:
        try:
            print(f"\nTesting {provider.upper()} TTS...")
            output_file = generate_audio(test_text, f"{test_file}_{provider}", provider=provider)
            print(f"✅{provider.upper()} TTS test successful: {output_file}")
        except Exception as e:
            print(f" {provider.upper()} TTS test failed: {e}")

def list_available_providers():
    """List available TTS providers and their status."""
    providers = {
        "local": "Local TTS implementation (uses espeak-ng, can be modified)",
        "azure": "Azure OpenAI TTS (requires AZURE_TTS_KEY and AZURE_TTS_ENDPOINT)",
        "gcp": "Google Cloud Text-to-Speech (requires GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS)"
    }
    
    print("Available TTS Providers:")
    for provider, description in providers.items():
        status = "✅ Available" if _test_provider(provider) else "❌ Not available"
        print(f"  {provider}: {description} - {status}")

def _test_provider(provider):
    """Test if a specific provider is available."""
    try:
        if provider == "local":
            result = subprocess.run(['espeak-ng', '--version'], capture_output=True, timeout=5)
            return result.returncode == 0
        elif provider == "azure":
            return bool(os.getenv("AZURE_TTS_KEY") and os.getenv("AZURE_TTS_ENDPOINT"))
        elif provider == "gcp":
            return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        return False
    except:
        return False

if __name__ == "__main__":
    # Get the provider from environment variable
    provider = "azure"
    
    print(f"Testing TTS provider: {provider.upper()}")
    print("="*50)
    
    # Test the specified provider
    test_text = "Hello, this is a test of text to speech functionality!"
    test_file = f"test_output_{provider}.mp3"
    
    try:
        output_file = generate_audio(test_text, test_file, provider=provider)
        print(f"{provider.upper()} TTS test successful: {output_file}")
    except Exception as e:
        print(f"{provider.upper()} TTS test failed: {e}")
        print("\nAvailable providers and their status:")
        list_available_providers() 
