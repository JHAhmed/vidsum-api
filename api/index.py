# uvicorn api.index:app --reload --port 8000

import yt_dlp
import os
from dotenv import load_dotenv
import subprocess
import uuid
import json
import requests 
from pydub import AudioSegment
from pathlib import Path
import base64
import tempfile

from fastapi import FastAPI, HTTPException, status 
from fastapi.security import HTTPBearer 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional

load_dotenv()

# Ensure ffmpeg is in the PATH
ffmpeg_dir = os.path.abspath("bin")
os.environ["PATH"] = f"{ffmpeg_dir}:{os.environ.get('PATH', '')}"

class YouTubeProcessor:
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            print(
                "Warning: OpenAI API key not provided. Transcription functionality will be unavailable.")

    def get_info(self, url):
        """Get information about a YouTube video."""
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_info = {
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'duration': info.get('duration'),
                    'upload_date': info.get('upload_date'),
                    'channel': info.get('uploader'),
                    'captions': sorted(list(set(list(info.get('subtitles', {}).keys()) + list(info.get('automatic_captions', {}).keys()))))
                }
                return video_info
        except Exception as e:
            print(f"Error getting video info for {url}: {e}")
            return None


    def get_captions(self, url, lang='en'):
        """Extract and return captions from a YouTube video.
        Prioritizes:
        1. Manually uploaded subtitles for the exact language (lang).
        2. Manually uploaded subtitles for language variants (lang-*).
        3. Auto-generated subtitles for the exact language (lang).
        """
        ydl_opts = {
            'quiet': True,
            'writesubtitles': True,        # Attempt to retrieve info for manually uploaded subtitles
            'writeautomaticsub': True,    # Attempt to retrieve info for auto-generated subtitles
            'subtitleslangs': [f'{lang}.*', lang, f'{lang}-*'], # Prioritize exact lang, then lang-*, then lang
                                            # This applies to both manual and auto subs.
            'skip_download': True,        # Do not download the video itself
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # These dictionaries will be populated by yt-dlp based on subtitleslangs and availability
                available_manual_subs = info.get('subtitles') or {}
                available_auto_subs = info.get('automatic_captions') or {}

                caption_data_list = None # This will hold the list of caption formats (e.g., vtt, srv1)

                # Priority 1: Exact 'lang' in manually uploaded subtitles
                if lang in available_manual_subs:
                    caption_data_list = available_manual_subs[lang]
                
                # Priority 2: 'lang-*' in manually uploaded subtitles
                # If multiple lang-* variants exist (e.g., en-US, en-GB), pick the first one alphabetically.
                if not caption_data_list:
                    matching_lang_variants = []
                    for sub_code, data_list in available_manual_subs.items():
                        if sub_code.startswith(f"{lang}-"):
                            matching_lang_variants.append((sub_code, data_list))
                    
                    if matching_lang_variants:
                        # Sort by language code for deterministic selection (e.g., en-AU before en-CA)
                        matching_lang_variants.sort(key=lambda x: x[0])
                        caption_data_list = matching_lang_variants[0][1] # Get data_list of the first sorted variant
                
                # Priority 3: Exact 'lang' in auto-generated subtitles
                if not caption_data_list and lang in available_auto_subs:
                    caption_data_list = available_auto_subs[lang]

                if not caption_data_list:
                    # print(f"No captions found for '{lang}' or '{lang}-*' (manual or auto).") # Optional: more detailed message
                    return None  # No suitable captions found

                # Find the best available format (usually .vtt or .srv1) from the selected caption_data_list
                # The original code assumed caption_data_list[0] is what we want.
                # We should ensure it's not empty and has a 'url'.
                if not caption_data_list or not isinstance(caption_data_list, list) or not caption_data_list[0].get('url'):
                    # print(f"Caption data for selected language is malformed or missing URL.") # Optional
                    return None

                caption_url = caption_data_list[0]['url'] # Assuming the first format is acceptable
                response = requests.get(caption_url)
                response.raise_for_status() # Raise an exception for HTTP errors

                # The original code implies the response is JSON containing VTT events.
                # This structure { "events": [ { "segs": [ { "utf8": "text" } ] } ] } is specific to YouTube's
                # JSON representation of VTT files (timedtext format=jsonv3).
                data = json.loads(response.text)
                captions = []

                for entry in data.get('events', []): # Safely get 'events'
                    segs = entry.get('segs')
                    if segs and isinstance(segs, list):
                        for seg in segs:
                            text = seg.get('utf8')
                            if text:
                                captions.append(text.strip()) # .strip() to remove leading/trailing whitespace

                return '\n'.join(captions)

        except yt_dlp.utils.DownloadError as e:
            # Specific exception for yt-dlp errors, e.g., video not found, private video
            print(f"yt-dlp error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            # Handle errors from requests library (e.g., network issues, HTTP errors not caught by raise_for_status if it's missed)
            print(f"Error fetching caption file: {e}")
            return None
        except Exception as e:
            # General exception handler
            print(f"Error extracting captions: {e}")
            return None




    def chunk_audio(self, audio_path, chunk_length_seconds=1200):
        """Split an audio file into chunks."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} does not exist.")
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        output_dir = os.path.dirname(audio_path)
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]

        for i, start_ms in enumerate(range(0, len(audio), chunk_length_seconds * 1000)):
            chunk = audio[start_ms : start_ms + chunk_length_seconds * 1000]
            chunk_path = os.path.join(output_dir, f"{base_filename}_chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)
        return chunks

    def transcribe_audio(self, url, bitrate=128):
        """Download audio, transcribe it (chunking if necessary), and clean up temporary files."""
        if not self.openai_client:
            print("OpenAI client not initialized. Transcription unavailable.")
            raise RuntimeError("OpenAI client not initialized. Transcription unavailable.")

        with tempfile.TemporaryDirectory() as temp_dir:
            unique_id = str(uuid.uuid4())
            audio_download_base = os.path.join(temp_dir, f"audio_{unique_id}")
            downloaded_audio_path = audio_download_base + ".mp3" # Expected full path

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': audio_download_base, # yt-dlp adds extension
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': str(bitrate),
                }],
                'quiet': True,
                'no_warnings': True,
            }

            try:
                print(f"Downloading audio for transcription to temporary path starting with {audio_download_base}...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                print(f"Audio downloaded to {downloaded_audio_path}")

                if not os.path.exists(downloaded_audio_path):
                    # Sometimes yt-dlp might choose a different extension if mp3 conversion fails or format is weird
                    # Try to find the actual downloaded file if primary name doesn't exist
                    found_files = [f for f in os.listdir(temp_dir) if f.startswith(f"audio_{unique_id}")]
                    if not found_files:
                        raise FileNotFoundError(f"Downloaded audio file not found at expected path {downloaded_audio_path} or similar.")
                    downloaded_audio_path = os.path.join(temp_dir, found_files[0])
                    print(f"Adjusted audio path to actual downloaded file: {downloaded_audio_path}")


                if Path(downloaded_audio_path).stat().st_size < 25_000_000: # 25MB limit
                    with open(downloaded_audio_path, 'rb') as f:
                        resp = self.openai_client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe", # Assuming this is a valid model for your OpenAI setup
                            file=f,
                            response_format="text",
                            prompt="The following is audio from a YouTube video. Transcribe it accurately, including any spoken names or titles, and ensure proper punctuation.",
                        )
                    print("Transcription successful.")
                    return resp # This is a string
                else:
                    print("Audio file is too large for direct transcription. Splitting into chunks...")
                    # chunk_audio creates files like audio_unique_id_chunk_N.mp3 in temp_dir
                    chunk_paths = self.chunk_audio(downloaded_audio_path)
                    transcriptions = []
                    for chunk_path in chunk_paths:
                        with open(chunk_path, 'rb') as f:
                            resp = self.openai_client.audio.transcriptions.create(
                                model="gpt-4o-mini-transcribe",
                                file=f,
                                response_format="text",
                                prompt="The following is audio from a YouTube video. Transcribe it accurately, including any spoken names or titles, and ensure proper punctuation.",
                            )
                        transcriptions.append(str(resp)) # Ensure it's a string
                    print("Chunked transcription successful.")
                    return "\n".join(transcriptions)

            except Exception as e:
                print(f"Error during audio download or transcription for {url}: {e}")
                # Re-raise the exception to be caught by the FastAPI endpoint
                raise
            # Temporary directory and its contents (original audio, chunks) are automatically cleaned up

    def download_video(self, url, output_path='video.mp4'):
        """Download a YouTube video as MP4 to the specified output_path."""
        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',
            'outtmpl': output_path, # Caller provides full path including filename and extension
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Video downloaded to {output_path}")
            return output_path # Return the path if successful
        except Exception as e:
            print(f"Error downloading video from {url} to {output_path}: {e}")
            return None

    def extract_frames(self, video_path, base_output_dir='frames', interval_seconds=60):
        """Extract frames from a video at specified intervals."""
        if not os.path.exists(video_path):
            print(f"Video file not found at {video_path}")
            return None
            
        unique_id = str(uuid.uuid4())
        # Creates a unique directory like 'frames-uuid' or 'custom_base_dir-uuid'
        output_folder = os.path.join(os.getcwd(), f"{base_output_dir}-{unique_id}")

        try:
            os.makedirs(output_folder, exist_ok=True)
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-vf", f"fps=1/{interval_seconds},scale=iw/2:ih/2", # Scales to half size
                os.path.join(output_folder, "frame_%04d.jpg")
            ], check=True, capture_output=True, text=True) # Added text=True for stderr

            
            print(f"Frames saved to: {output_folder}")
            return output_folder
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please ensure it's installed and in your system's PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error during frame extraction: {e.stderr}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during frame extraction: {e}")
            return None


    def delete_frames(self, frames_dir, video_path=None):
        """Delete all extracted frames from the specified directory and optionally the video file."""
        try:
            if os.path.exists(frames_dir):
                for frame_filename in os.listdir(frames_dir):
                    os.remove(os.path.join(frames_dir, frame_filename))
                os.rmdir(frames_dir)
                print(f"Deleted frames and directory: {frames_dir}")
            else:
                print(f"Frames directory not found, skipping deletion: {frames_dir}")

            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted video file: {video_path}")
        except Exception as e:
            print(f"Error deleting frames or video: {e}")

    def convert_frames_base64(self, folder_path):
        """Read images from a folder, convert to Base64, and return a list of strings."""
        base64_frames = []
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist for Base64 extraction.")
            return None # Or an empty list, depending on desired behavior

        try:
            for filename in sorted(os.listdir(folder_path)): # Sorted to maintain order
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with open(file_path, 'rb') as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        base64_frames.append(encoded_string)
            print(f"Successfully converted {len(base64_frames)} frames from {folder_path} to Base64.")
            return base64_frames
        except Exception as e:
            print(f"An error occurred while reading images for Base64 conversion: {e}")
            return None


app = FastAPI()
# security = HTTPBearer() # Kept, but not actively used for route protection yet

processor = YouTubeProcessor()

origins = [
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:4173",
    "https://your-sveltekit-app.vercel.app", # Replace with your actual URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class YouTubeRequest(BaseModel):
    url: str 
    interval: Optional[int] = 600
    lang: Optional[str] = None

@app.get("/")
async def read_root():
    return {"message": "Welcome to the YouTube Processing API!"}


@app.post("/info")
async def get_video_info(request: YouTubeRequest):
    """Get general information about a YouTube video."""
    info = processor.get_info(request.url)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Could not retrieve video info.")
    return info


@app.post("/captions")
async def get_video_captions(request: YouTubeRequest):
    """Download and return captions for a YouTube video."""
    # Use the lang from the request, defaulting to 'en' if not provided or empty
    selected_lang = request.lang if request.lang and request.lang.strip() else 'en'
    
    captions = processor.get_captions(request.url, lang=selected_lang)
    if captions is not None: # Check for None explicitly, as empty string "" is a valid (empty) caption
        return {"output": captions}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Captions for language '{selected_lang}' not found or could not be downloaded."
        )


@app.post("/transcribe")
async def transcribe_youtube_audio(request: YouTubeRequest):
    """Downloads audio from YouTube and transcribes it."""
    try:
        transcription = processor.transcribe_audio(request.url)
        if transcription is None : # Should not happen if transcribe_audio raises errors
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Failed to transcribe audio. Unknown error in processing."
            )
        return {"output": transcription}
    except RuntimeError as e: # Catch specific error from processor (e.g. OpenAI key missing)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except FileNotFoundError as e: # Catch if audio download fails critically
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Audio processing error: {e}")
    except Exception as e: # Catch any other unexpected errors from transcribe_audio
        print(f"Unexpected error in /transcribe endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during transcription.")


@app.post("/frames")
async def extract_video_frames(request: YouTubeRequest):
    """Download video, extract frames at intervals, return as Base64, and clean up."""
    if not request.url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You must provide a YouTube video URL.")

    frames_dir_to_delete_finally = None

    try:
        with tempfile.TemporaryDirectory() as temp_video_download_dir:
            unique_video_filename = os.path.join(temp_video_download_dir, f"video_{uuid.uuid4()}.mp4")
            
            downloaded_video_path = processor.download_video(request.url, output_path=unique_video_filename)
            if not downloaded_video_path:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to download video.")
                        
            frames_dir = processor.extract_frames(downloaded_video_path, interval_seconds=request.interval)
            if not frames_dir:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to extract frames from video.")
            frames_dir_to_delete_finally = frames_dir # Mark for deletion

            frames_list = processor.convert_frames_base64(frames_dir)
            if frames_list is None: # If convert_frames_base64 had an issue
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to convert frames to Base64.")

            return {"output": frames_list}
            
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Unexpected error in /frames endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during frame extraction: {e}")
    finally:
        # Clean up the 'frames-uuid' directory.
        # The downloaded video in temp_video_download_dir is cleaned up automatically by TemporaryDirectory.
        if frames_dir_to_delete_finally:
            processor.delete_frames(frames_dir_to_delete_finally, video_path=None)


class DeleteRequest(BaseModel): # Added for the /delete endpoint body
    frames_dir: str

@app.post("/delete")
async def delete_frames_endpoint(request: DeleteRequest): # Changed to use Pydantic model for JSON body
    """
    Delete all extracted frames from a specified directory.
    Expects a JSON body like: {"frames_dir": "path_to_frames_directory"}
    """
    if not request.frames_dir: # Should be caught by Pydantic if field is required
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="You must provide a 'frames_dir' in the request body."
        )
    
    # Basic check to prevent accidental deletion of common system paths - enhance as needed
    if not request.frames_dir.startswith("frames-"): # Assuming your frame dirs always start with "frames-"
        print(f"Attempt to delete potentially unsafe directory blocked: {request.frames_dir}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid directory specified for deletion. Path must be a recognized frames directory."
        )

    if not os.path.exists(request.frames_dir) or not os.path.isdir(request.frames_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {request.frames_dir}"
        )
        
    processor.delete_frames(request.frames_dir) # video_path is None by default
    return {"message": f"Attempted deletion of frames in {request.frames_dir}."}