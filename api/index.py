# api/index.py

# All your existing imports for YouTubeProcessor
import yt_dlp
import os
from dotenv import load_dotenv
import subprocess
import uuid
import re

# Import FastAPI specific libraries
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # For CORS


ffmpeg_dir = os.path.abspath("bin")
os.environ["PATH"] = f"{ffmpeg_dir}:{os.environ.get('PATH', '')}"


# Load environment variables once at the top
load_dotenv()

class YouTubeProcessor:
    def __init__(self, openai_api_key=None):
        # Load environment variables (useful if API key is in .env)
        load_dotenv()
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client only if an API key is provided
        self.openai_client = None
        if self.openai_api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            print("Warning: OpenAI API key not provided. Transcription functionality will be unavailable.")

    def parse_subs(self, input_path, include_timestamps=False):
        """
        Parse a YouTube subtitle file and return either:
          • a list of plain-text lines, or
          • a list of (timestamp, text) tuples if include_timestamps=True.
        """
        entries = []
        timestamp = None
        buffer = []

        try:
            with open(input_path, encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    # blank line → flush buffer
                    if not line:
                        if buffer:
                            text = ' '.join(buffer)
                            if include_timestamps:
                                entries.append((timestamp, text))
                            else:
                                entries.append(text)
                            buffer = []
                            timestamp = None
                        continue

                    # timestamp lines look like "00:01:41.200 --> 00:01:43.510 …"
                    if '-->' in line:
                        # cut off any trailing "align:…" metadata
                        timestamp = re.sub(r'\s+align:.*', '', line)
                    else:
                        # strip out any <c>…</c> tags or other HTML
                        clean = re.sub(r'<.*?>', '', line)
                        buffer.append(clean)

            # in case file didn't end with a blank line
            if buffer:
                text = ' '.join(buffer)
                if include_timestamps:
                    entries.append((timestamp, text))
                else:
                    entries.append(text)
        except FileNotFoundError:
            print(f"Error: Subtitle file not found at {input_path}")
            return []
        except Exception as e:
            print(f"An error occurred while parsing subtitles: {e}")
            return []

        return entries

    def download_captions(self, url, lang='en', output_path='subtitles.%(ext)s'):
        """Download captions from a YouTube video."""
        ydl_opts = {
            'quiet': True,
            'writesubtitles': True,             # get human‐uploaded if available
            'writeautomaticsub': True,          # grab auto‐generated if not
            'subtitleslangs': [lang],
            'skip_download': True,
            'outtmpl': output_path,
            'no_warnings': True, # Suppress some yt-dlp warnings
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Captions for {lang} downloaded to {output_path.replace('%(ext)s', lang + '.vtt')}")
        except Exception as e:
            print(f"Error downloading captions for {url}: {e}")

    def get_info(self, url):
        """Get information about a YouTube video."""
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'no_warnings': True, # Suppress some yt-dlp warnings
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                video_info = {
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'duration': info.get('duration'),  # in seconds
                    'upload_date': info.get('upload_date'),
                    'channel': info.get('uploader'),
                    'captions': list(info.get('subtitles', {}).keys())  # list of available subtitle languages
                }
                return video_info
        except Exception as e:
            print(f"Error getting video info for {url}: {e}")
            return None

    def extract_frames(self, video_path, base_output_dir='frames', interval_seconds=5):
        """
        Extract frames from a video at a specified interval.
        Requires ffmpeg to be installed and in your system's PATH.
        """
        unique_id = str(uuid.uuid4())
        output_folder = os.path.join(base_output_dir + '-' + unique_id)
        
        try:
            os.makedirs(output_folder, exist_ok=True)

            subprocess.run([
                "ffmpeg",  
                '-i', video_path,
                '-vf', f'fps=1/{interval_seconds}',
                os.path.join(output_folder, 'frame_%04d.jpg')
            ], check=True, capture_output=True) # check=True will raise an error if ffmpeg fails
            
            print(f"Frames saved to: {output_folder}")
            return output_folder
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please ensure it's installed and in your system's PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error during frame extraction: {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during frame extraction: {e}")
            return None

    def transcribe_audio(self, file_path):
        """Transcribe audio from a file using OpenAI's Whisper API."""
        if not self.openai_client:
            print("Error: OpenAI client not initialized. Please provide an API key.")
            return None
        
        try:
            with open(file_path, 'rb') as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcription.text
        except FileNotFoundError:
            print(f"Error: Audio file not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error during audio transcription: {e}")
            return None

    def download_video(self, url, output_path='video.mp4'):
        """Download a YouTube video as MP4."""
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # Prioritize mp4
            'outtmpl': output_path,
            'merge_output_format': 'mp4',  # Ensures a single MP4 file
            'quiet': True,
            'no_warnings': True, # Suppress some yt-dlp warnings
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Video downloaded to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error downloading video from {url}: {e}")
            return None

    def download_audio(self, url, output_path='audio.mp3'):
        """Download audio from a YouTube video as MP3."""
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192', # 192kbps
            }],
            'quiet': True,
            'no_warnings': True, # Suppress some yt-dlp warnings
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Audio downloaded to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error downloading audio from {url}: {e}")
            return None

# --- End of YouTubeProcessor Class Definition ---

# Initialize FastAPI app
app = FastAPI()
security = HTTPBearer()

API_TOKEN = "1vsapitfws1"

# Initialize YouTubeProcessor
# IMPORTANT: For serverless functions, instantiate processor inside the route or pass app context
# For simplicity here, we'll instantiate it globally.
# In a real large-scale API, you might want to consider how state is managed
# across serverless invocations, but for YouTubeProcessor, it's mostly stateless.
processor = YouTubeProcessor()

# --- CORS Configuration ---
# This is CRUCIAL for your SvelteKit frontend to talk to your API.
# Replace 'http://localhost:5173' with your Vercel SvelteKit deployment URL (e.g., https://your-sveltekit-app.vercel.app)
# You can also add localhost for local development of your SvelteKit app.
origins = [
    "http://localhost:5173",  # Your SvelteKit dev server
    "http://localhost:4173",  # SvelteKit preview
    "https://your-sveltekit-app.vercel.app", # Replace with your actual SvelteKit Vercel URL
    # Add any other specific origins your frontend might be hosted on
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, PUT, DELETE, OPTIONS, etc.
    allow_headers=["*"], # Allows all headers
)
# --- End CORS Configuration ---


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")



# --- Define your API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the YouTube Processing API!"}

@app.post("/info")
async def get_video_info(url: str, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    """
    Get general information about a YouTube video.
    """
    info = processor.get_info(url)
    if not info:
        raise HTTPException(status_code=404, detail="Could not retrieve video info.")
    return info

@app.post("/captions")
async def get_video_captions(url: str, lang: str = 'en', token: HTTPAuthorizationCredentials = Depends(verify_token)):
    """
    Download and return captions for a YouTube video.
    Note: This will download a file to the serverless function's ephemeral storage.
    For large-scale, consider streaming or directly returning text without saving files.
    """
    temp_caption_path = f"/tmp/subtitles.{lang}.vtt" # Vercel uses /tmp for writable storage
    processor.download_captions(url, lang=lang, output_path=temp_caption_path)

    if os.path.exists(temp_caption_path):
        captions_data = processor.parse_subs(temp_caption_path, include_timestamps=True)
        os.remove(temp_caption_path) # Clean up the temporary file
        return {"captions": captions_data}
    else:
        raise HTTPException(status_code=404, detail=f"Captions for language '{lang}' not found or could not be downloaded.")

@app.post("/transcribe")
async def transcribe_youtube_audio(url: str, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    """
    Downloads audio from a YouTube video and transcribes it using OpenAI Whisper.
    Note: This is a heavy operation. Serverless functions have execution time limits.
    For long videos, consider a background job queue.
    """
    # Using /tmp for temporary storage in Vercel serverless environment
    temp_audio_path = f"/tmp/audio_{uuid.uuid4().hex}.mp3" 

    downloaded_path = processor.download_audio(url, output_path=temp_audio_path)
    if not downloaded_path:
        raise HTTPException(status_code=500, detail="Failed to download audio.")

    try:
        transcription = processor.transcribe_audio(downloaded_path)
        if not transcription:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio. OpenAI key might be missing or API error.")
        return {"transcription": transcription}
    finally:
        if os.path.exists(downloaded_path):
            os.remove(downloaded_path) # Clean up

# You can add more endpoints for other functions (e.g., download_video, extract_frames)
# Be mindful of the file sizes and execution times for serverless functions.
# Video downloads and frame extractions are typically resource-intensive and might
# hit serverless limits for longer videos. Consider background processing for these.