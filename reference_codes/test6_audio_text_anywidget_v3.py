import streamlit as st
import numpy as np
import base64
import json
import os
import tempfile
from streamlit_anywidget import anywidget
from anywidget import AnyWidget
import traitlets
from io import BytesIO
import httpx
import logging
import time
import uuid
from typing import Optional, Dict, List, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if ElevenLabs key is available
ELEVEN_LABS_API_KEY = st.secrets.get("ELEVEN_LABS_API_KEY", "")
ELEVENLABS_AVAILABLE = bool(ELEVEN_LABS_API_KEY)

class AudioTranscriptWidget(AnyWidget):
    """
    A custom widget that synchronizes audio playback with transcript text.
    Uses anywidget to enable JavaScript-Python communication.
    """
    # Bi-directional data with synchronization
    audio_data_base64 = traitlets.Unicode(allow_none=True).tag(sync=True)
    transcript = traitlets.Dict().tag(sync=True)
    current_time = traitlets.Float(0.0).tag(sync=True)
    is_playing = traitlets.Bool(False).tag(sync=True)
    
    # JavaScript for the widget frontend
    _esm = """
    function render({ model, el }) {
        // Create container elements
        const container = document.createElement("div");
        container.className = "audio-transcript-container";
        
        // Audio player
        const audioContainer = document.createElement("div");
        audioContainer.className = "audio-player-container";
        
        const audioEl = document.createElement("audio");
        audioEl.controls = true;
        audioEl.id = "audio-player";
        
        // Play/pause button for custom control
        const controlsContainer = document.createElement("div");
        controlsContainer.className = "controls-container";
        
        const playBtn = document.createElement("button");
        playBtn.textContent = "Play";
        playBtn.className = "control-button play-button";
        
        const pauseBtn = document.createElement("button");
        pauseBtn.textContent = "Pause";
        pauseBtn.className = "control-button pause-button";
        pauseBtn.style.display = "none";
        
        const timeDisplay = document.createElement("div");
        timeDisplay.className = "time-display";
        timeDisplay.textContent = "0:00 / 0:00";
        
        // Progress bar
        const progressContainer = document.createElement("div");
        progressContainer.className = "progress-container";
        
        const progressBar = document.createElement("div");
        progressBar.className = "progress-bar";
        
        const progressIndicator = document.createElement("div");
        progressIndicator.className = "progress-indicator";
        progressBar.appendChild(progressIndicator);
        
        // Transcript container
        const transcriptContainer = document.createElement("div");
        transcriptContainer.className = "transcript-container";
        
        // Add elements to the DOM
        controlsContainer.appendChild(playBtn);
        controlsContainer.appendChild(pauseBtn);
        controlsContainer.appendChild(timeDisplay);
        
        progressContainer.appendChild(progressBar);
        
        audioContainer.appendChild(audioEl);
        audioContainer.appendChild(controlsContainer);
        audioContainer.appendChild(progressContainer);
        
        container.appendChild(audioContainer);
        container.appendChild(transcriptContainer);
        
        el.appendChild(container);
        
        // Function to format time in MM:SS format
        function formatTime(seconds) {
            const minutes = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
        }
        
        // Function to update UI when audio is loaded
        function handleAudioLoaded() {
            const duration = audioEl.duration;
            timeDisplay.textContent = `0:00 / ${formatTime(duration)}`;
        }
        
        // Function to update transcript highlighting based on current time
        function updateTranscriptHighlighting(currentTime) {
            const transcript = model.get("transcript") || {};
            const segments = transcript.segments || [];
            
            // Clear existing transcript if needed
            if (transcriptContainer.innerHTML === "" && segments.length > 0) {
                segments.forEach((segment, index) => {
                    const segmentEl = document.createElement("span");
                    segmentEl.setAttribute("data-start", segment.start);
                    segmentEl.setAttribute("data-end", segment.end);
                    segmentEl.setAttribute("data-index", index);
                    
                    // Add speaker information if available
                    if (segment.speaker_id) {
                        segmentEl.setAttribute("data-speaker", segment.speaker_id);
                        segmentEl.classList.add("speaker-" + segment.speaker_id.replace("speaker_", ""));
                    }
                    
                    segmentEl.textContent = segment.text + " ";
                    transcriptContainer.appendChild(segmentEl);
                });
            }
            
            // Update highlighting
            const transcriptElements = transcriptContainer.querySelectorAll("span");
            transcriptElements.forEach(spanEl => {
                const start = parseFloat(spanEl.getAttribute("data-start"));
                const end = parseFloat(spanEl.getAttribute("data-end"));
                
                if (currentTime >= start && currentTime <= end) {
                    spanEl.classList.add("highlighted");
                    
                    // Auto-scroll to the highlighted text
                    spanEl.scrollIntoView({ behavior: "smooth", block: "center" });
                } else {
                    spanEl.classList.remove("highlighted");
                }
            });
            
            // Update time display
            if (audioEl.duration) {
                timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(audioEl.duration)}`;
            }
            
            // Update progress bar
            if (audioEl.duration) {
                const progress = (currentTime / audioEl.duration) * 100;
                progressIndicator.style.width = `${progress}%`;
            }
        }
        
        // Audio event listeners
        audioEl.addEventListener("loadedmetadata", handleAudioLoaded);
        
        audioEl.addEventListener("timeupdate", () => {
            const currentTime = audioEl.currentTime;
            model.set("current_time", currentTime);
            model.save_changes();
            
            updateTranscriptHighlighting(currentTime);
        });
        
        audioEl.addEventListener("play", () => {
            model.set("is_playing", true);
            model.save_changes();
            playBtn.style.display = "none";
            pauseBtn.style.display = "inline-block";
        });
        
        audioEl.addEventListener("pause", () => {
            model.set("is_playing", false);
            model.save_changes();
            playBtn.style.display = "inline-block";
            pauseBtn.style.display = "none";
        });
        
        audioEl.addEventListener("ended", () => {
            model.set("is_playing", false);
            model.save_changes();
            playBtn.style.display = "inline-block";
            pauseBtn.style.display = "none";
        });
        
        // Control button event listeners
        playBtn.addEventListener("click", () => {
            audioEl.play();
        });
        
        pauseBtn.addEventListener("click", () => {
            audioEl.pause();
        });
        
        // Progress bar click for seeking
        progressBar.addEventListener("click", (e) => {
            const rect = progressBar.getBoundingClientRect();
            const position = (e.clientX - rect.left) / rect.width;
            if (audioEl.duration) {
                audioEl.currentTime = position * audioEl.duration;
            }
        });
        
        // Sync from Python to JavaScript
        model.on("change:audio_data_base64", () => {
            const audioDataBase64 = model.get("audio_data_base64");
            if (audioDataBase64) {
                // Create a blob URL for the audio data from base64
                const byteCharacters = atob(audioDataBase64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], { type: "audio/mp3" });
                const url = URL.createObjectURL(blob);
                
                // Set the audio source
                audioEl.src = url;
                audioEl.load();
            }
        });
        
        model.on("change:transcript", () => {
            const transcript = model.get("transcript");
            
            // Clear existing transcript
            transcriptContainer.innerHTML = "";
            
            // Rebuild transcript
            if (transcript && transcript.segments) {
                transcript.segments.forEach((segment, index) => {
                    const segmentEl = document.createElement("span");
                    segmentEl.setAttribute("data-start", segment.start);
                    segmentEl.setAttribute("data-end", segment.end);
                    segmentEl.setAttribute("data-index", index);
                    
                    // Add speaker information if available
                    if (segment.speaker_id) {
                        segmentEl.setAttribute("data-speaker", segment.speaker_id);
                        segmentEl.classList.add("speaker-" + segment.speaker_id.replace("speaker_", ""));
                    }
                    
                    segmentEl.textContent = segment.text + " ";
                    transcriptContainer.appendChild(segmentEl);
                });
            }
            
            // Update highlighting based on current time
            updateTranscriptHighlighting(model.get("current_time") || 0);
        });
        
        model.on("change:current_time", () => {
            const currentTime = model.get("current_time");
            if (Math.abs(audioEl.currentTime - currentTime) > 0.5) {
                audioEl.currentTime = currentTime;
            }
            updateTranscriptHighlighting(currentTime);
        });
        
        model.on("change:is_playing", () => {
            const isPlaying = model.get("is_playing");
            if (isPlaying && audioEl.paused) {
                audioEl.play();
            } else if (!isPlaying && !audioEl.paused) {
                audioEl.pause();
            }
        });
        
        // Initial sync of data
        if (model.get("audio_data_base64")) {
            const audioDataBase64 = model.get("audio_data_base64");
            const byteCharacters = atob(audioDataBase64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: "audio/mp3" });
            const url = URL.createObjectURL(blob);
            audioEl.src = url;
            audioEl.load();
        }
        
        if (model.get("transcript")) {
            updateTranscriptHighlighting(model.get("current_time") || 0);
        }
    }
    
    export default { render };
    """
    
    # CSS for the widget with speaker diarization support
    _css = """
    .audio-transcript-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        font-family: sans-serif;
        gap: 20px;
    }
    
    .audio-player-container {
        width: 100%;
        padding: 15px;
        background-color: #f0f0f0;
        border-radius: 8px;
    }
    
    .controls-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
    }
    
    .control-button {
        padding: 8px 15px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    
    .play-button {
        background-color: #4CAF50;
    }
    
    .pause-button {
        background-color: #f44336;
    }
    
    .time-display {
        margin-left: auto;
        font-size: 14px;
        color: #555;
    }
    
    .progress-container {
        margin-top: 10px;
        width: 100%;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #ddd;
        border-radius: 4px;
        cursor: pointer;
        overflow: hidden;
    }
    
    .progress-indicator {
        height: 100%;
        width: 0%;
        background-color: #4CAF50;
        border-radius: 4px;
        transition: width 0.1s linear;
    }
    
    .transcript-container {
        padding: 15px;
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        max-height: 300px;
        overflow-y: auto;
        line-height: 1.6;
    }
    
    .transcript-container span {
        display: inline;
        transition: background-color 0.3s ease;
    }
    
    .transcript-container .highlighted {
        background-color: #ffff99;
        font-weight: bold;
    }
    
    /* Speaker colors for diarization */
    .speaker-1 {
        color: #3366cc;
    }
    
    .speaker-2 {
        color: #dc3912;
    }
    
    .speaker-3 {
        color: #ff9900;
    }
    
    .speaker-4 {
        color: #109618;
    }
    
    .speaker-5 {
        color: #990099;
    }
    
    .speaker-6 {
        color: #0099c6;
    }
    
    audio {
        width: 100%;
        display: none; /* Hide default audio controls */
    }
    """

def check_elevenlabs_api_key():
    """Check if the ElevenLabs API key exists in the environment variables"""
    api_key = st.secrets.get("ELEVEN_LABS_API_KEY", "")
    if not api_key:
        st.error("Error: ELEVEN_LABS_API_KEY not found in secrets")
        return False
    return True

def transcribe_audio(audio_bytes, model_id="scribe_v1", language_code=None, 
                    timestamps_granularity="word", diarize=False, 
                    tag_audio_events=True, num_speakers=None):
    """Transcribe audio using ElevenLabs API with improved file handling"""
    if not check_elevenlabs_api_key():
        return {"error": "API key not available"}
    
    temp_file = None
    try:
        # Create a unique temporary file name
        unique_id = str(uuid.uuid4())
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"elevenlabs_audio_{unique_id}.mp3")
        
        # Write the audio data to the file - using with statement for proper cleanup
        with open(temp_path, 'wb') as f:
            f.write(audio_bytes)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Prepare the API request
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        # Set up headers
        headers = {"xi-api-key": ELEVEN_LABS_API_KEY}
        
        # Prepare the form data
        data = {
            "model_id": model_id,
            "timestamps_granularity": timestamps_granularity,
            "diarize": str(diarize).lower(),
            "tag_audio_events": str(tag_audio_events).lower()
        }
        
        if language_code:
            data["language_code"] = language_code
            
        if num_speakers:
            data["num_speakers"] = str(num_speakers)
        
        # Make a copy of the file for upload to avoid file locking issues
        with open(temp_path, 'rb') as f:
            file_content = f.read()
        
        # Prepare files for multipart form using BytesIO to avoid file locking
        files = {
            "file": (f"audio_{unique_id}.mp3", BytesIO(file_content), "audio/mpeg")
        }
        
        # Make the API request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=300) as client:  # 5-minute timeout
                    response = client.post(url, headers=headers, data=data, files=files)
                
                # Check for successful response
                if response.status_code == 200:
                    return response.json()
                else:
                    error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                    error_msg = f"API request failed with status {response.status_code}"
                    logger.error(f"{error_msg}: {error_detail}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retrying
                        continue
                    
                    return {
                        "error": error_msg,
                        "detail": error_detail
                    }
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    time.sleep(1)
                    continue
                raise
    
    except Exception as e:
        logger.error(f"Error in transcription request: {str(e)}")
        return {"error": str(e)}
    
    finally:
        # Clean up: ensure the temporary file is removed
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temporary file: {str(cleanup_error)}")

def convert_elevenlabs_transcript_to_segments(transcript_data, segment_strategy="sentences"):
    """Convert ElevenLabs speech-to-text response to segment format for the widget"""
    if "error" in transcript_data:
        return {"segments": [], "error": transcript_data["error"]}
    
    if "words" not in transcript_data:
        return {"segments": [], "error": "No word data found in transcript"}
    
    words = transcript_data["words"]
    
    # Helper function to check end of sentence
    def is_end_of_sentence(text):
        return text and any(text.endswith(char) for char in ['.', '!', '?', ':', ';'])
    
    # Different segmentation strategies
    if segment_strategy == "word_by_word":
        # One segment per word
        segments = []
        for word in words:
            if word.get("type") == "word":
                segments.append({
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "text": word.get("text", ""),
                    "speaker_id": word.get("speaker_id", None)
                })
    
    elif segment_strategy == "speakers" and any("speaker_id" in word for word in words):
        # Group by speaker changes
        segments = []
        current_segment = {
            "start": None,
            "end": None,
            "text": "",
            "speaker_id": None
        }
        
        for word in words:
            if word.get("type") == "spacing":
                continue
                
            # Get the current word's speaker
            current_speaker = word.get("speaker_id")
            
            # If this is the first word or speaker changed, start a new segment
            if current_segment["start"] is None or current_speaker != current_segment["speaker_id"]:
                # Save the previous segment if it has content
                if current_segment["text"]:
                    segments.append(current_segment)
                
                # Start a new segment
                current_segment = {
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "text": word.get("text", ""),
                    "speaker_id": current_speaker
                }
            else:
                # Add to current segment
                current_segment["text"] += " " + word.get("text", "")
                current_segment["end"] = word.get("end", 0)
        
        # Add the last segment
        if current_segment["text"]:
            segments.append(current_segment)
    
    elif segment_strategy == "time_blocks":
        # Group into segments of approximately 5 seconds
        block_duration = 5.0
        segments = []
        current_segment = {
            "start": words[0].get("start", 0) if words else 0,
            "end": None,
            "text": "",
            "speaker_id": words[0].get("speaker_id") if words and "speaker_id" in words[0] else None
        }
        
        for word in words:
            if word.get("type") == "spacing":
                continue
                
            # Check if adding this word would exceed the block duration
            if word.get("end", 0) - current_segment["start"] > block_duration and current_segment["text"]:
                # Finish current segment
                current_segment["end"] = word.get("start", 0)
                segments.append(current_segment)
                
                # Start a new segment
                current_segment = {
                    "start": word.get("start", 0),
                    "end": None,
                    "text": word.get("text", ""),
                    "speaker_id": word.get("speaker_id") if "speaker_id" in word else None
                }
            else:
                # Add space if not the first word
                if current_segment["text"]:
                    current_segment["text"] += " "
                # Add the word to the current segment
                current_segment["text"] += word.get("text", "")
                current_segment["end"] = word.get("end", 0)
        
        # Add the last segment
        if current_segment["text"]:
            segments.append(current_segment)
    
    else:  # Default to sentences
        # Group by sentence boundaries
        segments = []
        current_segment = {
            "start": None,
            "end": None,
            "text": "",
            "speaker_id": None
        }
        
        for word in words:
            if word.get("type") == "spacing":
                continue
                
            # If this is the first word in a segment, set the start time and speaker
            if current_segment["start"] is None:
                current_segment["start"] = word.get("start", 0)
                current_segment["speaker_id"] = word.get("speaker_id") if "speaker_id" in word else None
            
            # Add space if not the first word
            if current_segment["text"]:
                current_segment["text"] += " "
                
            # Add the word to the current segment
            current_segment["text"] += word.get("text", "")
            current_segment["end"] = word.get("end", 0)
            
            # Check if this word ends a sentence
            if is_end_of_sentence(word.get("text", "")):
                segments.append(current_segment)
                
                # Start a new segment
                current_segment = {
                    "start": None,
                    "end": None,
                    "text": "",
                    "speaker_id": None
                }
        
        # Add the last segment if not empty
        if current_segment["text"]:
            segments.append(current_segment)
    
    # Add language information if available
    result = {"segments": segments}
    if "language_code" in transcript_data:
        result["language_code"] = transcript_data["language_code"]
    if "language_probability" in transcript_data:
        result["language_probability"] = transcript_data["language_probability"]
    
    return result

def text_to_speech_with_timestamps(text, segments=None, voice_id="JBFqnCBsd6RMkjVDRZzb"):
    """Convert text to speech using ElevenLabs API with timestamp information"""
    if not check_elevenlabs_api_key():
        return None, None
        
    try:
        # Create ElevenLabs API client
        url = "https://api.elevenlabs.io/v1/text-to-speech"
        headers = {
            "xi-api-key": ELEVEN_LABS_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Prepare the request
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        # Add voice_id path parameter
        url = f"{url}/{voice_id}"
        
        # Send the request
        with httpx.Client(timeout=120) as client:
            response = client.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            audio_bytes = response.content
            
            # For simplicity, use the provided segments directly
            # (In a production implementation, you would map them using the character_timestamps)
            return audio_bytes, {"segments": segments if segments else []}
        else:
            error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            logger.error(f"ElevenLabs API Error: {response.status_code} - {error_detail}")
            return None, None
    
    except Exception as e:
        logger.error(f"Error generating speech with timestamps: {str(e)}")
        return None, None

def generate_sample_transcript():
    """Generate a sample transcript with segments"""
    return {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello and welcome to this demonstration."},
            {"start": 2.5, "end": 5.0, "text": "Today we're showing how to synchronize audio with text."},
            {"start": 5.0, "end": 8.0, "text": "This is a powerful feature for accessibility and education."},
            {"start": 8.0, "end": 11.0, "text": "As the audio plays, you can see the corresponding text highlighted."},
            {"start": 11.0, "end": 15.0, "text": "This makes it easier to follow along and understand the content."}
        ]
    }

def main():
    st.title("Audio-Text Synchronization Tool")
    st.write("Upload audio to generate a transcript or enter text to generate audio. Then watch as the text highlights in sync with audio playback.")
    
    # Initialize session state
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "audio_data_base64" not in st.session_state:
        st.session_state.audio_data_base64 = None
    if "transcript" not in st.session_state:
        st.session_state.transcript = {"segments": []}
    if "elevenlabs_transcript_raw" not in st.session_state:
        st.session_state.elevenlabs_transcript_raw = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "elevenlabs_voices" not in st.session_state:
        st.session_state.elevenlabs_voices = [
            ("Rachel (Female)", "JBFqnCBsd6RMkjVDRZzb"),
            ("Adam (Male)", "pNInz6obpgDQGcFmaJgB"),
            ("Antoni (Male)", "ErXwobaYiN019PkySvjV"),
            ("Bella (Female)", "EXAVITQu4vr4xnSDxMaL"),
            ("Elli (Female)", "MF3mGyEYCl7XYWbV9V6O")
        ]
    
    # Check for ElevenLabs API key
    if not ELEVENLABS_AVAILABLE:
        st.warning("ElevenLabs API key not found. Some features will be limited.")
        with st.expander("Set API Key"):
            api_key = st.text_input("ElevenLabs API Key:", type="password")
            if st.button("Save API Key") and api_key:
                os.environ["ELEVEN_LABS_API_KEY"] = api_key
                st.success("API key saved for this session!")
                st.rerun()
    
    # Main workflow selection - either upload audio or generate from text
    st.header("1. Choose Your Workflow")
    workflow = st.radio(
        "Select how you want to create synchronized audio-text content:",
        ["Upload Audio → Generate Transcript", "Upload or Create Transcript → Generate Audio"]
    )
    
    # WORKFLOW 1: Upload Audio → Generate Transcript
    if workflow == "Upload Audio → Generate Transcript":
        st.header("2. Upload Audio File")
        audio_file = st.file_uploader(
            "Upload an audio file (MP3, WAV, etc.)",
            type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm"]
        )
        
        if audio_file is not None:
            # Store raw audio data
            audio_bytes = audio_file.read()
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_data_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Display audio player for preview
            st.success(f"Audio file '{audio_file.name}' loaded successfully!")
            st.audio(audio_bytes)
            
            # Transcription options
            st.header("3. Generate Transcript")
            
            with st.expander("Transcription Options", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Language selection
                    language_options = {
                        "": "Auto-detect",
                        "en": "English",
                        "es": "Spanish",
                        "fr": "French",
                        "de": "German",
                        "it": "Italian",
                        "pt": "Portuguese",
                        "pl": "Polish"
                    }
                    language_code = st.selectbox(
                        "Language:",
                        options=list(language_options.keys()),
                        format_func=lambda x: language_options[x]
                    )
                    language_code = language_code if language_code else None
                    
                    # Segmentation strategy
                    segment_strategy = st.selectbox(
                        "Segmentation Strategy:",
                        options=["sentences", "time_blocks", "speakers", "word_by_word"],
                        help="How to group words into transcript segments"
                    )
                
                with col2:
                    # Timestamp granularity
                    timestamp_options = {
                        "word": "Word-level timestamps",
                        "character": "Character-level timestamps"
                    }
                    timestamps_granularity = st.selectbox(
                        "Timestamp Detail:",
                        options=list(timestamp_options.keys()),
                        format_func=lambda x: timestamp_options[x]
                    )
                    
                    # Speaker diarization
                    diarize = st.checkbox(
                        "Identify Speakers (Diarization)",
                        value=False
                    )
                    
                    # Number of speakers (only if diarization is enabled)
                    num_speakers = None
                    if diarize:
                        num_speakers = st.slider(
                            "Maximum Number of Speakers:",
                            min_value=1,
                            max_value=32,
                            value=2
                        )
            
            # Generate transcript button
            if st.button("Generate Transcript from Audio"):
                with st.spinner("Transcribing audio... This may take a while depending on file length."):
                    # Call the transcription API with the fixed function
                    transcript_data = transcribe_audio(
                        audio_bytes=st.session_state.audio_data,
                        model_id="scribe_v1",
                        language_code=language_code,
                        timestamps_granularity=timestamps_granularity,
                        diarize=diarize,
                        num_speakers=num_speakers
                    )
                    
                    if "error" not in transcript_data:
                        # Store raw response
                        st.session_state.elevenlabs_transcript_raw = transcript_data
                        
                        # Convert to segment format
                        transcript_segments = convert_elevenlabs_transcript_to_segments(
                            transcript_data,
                            segment_strategy=segment_strategy
                        )
                        
                        # Store processed transcript
                        st.session_state.transcript = transcript_segments
                        st.session_state.processing_complete = True
                        
                        st.success("Transcription completed successfully!")
                        
                        # Display language detection info if available
                        if "language_code" in transcript_data:
                            lang_code = transcript_data["language_code"]
                            lang_prob = transcript_data.get("language_probability", 0)
                            st.info(f"Detected language: {lang_code} (confidence: {lang_prob:.2f})")
                        
                        # Display the full text
                        with st.expander("Full Transcript Text", expanded=True):
                            st.write(transcript_data.get("text", "No text available"))
                        
                        # Export transcript as JSON
                        transcript_json = json.dumps(transcript_segments, indent=2)
                        st.download_button(
                            label="Download Transcript as JSON",
                            data=transcript_json,
                            file_name="transcript.json",
                            mime="application/json"
                        )
                    else:
                        st.error(f"Transcription failed: {transcript_data.get('error')}")
                        if "detail" in transcript_data:
                            st.code(transcript_data["detail"])
            
            # Alternative: Upload transcript option
            st.header("Or Upload an Existing Transcript")
            transcript_file = st.file_uploader("Upload Transcript (JSON)", type=["json"])
            
            if transcript_file is not None:
                try:
                    # Load transcript JSON
                    transcript_data = json.load(transcript_file)
                    st.session_state.transcript = transcript_data
                    st.session_state.processing_complete = True
                    st.success(f"Transcript file '{transcript_file.name}' loaded successfully!")
                except json.JSONDecodeError:
                    st.error("Invalid transcript file format. Please upload a valid JSON file.")
    
    # WORKFLOW 2: Upload or Create Transcript → Generate Audio
    else:
        st.header("2. Create or Upload Transcript")
        transcript_source = st.radio(
            "Transcript Source:",
            ["Enter Text", "Upload Transcript File"]
        )
        
        if transcript_source == "Enter Text":
            text_input = st.text_area(
                "Enter text to convert to speech:",
                value="Hello and welcome to this demonstration. Today we're showing how to synchronize ElevenLabs speech with text highlighting. This is a powerful feature for accessibility and education.",
                height=150
            )
            
            if text_input:
                # Generate simple segments from sentences
                sentences = text_input.split('.')
                segments = []
                current_time = 0.0
                
                for sentence in sentences:
                    if sentence.strip():
                        # Estimate duration based on word count (rough approximation)
                        words = len(sentence.split())
                        duration = max(1.0, words / 3.0)  # Assume 3 words per second
                        
                        segments.append({
                            "text": sentence.strip() + ".",
                            "start": current_time,
                            "end": current_time + duration
                        })
                        
                        current_time += duration
                
                # Store as transcript
                st.session_state.transcript = {"segments": segments}
                st.session_state.processing_complete = True
        else:
            # Upload transcript option
            transcript_file = st.file_uploader("Upload Transcript (JSON)", type=["json"], key="tts_transcript")
            
            if transcript_file is not None:
                try:
                    # Load transcript JSON
                    transcript_data = json.load(transcript_file)
                    st.session_state.transcript = transcript_data
                    st.session_state.processing_complete = True
                    st.success(f"Transcript file '{transcript_file.name}' loaded successfully!")
                    
                    # Show preview
                    with st.expander("Transcript Preview"):
                        if "segments" in transcript_data:
                            for i, segment in enumerate(transcript_data["segments"][:5]):
                                st.write(f"[{segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s]: {segment.get('text', '')}")
                            if len(transcript_data["segments"]) > 5:
                                st.write(f"... and {len(transcript_data['segments']) - 5} more segments")
                except json.JSONDecodeError:
                    st.error("Invalid transcript file format. Please upload a valid JSON file.")
        
        # Generate audio section
        if st.session_state.transcript.get("segments"):
            st.header("3. Generate Audio")
            
            # Voice selection
            voice_names = [name for name, _ in st.session_state.elevenlabs_voices]
            selected_voice_name = st.selectbox("Select Voice:", options=voice_names)
            
            # Get voice ID from name
            voice_id = next((id for name, id in st.session_state.elevenlabs_voices if name == selected_voice_name), 
                           "JBFqnCBsd6RMkjVDRZzb")  # Default to Rachel if not found
            
            # Generate button
            if st.button("Generate Audio from Transcript"):
                # Extract text from segments
                segments = st.session_state.transcript["segments"]
                text_input = " ".join([segment.get("text", "") for segment in segments])
                
                with st.spinner("Generating audio... This may take a while depending on text length."):
                    # Generate audio with timestamps
                    audio_bytes, updated_segments = text_to_speech_with_timestamps(
                        text=text_input,
                        segments=segments,
                        voice_id=voice_id
                    )
                    
                    if audio_bytes and updated_segments:
                        # Store in session state
                        st.session_state.audio_data = audio_bytes
                        st.session_state.audio_data_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        st.session_state.transcript = updated_segments
                        st.session_state.processing_complete = True
                        
                        st.success("Audio generated successfully!")
                        
                        # Play audio
                        st.audio(audio_bytes)
                        
                        # Provide a download button
                        st.download_button(
                            label="Download Generated Audio",
                            data=audio_bytes,
                            file_name="elevenlabs_audio.mp3",
                            mime="audio/mp3"
                        )
                    else:
                        st.error("Failed to generate audio. Please check your API key and try again.")
            
            # Alternative: Upload audio option
            st.header("Or Upload an Existing Audio File")
            audio_file = st.file_uploader(
                "Upload Audio File (MP3, WAV, etc.)",
                type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm"],
                key="tts_audio"
            )
            
            if audio_file is not None:
                # Store raw audio data
                audio_bytes = audio_file.read()
                st.session_state.audio_data = audio_bytes
                st.session_state.audio_data_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                st.session_state.processing_complete = True
                
                # Display audio player for preview
                st.success(f"Audio file '{audio_file.name}' loaded successfully!")
                st.audio(audio_bytes)
    
    # Display the synchronized player if processing is complete
    if st.session_state.processing_complete and st.session_state.audio_data_base64 and st.session_state.transcript.get("segments"):
        st.header("4. Synchronized Audio-Text Player")
        
        # Initialize and render the synchronized player widget
        audio_transcript_widget = AudioTranscriptWidget(
            audio_data_base64=st.session_state.audio_data_base64,
            transcript=st.session_state.transcript
        )
        
        # Render the widget with anywidget
        widget_state = anywidget(audio_transcript_widget, key="audio_transcript")
        
        # Display transcript text separately
        with st.expander("Transcript Text"):
            segments = st.session_state.transcript.get('segments', [])
            for segment in segments:
                speaker_info = f" [Speaker: {segment.get('speaker_id', 'unknown')}]" if segment.get('speaker_id') else ""
                st.write(f"[{segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s]{speaker_info}: {segment.get('text', '')}")
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            # Export transcript
            transcript_json = json.dumps(st.session_state.transcript, indent=2)
            st.download_button(
                label="Download Transcript as JSON",
                data=transcript_json,
                file_name="transcript.json",
                mime="application/json"
            )
        
        with col2:
            # Export audio
            if st.session_state.audio_data:
                st.download_button(
                    label="Download Audio (MP3)",
                    data=st.session_state.audio_data,
                    file_name="audio.mp3",
                    mime="audio/mp3"
                )
    
    # Instructions in the sidebar
    with st.sidebar:
        st.title("Audio-Text Sync")
        st.markdown("""
        ### Quick Start
        
        This tool helps you create synchronized audio and text content using ElevenLabs' AI capabilities.
        
        **Two main workflows:**
        
        1. **Audio to Text**: Upload audio to generate a timestamped transcript
        2. **Text to Audio**: Create or upload a transcript to generate synchronized audio
        
        **How It Works:**
        
        - Audio transcription with timestamp precision
        - Text-to-speech with synchronized timestamps
        - Real-time text highlighting during playback
        
        **Applications:**
        
        - Educational content
        - Accessibility features
        - Video subtitling
        - Language learning
        """)
        
        # API Key setup
        st.header("ElevenLabs API Key")
        api_key = st.text_input("Enter your API key:", type="password", key="api_key_sidebar")
        if st.button("Save API Key", key="save_api_sidebar"):
            os.environ["ELEVEN_LABS_API_KEY"] = api_key
            st.success("API key saved!")
            st.rerun()

if __name__ == "__main__":
    main()