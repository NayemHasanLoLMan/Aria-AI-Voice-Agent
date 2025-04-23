from langchain_ollama import OllamaLLM
import os
from os import PathLike
from time import time
import asyncio
from typing import Union, Dict, List
from dataclasses import dataclass
from pathlib import Path
import threading
import queue
import wave
import struct
import pyaudio
import torch
import sys
import pygame
from pygame import mixer
import numpy as np
from dotenv import load_dotenv
from deepgram import Deepgram
import scipy.io.wavfile as wavfile

def setup_kokoro():
    """Setup Kokoro-82M repository and add it to Python path"""
    kokoro_path = Path("Kokoro-82M")
    if not kokoro_path.exists():
        os.system('git lfs install')
        os.system('git clone https://huggingface.co/hexgrad/Kokoro-82M')
    
    # Add Kokoro-82M directory to Python path
    kokoro_dir = str(kokoro_path.absolute())
    if kokoro_dir not in sys.path:
        sys.path.append(kokoro_dir)

# Setup Kokoro before importing its modules
setup_kokoro()

# Now we can import Kokoro modules
from models import build_model
from kokoro import generate

# Initialize Pygame for audio
pygame.init()

@dataclass
class JarvisConfig:
    """Configuration for Aria assistant"""
    recording_path: Path = Path("audio/recording.wav")
    response_path: Path = Path("audio/response.wav")
    conversation_log: Path = Path("conv.txt")
    status_log: Path = Path("status.txt")
    context: str = '''You are Aria, a friendly and professional AI assistant. You are here to help and provide information to the user. You are knowledgeable and have access to a wide range of information. You are polite, respectful, and always ready to assist.'''
    voice: str = "af"  # Default Kokoro voice
    chunk_size: int = 1024
    sample_rate: int = 44100
    silence_threshold: float = 0.1
    voice_activation_chunks: int = 3
    kokoro_dir: Path = Path("Kokoro-82M")
    model_path: Path = Path("Kokoro-82M/kokoro-v0_19.pth")
    voices_path: Path = Path("Kokoro-82M/voices")

class KokoroTTS:
    def __init__(self, config: JarvisConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self._setup_model()

    def _setup_model(self):
        if not self.config.kokoro_dir.exists():
            setup_kokoro()
            
        # Build model and load voicepack
        try:
            self.model = build_model(str(self.config.model_path), self.device)
            self.voicepack = torch.load(
                self.config.voices_path / f'{self.config.voice}.pt',
                map_location=self.device,
                weights_only=True
            )
            print("Successfully loaded Kokoro model and voicepack")
        except Exception as e:
            print(f"Error loading Kokoro model: {str(e)}")
            raise

    def generate_speech(self, text: str, output_path: Path) -> None:
        try:
            audio, _ = generate(
                self.model,
                text,
                self.voicepack,
                lang=self.config.voice[0]
            )
            
            # Save as WAV file
            wavfile.write(str(output_path), 24000, audio)
            print(f"Generated speech saved to {output_path}")
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise

class VoiceActivityDetector:
    def __init__(self, config: JarvisConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        self._setup_stream()

    def _setup_stream(self):
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size
        )

    def get_audio_level(self) -> float:
        try:
            data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            return np.max(np.abs(audio_data))
        except Exception:
            return 0.0

    def detect_voice(self) -> bool:
        if not self.is_listening:
            return False

        consecutive_voiced = 0
        for _ in range(self.config.voice_activation_chunks):
            level = self.get_audio_level()
            if level > self.config.silence_threshold:
                consecutive_voiced += 1
            else:
                consecutive_voiced = 0

            if consecutive_voiced >= self.config.voice_activation_chunks:
                return True

        return False

    def start(self):
        self.is_listening = True

    def stop(self):
        self.is_listening = False

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

class InterruptiblePlayer:
    def __init__(self, vad: VoiceActivityDetector):
        self.is_playing = False
        self.should_stop = False
        self.audio_queue = queue.Queue()
        self.vad = vad

    def play_audio(self, sound_path: Path) -> None:
        self.is_playing = True
        self.should_stop = False
        
        try:
            sound = mixer.Sound(str(sound_path))
            channel = sound.play()
            
            self.vad.start()
            
            while channel.get_busy() and not self.should_stop:
                if self.vad.detect_voice():
                    self.should_stop = True
                    channel.stop()
                    break
                pygame.time.wait(50)
            
            self.vad.stop()
        except Exception as e:
            print(f"Error playing audio: {str(e)}")
        finally:
            self.is_playing = False

    def stop(self):
        self.should_stop = True
        mixer.stop()

class JarvisAssistant:
    def __init__(self, config: JarvisConfig):
        self.config = config
        self._setup_apis()
        self.conversation_history: List[Dict[str, str]] = []
        self.vad = VoiceActivityDetector(config)
        self.player = InterruptiblePlayer(self.vad)
        self.tts = KokoroTTS(config)

    def _setup_apis(self) -> None:
        load_dotenv()

        # Setup LangChain with Ollama LLM
        self.llama_model = OllamaLLM(model="llama3.2")
        
        # Setup Deepgram
        self.deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
        mixer.init()

    def log(self, message: str) -> None:
        print(message)
        self.config.status_log.write_text(message)

    def append_to_conversation(self, text: str) -> None:
        with open(self.config.conversation_log, "a") as f:
            f.write(f"{text}\n")

    async def transcribe_audio(self, file_path: Union[str, PathLike]) -> str:
        with open(file_path, "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/wav"}
            response = await self.deepgram.transcription.prerecorded(source)
            words = response["results"]["channels"][0]["alternatives"][0]["words"]
            return " ".join(word_dict.get("word", "") for word_dict in words)

    def get_gpt_response(self, user_input: str) -> str:
        prompt = self.config.context + "\n" + "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in self.conversation_history
        )
        prompt += f"\nUser: {user_input}\nAssistant:"

        response = self.llama_model.invoke(input=prompt)
        print("Debug: LLM response:", response)
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("text", "No response text found")
        else:
            raise ValueError("Unexpected response format from LLM")

    def generate_and_play_audio(self, text: str) -> None:
        try:
            self.tts.generate_speech(text, self.config.response_path)
            
            play_thread = threading.Thread(
                target=self.player.play_audio,
                args=(self.config.response_path,)
            )
            play_thread.start()
            play_thread.join()

        except Exception as e:
            self.log(f"Error generating or playing audio: {str(e)}")
            raise

    async def conversation_loop(self) -> None:
        while True:
            try:
                self.log("Listening...")
                speech_to_text()
                self.log("Done listening")

                user_input = await self.transcribe_audio(self.config.recording_path)
                self.append_to_conversation(f"User: {user_input}")

                if self.player.is_playing:
                    self.player.stop()

                response = self.get_gpt_response(user_input)
                self.append_to_conversation(f"Aria: {response}")

                self.generate_and_play_audio(response)

                self.conversation_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])

                print(f"\n --- USER: {user_input}\n --- Aria: {response}\n")

            except Exception as e:
                self.log(f"Error occurred: {str(e)}")
                continue

    def cleanup(self):
        self.vad.cleanup()

def main():
    # Create necessary directories
    Path("audio").mkdir(exist_ok=True)
    
    config = JarvisConfig()
    jarvis = JarvisAssistant(config)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(jarvis.conversation_loop())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        jarvis.cleanup()
        loop.close()

if __name__ == "__main__":
    main()