from langchain_community.llms import Ollama
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

from dotenv import load_dotenv
from deepgram import Deepgram
import pygame
from pygame import mixer
import elevenlabs
import numpy as np

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
    voice: str = "Aria"
    chunk_size: int = 1024
    sample_rate: int = 44100
    silence_threshold: float = 0.1
    voice_activation_chunks: int = 3

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
        self.is_playing = False

    def stop(self):
        self.should_stop = True
        mixer.stop()

def record_audio(filename: Union[str, PathLike], duration: int = 5) -> None:
    """Record audio from microphone and save to WAV file"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)

    frames = []

    print("* recording")

    # Record for specified duration
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(str(filename), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def speech_to_text() -> None:
    """Record audio and save it to the specified path"""
    # Record audio for 5 seconds
    record_audio("audio/recording.wav", duration=5)

class JarvisAssistant:
    def __init__(self, config: JarvisConfig):
        self.config = config
        self._setup_apis()
        self.conversation_history: List[Dict[str, str]] = []
        self.vad = VoiceActivityDetector(config)
        self.player = InterruptiblePlayer(self.vad)

    def _setup_apis(self) -> None:
        load_dotenv()

        # Setup LangChain with Ollama
        self.llama_model = Ollama(model="llama2")
        
        # Setup ElevenLabs
        elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))
        available_voices = elevenlabs.voices()
        if not any(voice.name == self.config.voice for voice in available_voices):
            self.config.voice = available_voices[0].name if available_voices else "Josh"
        
        # Setup Deepgram
        self.deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
        mixer.init()

    def log(self, message: str) -> None:
        print(message)
        with open(self.config.status_log, "w") as f:
            f.write(message)

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
        print("Debug: Llama2 response:", response)
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("text", "No response text found")
        else:
            raise ValueError("Unexpected response format from Llama2")

    def generate_and_play_audio(self, text: str) -> None:
        try:
            audio = elevenlabs.generate(
                text=text,
                voice=self.config.voice,
                model="eleven_monolingual_v1"
            )
            elevenlabs.save(audio, str(self.config.response_path))
            
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

                start_time = time()
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
    os.makedirs("audio", exist_ok=True)
    
    config = JarvisConfig()
    jarvis = JarvisAssistant(config)
    
    # Ensure all log files exist
    config.conversation_log.touch(exist_ok=True)
    config.status_log.touch(exist_ok=True)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(jarvis.conversation_loop())
    finally:
        jarvis.cleanup()
        loop.close()

if __name__ == "__main__":
    main()