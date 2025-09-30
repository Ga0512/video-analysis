import cv2
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from ollama import chat
import time
import os
import tempfile

VIDEO_PATH = "videos/test2.mp4"
BLOCK_DURATION = 30  # seconds per block

def initialize_models():
    print("Loading models...")
    whisper_model = WhisperModel("medium")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    ollama_client = "llama3.2:3b"
    print("Models loaded!")
    return whisper_model, blip_processor, blip_model, ollama_client


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return video, fps, duration


def transcribe_block(whisper_model, video_path, start_time, end_time):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    import subprocess
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-vn",
        "-y",
        tmp_audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    segments, _ = whisper_model.transcribe(tmp_audio_path, task="transcribe", language=None)
    text = " ".join([seg.text for seg in segments])

    os.remove(tmp_audio_path)
    return text.strip()


def summarize_text_llama(ollama_client, text, language="auto-detect", size="short", persona="Expert", extra_prompts="", max_tokens=200):
    if not text.strip():
        return "[No speech detected in this block]"
    
    system_prompt = f"""
    You are a video content summary generator. 
    Summarize the following text clearly and in detail, taking into account the scene, maintaining context, and highlighting important nuances. 
    Generate only the summary, no additional information.
    {extra_prompts}

    **Instructions**:
        - Personality: {persona}
        - Language: {language}
        - Summary length: {size}
    """
    
    response = chat(
        ollama_client,
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}],
        options={
            "num_predict": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 10,
            "repeat_penalty": 1.1
        }
    )
    return response['message']['content']


def describe_frame(video, fps, start_time, end_time, blip_processor, blip_model):
    frame_time = (start_time + end_time) / 2
    frame_index = int(frame_time * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        return "[Frame not available]"
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = blip_processor.decode(out[0], skip_special_tokens=True)
    return description


def create_blocks(video_path, block_duration, whisper_model, blip_processor, blip_model, ollama_client):
    video, fps, duration = get_video_info(video_path)
    blocks = []
    num_blocks = int(duration // block_duration) + 1

    for i in range(num_blocks):
        start_time = i * block_duration
        end_time = min((i+1) * block_duration, duration)
        print(f"\n🔹 Processing block {i+1}/{num_blocks}: {start_time:.1f}s - {end_time:.1f}s")

        # Transcription
        block_text = transcribe_block(whisper_model, video_path, start_time, end_time)
        print(f"🔊 Block {i+1} transcription:")
        print(block_text if block_text else "[No speech]")

        # Frame description
        frame_description = describe_frame(video, fps, start_time, end_time, blip_processor, blip_model)
        print(f"🖼️ Block {i+1} frame description:")
        print(frame_description)

        # Summary
        summary = summarize_text_llama(
            ollama_client, 
            f"Transcription: {block_text}\nVisual description: {frame_description}",
            max_tokens=600
        )
        print(f"📝 Block {i+1} summary:")
        print(summary)

        blocks.append({
            "start_time": start_time,
            "end_time": end_time,
            "transcription": block_text,
            "audio_summary": summary,
            "frame_description": frame_description
        })

    video.release()
    return blocks


def final_video_summary(blocks, ollama_client):
    combined_texts = []
    for block in blocks:
        text = f"Audio summary: {block['audio_summary']}\n" \
               f"Frame description: {block['frame_description']}\n"
        combined_texts.append(text)
    
    full_text = "\n".join(combined_texts)
    final_summary = summarize_text_llama(
        ollama_client,
        text=full_text,
        language="english",
        persona="Persuasive",
        size="medium",
        extra_prompts="Present as a list of key points",
        max_tokens=500
    )
    return final_summary


if __name__ == "__main__":
    start_total = time.time()

    whisper_model, blip_processor, blip_model, ollama_client = initialize_models()
    blocks = create_blocks(VIDEO_PATH, BLOCK_DURATION, whisper_model, blip_processor, blip_model, ollama_client)

    print("\n=== FINAL VIDEO SUMMARY ===")
    summary = final_video_summary(blocks, ollama_client)
    print(summary)

    end_total = time.time()
    print(f"\n⏱ Total execution time: {end_total - start_total:.2f} seconds")
