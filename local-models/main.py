import os
import cv2
import time
import tempfile
import argparse
import subprocess
from PIL import Image
from datetime import timedelta
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from ollama import chat

SIZE_TO_TOKENS = {
    "short": 200,
    "medium": 700,
    "large": 1600
}


def initialize_models():
    print("Loading models...")
    whisper_model = WhisperModel("medium")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    ollama_client = "mistral:7b"
    print("Models loaded!")
    return whisper_model, blip_processor, blip_model, ollama_client


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return video, fps, duration


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - total_seconds) * 1000)
    return str(timedelta(seconds=total_seconds)) + f".{millis:03d}"


def transcribe_segments(whisper_model, video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "-y",
        tmp_audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    segments, _ = whisper_model.transcribe(tmp_audio_path, task="transcribe", language=None)
    os.remove(tmp_audio_path)
    return list(segments)


def summarize_text_llama(ollama_client, text, language="auto-detect", size="short", persona="Expert", extra_prompts=""):
    if not text.strip():
        return "[No speech detected in this block]"
    
    system_prompt = f"""
You are a video content summarizer.
Summarize the following text clearly and contextually, maintaining coherence and highlighting key ideas.
Output only the summary — no metadata or extra commentary.

{extra_prompts}

Personality: {persona}
Language: {language}
Summary length: {size}
"""
    
    response = chat(
        ollama_client,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text}
        ],
        options={
            "num_predict": SIZE_TO_TOKENS.get(size, 500),
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
    return blip_processor.decode(out[0], skip_special_tokens=True)


def create_blocks_smart(video_path, block_duration, whisper_model, blip_processor, blip_model, ollama_client):
    video, fps, duration = get_video_info(video_path)
    all_segments = transcribe_segments(whisper_model, video_path)

    blocks = []
    current_block = []
    current_start = None
    block_time = 0.0
    prev_summary = ""

    for seg in all_segments:
        if not current_block:
            current_start = seg.start
            block_time = 0.0

        if block_time + (seg.end - seg.start) > block_duration and current_block:
            block_end = current_block[-1].end
            block_text = " ".join([s.text.strip() for s in current_block])
            frame_description = describe_frame(video, fps, current_start, block_end, blip_processor, blip_model)

            summary_input = f"""
Previous summary (context): {prev_summary}
Current block transcription: {block_text}
Visual description: {frame_description}
"""
            summary = summarize_text_llama(ollama_client, summary_input)
            prev_summary = summary

            print(f"\n🔹 Block {len(blocks)+1}: {format_timestamp(current_start)} → {format_timestamp(block_end)}")
            for s in current_block:
                print(f"   [{format_timestamp(s.start)} → {format_timestamp(s.end)}] {s.text.strip()}")
            print(f"🖼️ Frame: {frame_description}")
            print(f"📝 Summary: {summary}")

            blocks.append({
                "start_time": current_start,
                "end_time": block_end,
                "transcription": block_text,
                "audio_summary": summary,
                "frame_description": frame_description
            })

            current_block = [seg]
            current_start = seg.start
            block_time = seg.end - seg.start
        else:
            current_block.append(seg)
            block_time = seg.end - current_start

    if current_block:
        block_end = current_block[-1].end
        block_text = " ".join([s.text.strip() for s in current_block])
        frame_description = describe_frame(video, fps, current_start, block_end, blip_processor, blip_model)

        summary_input = f"""
Previous summary (context): {prev_summary}
Current block transcription: {block_text}
Visual description: {frame_description}
"""
        summary = summarize_text_llama(ollama_client, summary_input)
        prev_summary = summary

        print(f"\n🔹 Block {len(blocks)+1}: {format_timestamp(current_start)} → {format_timestamp(block_end)}")
        for s in current_block:
            print(f"   [{format_timestamp(s.start)} → {format_timestamp(s.end)}] {s.text.strip()}")
        print(f"🖼️ Frame: {frame_description}")
        print(f"📝 Summary: {summary}")

        blocks.append({
            "start_time": current_start,
            "end_time": block_end,
            "transcription": block_text,
            "audio_summary": summary,
            "frame_description": frame_description
        })

    video.release()
    return blocks


def final_video_summary(blocks, ollama_client, language, persona, size, extra_prompts):
    combined_texts = []
    for i, block in enumerate(blocks, start=1):
        text = f"Block {i} ({format_timestamp(block['start_time'])} → {format_timestamp(block['end_time'])}):\n" \
               f"Audio summary: {block['audio_summary']}\n" \
               f"Frame description: {block['frame_description']}\n"
        combined_texts.append(text)
    
    full_text = "\n".join(combined_texts)
    return summarize_text_llama(
        ollama_client,
        text=full_text,
        language=language,
        persona=persona,
        size=size,
        extra_prompts=extra_prompts,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Video summarizer using Whisper, BLIP, and Ollama")
    parser.add_argument("video_path", type=str, help="Path or URL to the video file")
    parser.add_argument("--block_duration", type=int, default=30, help="Duration (seconds) of each block")
    parser.add_argument("--language", type=str, default="portuguese", help="Language of the final summary")
    parser.add_argument("--size", type=str, choices=["short", "medium", "large"], default="short", help="Summary size")
    parser.add_argument("--persona", type=str, default="Expert", help="Persona style for the summary")
    parser.add_argument("--extra_prompts", type=str, default="", help="Additional instructions for the summary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from utils.download_url import download

    VIDEO_PATH = args.video_path
    if VIDEO_PATH.startswith(("http://", "https://")):
        VIDEO_PATH = download(VIDEO_PATH)

    start_total = time.time()
    whisper_model, blip_processor, blip_model, ollama_client = initialize_models()

    blocks = create_blocks_smart(
        VIDEO_PATH,
        args.block_duration,
        whisper_model,
        blip_processor,
        blip_model,
        ollama_client
    )

    print("\n=== FINAL VIDEO SUMMARY ===")
    summary = final_video_summary(
        blocks,
        ollama_client,
        args.language,
        args.persona,
        args.size,
        args.extra_prompts
    )
    print(summary)

    end_total = time.time()
    print(f"\n⏱ Total execution time: {end_total - start_total:.2f} seconds")
