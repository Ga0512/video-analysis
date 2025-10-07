import os
import io
import cv2
import time
import tempfile
import subprocess
from PIL import Image
from dotenv import load_dotenv
from datetime import timedelta
from groq import Groq
from google import genai
from google.genai import types
import argparse

load_dotenv()

client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SIZE_TO_TOKENS = {
    "short": 200,
    "medium": 700,
    "large": 1600
}


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - total_seconds) * 1000)
    return str(timedelta(seconds=total_seconds)) + f".{millis:03d}"


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return video, fps, duration


def transcribe_segments(video_path, whisper_model="whisper-large-v3-turbo"):
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

    with open(tmp_audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=whisper_model,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    os.remove(tmp_audio_path)
    segs = getattr(transcription, "segments", None)
    if segs is None:
        raise ValueError("No segments returned. Check model or response_format parameters.")

    segments = []
    for s in segs:
        start = getattr(s, "start", None) if hasattr(s, "start") else s.get("start")
        end = getattr(s, "end", None) if hasattr(s, "end") else s.get("end")
        text = getattr(s, "text", None) if hasattr(s, "text") else s.get("text")
        segments.append({"start": float(start), "end": float(end), "text": text})
    return segments


def summarize_text_llama(text, language="auto-detect", size="short", persona="Expert", extra_prompt="", model="openai/gpt-oss-120b"):
    if not text.strip():
        return "[No speech detected in this block]"

    system_prompt = f"""
You are a video content summarizer.
Summarize the text clearly and contextually, keeping key ideas and transitions.
Generate only the summary — no metadata or filler.

{extra_prompt}

Personality: {persona}
Language: {language}
Summary length: {size}
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        model=model,
        max_tokens=SIZE_TO_TOKENS.get(size, 500),
        temperature=0.1,
        top_p=0.9
    )

    return chat_completion.choices[0].message.content


def describe_frame(video, fps, start_time, end_time):
    frame_time = (start_time + end_time) / 2
    frame_index = int(frame_time * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        return "[Frame not available]"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    response = client_gemini.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            'Describe this image in one line.'
        ]
    )
    return response.text

import re

def split_into_sentences(text):
    """
    Divide o texto em frases completas.
    Critério: começa com letra maiúscula e termina em ., ! ou ?
    """
    sentences = re.findall(r"[A-ZÀ-Ý][^.!?]*[.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


def create_blocks_smart(video_path, block_duration):
    video, fps, duration = get_video_info(video_path)
    all_segments = transcribe_segments(video_path)

    blocks = []
    prev_summary = ""
    current_text = ""
    current_start = None
    current_time = 0.0

    for seg in all_segments:
        if current_start is None:
            current_start = seg["start"]
            current_time = 0.0

        # adiciona o texto do segmento
        segment_text = seg["text"].strip()
        current_text += " " + segment_text
        current_time = seg["end"] - current_start

        # quando passa o limite de tempo, cria bloco respeitando frases completas
        if current_time >= block_duration:
            sentences = split_into_sentences(current_text)
            if not sentences:
                continue

            # coleta apenas as frases que cabem dentro da duração
            cumulative_time = 0.0
            valid_sentences = []
            remaining_text = ""
            for s in sentences:
                valid_sentences.append(s)
                cumulative_time = block_duration * (len(" ".join(valid_sentences)) / len(current_text))
                if cumulative_time >= block_duration:
                    valid_sentences.pop()
                    remaining_text = " ".join(sentences[len(valid_sentences):])
                    break

            block_text = " ".join(valid_sentences).strip()
            if not block_text:
                continue

            block_end = seg["end"]
            frame_description = describe_frame(video, fps, current_start, block_end)

            summary_input = f"""
            Previous summary (context): {prev_summary}
            Current block transcription: {block_text}
            Visual description: {frame_description}
            """
            summary = summarize_text_llama(summary_input)
            prev_summary = summary

            duracao = block_end - current_start
            print(f"\n🔹 Block {len(blocks)+1}: {format_timestamp(current_start)} → {format_timestamp(block_end)} "
                  f"(duração: {duracao:.2f}s)")

            # imprime cada frase com timestamp aproximado
            start_tmp = current_start
            sentences_in_block = split_into_sentences(block_text)
            for s in sentences_in_block:
                print(f"   [{format_timestamp(start_tmp)}] {s}")
                start_tmp += duracao / max(1, len(sentences_in_block))

            print(f"🖼️ Frame: {frame_description}")
            print(f"📝 Summary: {summary}")

            blocks.append({
                "start_time": current_start,
                "end_time": block_end,
                "transcription": block_text,
                "audio_summary": summary,
                "frame_description": frame_description
            })

            # prepara o próximo bloco
            current_text = remaining_text
            current_start = seg["end"] - (len(remaining_text) / max(len(block_text), 1)) * block_duration
            current_time = len(remaining_text) / max(len(block_text), 1) * block_duration

    # trata o último bloco restante
    if current_text.strip():
        block_end = duration
        frame_description = describe_frame(video, fps, current_start, block_end)

        summary_input = f"""
        Previous summary (context): {prev_summary}
        Current block transcription: {current_text.strip()}
        Visual description: {frame_description}
        """
        summary = summarize_text_llama(summary_input)
        prev_summary = summary

        duracao = block_end - current_start
        print(f"\n🔹 Block {len(blocks)+1}: {format_timestamp(current_start)} → {format_timestamp(block_end)} "
              f"(duração: {duracao:.2f}s)")
        sentences_in_block = split_into_sentences(current_text)
        for s in sentences_in_block:
            print(f"   [{format_timestamp(current_start)}] {s}")
            current_start += duracao / max(1, len(sentences_in_block))
        print(f"🖼️ Frame: {frame_description}")
        print(f"📝 Summary: {summary}")

        blocks.append({
            "start_time": current_start,
            "end_time": block_end,
            "transcription": current_text.strip(),
            "audio_summary": summary,
            "frame_description": frame_description
        })

    video.release()
    return blocks


def final_video_summary(blocks, language, persona, size, extra_prompts):
    combined_texts = []
    for i, block in enumerate(blocks):
        text = f"Block {i+1} ({format_timestamp(block['start_time'])} → {format_timestamp(block['end_time'])}):\n" \
               f"Audio summary: {block['audio_summary']}\n" \
               f"Frame description: {block['frame_description']}\n"
        combined_texts.append(text)

    full_text = "\n".join(combined_texts)
    return summarize_text_llama(
        full_text,
        language=language,
        size="medium",
        persona=persona,
        extra_prompt=extra_prompts,
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

    BLOCK_DURATION = args.block_duration
    LANGUAGE = args.language
    SIZE = args.size
    PERSONA = args.persona
    EXTRA_PROMPTS = args.extra_prompts

    start_total = time.time()
    blocks = create_blocks_smart(VIDEO_PATH, BLOCK_DURATION)

    print("\n=== FINAL VIDEO SUMMARY ===")
    summary = final_video_summary(blocks, LANGUAGE, PERSONA, SIZE, EXTRA_PROMPTS)
    print(summary)

    end_total = time.time()
    print(f"\n⏱ Total execution time: {end_total - start_total:.2f} seconds")
