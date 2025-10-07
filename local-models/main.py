import cv2
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from ollama import chat
import time
import os
import tempfile
import argparse
import subprocess
from datetime import timedelta

SIZE_TO_TOKENS = {
    "short": 200,
    "medium": 700,
    "large": 1600
}


# =========================
# Inicialização de modelos
# =========================
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


# =========================
# Helpers de formatação
# =========================
def format_timestamp(seconds: float) -> str:
    """Converte segundos em HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - total_seconds) * 1000)
    return str(timedelta(seconds=total_seconds)) + f".{millis:03d}"


# =========================
# Transcrição por segmentos
# =========================
def transcribe_segments(whisper_model, video_path):
    """
    Transcreve o vídeo inteiro em segmentos com timestamps.
    """
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


# =========================
# Resumo com LLaMA/Ollama
# =========================
def summarize_text_llama(ollama_client, text, language="auto-detect", size="short", persona="Expert", extra_prompts=""):
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
            "num_predict": SIZE_TO_TOKENS.get(size, 500),
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 10,
            "repeat_penalty": 1.1
        }
    )
    return response['message']['content']


# =========================
# Descrição visual (BLIP)
# =========================
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


# =========================
# Criação de blocos inteligentes + CONTEXTO

import re

def split_into_sentences(text):
    """
    Divide um texto em frases completas.
    Critério: começa com letra maiúscula e termina em ., ! ou ?
    """
    sentences = re.findall(r"[A-ZÀ-Ý][^.!?]*[.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


def create_blocks_smart(video_path, block_duration, whisper_model, blip_processor, blip_model, ollama_client):
    video, fps, duration = get_video_info(video_path)
    all_segments = transcribe_segments(whisper_model, video_path)

    blocks = []
    prev_summary = ""
    current_text = ""
    current_start = None
    current_time = 0.0

    for seg in all_segments:
        if current_start is None:
            current_start = seg.start
            current_time = 0.0

        # adiciona texto do segmento atual
        segment_text = seg.text.strip()
        current_text += " " + segment_text
        current_time = seg.end - current_start

        # verifica se já atingiu o limite de tempo
        if current_time >= block_duration:
            # quebra em frases completas
            sentences = split_into_sentences(current_text)
            if not sentences:
                continue

            # tenta manter apenas as frases que cabem dentro do limite
            cumulative_time = 0.0
            valid_sentences = []
            remaining_text = ""
            for s in sentences:
                valid_sentences.append(s)
                # estimativa: divide tempo proporcionalmente pelo número de caracteres
                cumulative_time = block_duration * (len(" ".join(valid_sentences)) / len(current_text))
                if cumulative_time >= block_duration:
                    valid_sentences.pop()  # remove última que excedeu
                    remaining_text = " ".join(sentences[len(valid_sentences):])
                    break

            block_text = " ".join(valid_sentences).strip()
            if not block_text:
                continue

            block_end = seg.end
            frame_description = describe_frame(video, fps, current_start, block_end, blip_processor, blip_model)

            summary_input = f"""
Previous summary (context): {prev_summary}

Current block transcription: {block_text}
Visual description: {frame_description}
"""
            summary = summarize_text_llama(ollama_client, summary_input)
            prev_summary = summary

            duracao = block_end - current_start
            print(f"\n🔹 Block {len(blocks)+1}: {format_timestamp(current_start)} → {format_timestamp(block_end)} "
                  f"(duração: {duracao:.2f}s)")

            # imprime cada frase individualmente com timestamp aproximado
            start_tmp = current_start
            for s in split_into_sentences(block_text):
                print(f"   [{format_timestamp(start_tmp)}] {s}")
                start_tmp += duracao / max(1, len(split_into_sentences(block_text)))

            print(f"🖼️ Frame description: {frame_description}")
            print(f"📝 Summary: {summary}")

            blocks.append({
                "start_time": current_start,
                "end_time": block_end,
                "transcription": block_text,
                "audio_summary": summary,
                "frame_description": frame_description
            })

            # prepara próximo bloco
            current_text = remaining_text
            current_start = seg.end - (len(remaining_text) / max(len(block_text), 1)) * block_duration
            current_time = len(remaining_text) / max(len(block_text), 1) * block_duration

    # trata o último bloco
    if current_text.strip():
        block_end = duration
        frame_description = describe_frame(video, fps, current_start, block_end, blip_processor, blip_model)
        summary_input = f"""
Previous summary (context): {prev_summary}

Current block transcription: {current_text.strip()}
Visual description: {frame_description}
"""
        summary = summarize_text_llama(ollama_client, summary_input)
        prev_summary = summary

        duracao = block_end - current_start
        print(f"\n🔹 Block {len(blocks)+1}: {format_timestamp(current_start)} → {format_timestamp(block_end)} "
              f"(duração: {duracao:.2f}s)")
        for s in split_into_sentences(current_text):
            print(f"   [{format_timestamp(current_start)}] {s}")
            current_start += duracao / max(1, len(split_into_sentences(current_text)))
        print(f"🖼️ Frame description: {frame_description}")
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


# =========================
# Resumo final
# =========================
def final_video_summary(blocks, ollama_client, language, persona, size, extra_prompts):
    combined_texts = []
    for i, block in enumerate(blocks, start=1):
        text = f"Block {i} ({format_timestamp(block['start_time'])} → {format_timestamp(block['end_time'])}):\n" \
               f"Audio summary: {block['audio_summary']}\n" \
               f"Frame description: {block['frame_description']}\n"
        combined_texts.append(text)
    
    full_text = "\n".join(combined_texts)
    final_summary = summarize_text_llama(
        ollama_client,
        text=full_text,
        language=language,
        persona=persona,
        size=size,
        extra_prompts=extra_prompts,
    )
    return final_summary


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Video summarizer with Whisper, BLIP and Ollama"
    )
    parser.add_argument("video_path", type=str, help="Path or URL to the video file")
    parser.add_argument("--block_duration", type=int, default=30,
                        help="Duration (in seconds) of each block (approximate, blocks won't cut sentences)")
    parser.add_argument("--language", type=str, default="portuguese",
                        help="Language of the final summary (e.g., english, portuguese, auto-detect)")
    parser.add_argument("--size", type=str, choices=["short", "medium", "large"], default="short",
                        help="Summary size: short, medium, large")
    parser.add_argument("--persona", type=str, default="Expert",
                        help="Persona style for the summary (e.g., Expert, Funny, Journalist)")
    parser.add_argument("--extra_prompts", type=str, default="",
                        help="Additional instructions to guide the summary")
    return parser.parse_args()


# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()

    from utils.download_url import download
    VIDEO_PATH = args.video_path
    if VIDEO_PATH.startswith("http://") or VIDEO_PATH.startswith("https://"):
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
