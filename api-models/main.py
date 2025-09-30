import cv2
from PIL import Image
from groq import Groq
import time
import tempfile
import os
from google import genai
from google.genai import types
import io

client_gemini = genai.Client(api_key="")
client = Groq(api_key="")

VIDEO_PATH = "videos/test2.mp4"
BLOCK_DURATION = 30  # seconds per block


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return video, fps, duration


def transcribe_block(video_path, start_time, end_time, whisper_model="whisper-large-v3-turbo"):
    # Create a unique temporary file
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

    # Open the extracted audio and send to Groq for transcription
    with open(tmp_audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=whisper_model
        )

    os.remove(tmp_audio_path)
    return transcription.text.strip()


def summarize_text_llama(text, language="auto-detect", size="short", persona="Expert", extra_prompt="", model="openai/gpt-oss-120b", max_tokens=100):
    if not text.strip():
        return "[No speech detected in this block]"

    system_prompt = f"""
    You are a video content summary generator.
    Summarize the following text clearly and in detail, taking into account the scene, maintaining context, and highlighting important nuances.
    Generate only the summary, without any extra information.

    {extra_prompt}

    **Instructions**:
    - Personality: {persona}
    - Language: {language}
    - Summary length: {size}
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        model=model,
        max_tokens=max_tokens,
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
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            'Describe this image in one line.'
        ]
    )

    return response.text


def create_blocks(video_path, block_duration):
    video, fps, duration = get_video_info(video_path)
    blocks = []
    num_blocks = int(duration // block_duration) + 1

    for i in range(num_blocks):
        start_time = i * block_duration
        end_time = min((i + 1) * block_duration, duration)
        print(f"\n🔹 Processing block {i+1}/{num_blocks}: {start_time:.1f}s - {end_time:.1f}s")

        # Transcription
        block_text = transcribe_block(video_path, start_time, end_time)
        print(f"🔊 Block {i+1} transcription:")
        print(block_text if block_text else "[No speech]")

        # Frame description
        frame_description = describe_frame(video, fps, start_time, end_time)
        print(f"🖼️ Block {i+1} frame description:")
        print(frame_description)

        # Summary
        summary = summarize_text_llama(
            f"Transcription: {block_text}\nVisual description: {frame_description}",
            max_tokens=500
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


def final_video_summary(blocks):
    combined_texts = []
    for i, block in enumerate(blocks):
        text = f"Block {i+1} ({block['start_time']:.1f}s-{block['end_time']:.1f}s):\n" \
               f"Audio summary: {block['audio_summary']}\n" \
               f"Frame description: {block['frame_description']}\n"
        combined_texts.append(text)

    full_text = "\n".join(combined_texts)
    final_summary = summarize_text_llama(
        full_text,
        language='english',
        size="medium",
        persona='Persuasive',
        extra_prompt="Present as a list of key points",
        max_tokens=500
    )
    return final_summary


if __name__ == "__main__":
    start_total = time.time()

    blocks = create_blocks(VIDEO_PATH, BLOCK_DURATION)

    print("\n=== FINAL VIDEO SUMMARY ===")
    summary = final_video_summary(blocks)
    print(summary)

    end_total = time.time()
    print(f"\n⏱ Total execution time: {end_total - start_total:.2f} seconds")
