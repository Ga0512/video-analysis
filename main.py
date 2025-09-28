import cv2
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from ollama import chat
import time

VIDEO_PATH = "test2.mp4"
BLOCK_DURATION = 30  

def inicializar_modelos():
    print("Carregando modelos...")
    whisper_model = WhisperModel("medium")  
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    ollama_client = "llama3.2:3b"  
    print("Modelos carregados!")
    return whisper_model, blip_processor, blip_model, ollama_client


def obter_info_video(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = frame_count / fps
    return video, fps, duracao


def transcrever_bloco(whisper_model, video_path, start_time, end_time):
    segments, _ = whisper_model.transcribe(video_path, task="transcribe", language=None)  # detecção automática de idioma
    texto = " ".join([seg.text for seg in segments if seg.start >= start_time and seg.end <= end_time])
    return texto.strip()


def resumir_texto_llama(ollama_client, texto, max_tokens=100):
    if not texto.strip():
        return "[Sem fala detectada neste bloco]"
    prompt = f"""
Você é um assistente especialista em resumir conteúdo multimodal de vídeos. 
Resuma o seguinte texto de forma clara, detalhada, objetiva e natural, mantendo contexto e nuances importantes:

{texto}

Resumo detalhado:
"""
    resposta = chat(ollama_client, 
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        "num_predict": max_tokens,       
                        "temperature": 0.1,        
                        "top_p": 0.9,              
                        "top_k": 10,              
                        "repeat_penalty": 1.1      
                    })
    return resposta['message']['content']


def descrever_frame(video, fps, start_time, end_time, blip_processor, blip_model):
    frame_time = (start_time + end_time) / 2
    frame_index = int(frame_time * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        return "[Frame não disponível]"
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    descricao = blip_processor.decode(out[0], skip_special_tokens=True)
    return descricao


def criar_blocos(video_path, block_duration, whisper_model, blip_processor, blip_model, ollama_client):
    video, fps, duracao = obter_info_video(video_path)
    blocos = []
    num_blocos = int(duracao // block_duration) + 1

    for i in range(num_blocos):
        start_time = i * block_duration
        end_time = min((i+1) * block_duration, duracao)
        print(f"\n🔹 Processando bloco {i+1}/{num_blocos}: {start_time:.1f}s - {end_time:.1f}s")

        # Transcrição
        texto_bloco = transcrever_bloco(whisper_model, video_path, start_time, end_time)
        print(f"🔊 Transcrição do bloco {i+1}:")
        print(texto_bloco if texto_bloco else "[Sem fala]")

        # Resumo
        resumo = resumir_texto_llama(ollama_client, texto_bloco)
        print(f"📝 Resumo do bloco {i+1}:")
        print(resumo)

        # Frame
        descricao = descrever_frame(video, fps, start_time, end_time, blip_processor, blip_model)
        print(f"🖼️ Descrição do frame do bloco {i+1}:")
        print(descricao)

        blocos.append({
            "tempo_inicio": start_time,
            "tempo_fim": end_time,
            "transcricao": texto_bloco,
            "resumo_audio": resumo,
            "descricao_frame": descricao
        })

    video.release()
    return blocos


def resumo_final_video(blocos, ollama_client):
    textos_combinados = []
    for i, bloco in enumerate(blocos):
        texto = f"Bloco {i+1} ({bloco['tempo_inicio']:.1f}s-{bloco['tempo_fim']:.1f}s):\n" \
                f"Resumo áudio: {bloco['resumo_audio']}\n" \
                f"Descrição do frame: {bloco['descricao_frame']}\n"
        textos_combinados.append(texto)
    texto_completo = "\n".join(textos_combinados)
    resumo_final = resumir_texto_llama(ollama_client, texto_completo, max_tokens=500)
    return resumo_final


if __name__ == "__main__":
    start_total = time.time()

    whisper_model, blip_processor, blip_model, ollama_client = inicializar_modelos()
    blocos = criar_blocos(VIDEO_PATH, BLOCK_DURATION, whisper_model, blip_processor, blip_model, ollama_client)

    print("\n=== RESUMO FINAL DO VÍDEO ===")
    resumo = resumo_final_video(blocos, ollama_client)
    print(resumo)

    end_total = time.time()
    print(f"\n⏱ Tempo total de execução: {end_total - start_total:.2f} segundos")
