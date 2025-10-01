import yt_dlp
import os

def download(url, output_folder="downloads", output_template="%(title)s.%(ext)s"):
    os.makedirs(output_folder, exist_ok=True)

    # Define o caminho completo (pasta + template)
    outtmpl = os.path.join(output_folder, output_template)

    ydl_opts = {
        'format': 'best',
        'outtmpl': outtmpl,
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Gera o caminho final do arquivo
            filename = ydl.prepare_filename(info)

        print(f"Vídeo baixado com sucesso em: {filename}")
        return filename

    except Exception as e:
        print(f"Erro no download: {e}")
        return None

