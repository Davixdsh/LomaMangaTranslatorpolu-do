import asyncio

import io

import json

import os

import re

import threading

import uuid

import zipfile

from datetime import datetime, timezone

from pathlib import Path



import ebooklib

import fitz

import pytesseract

from ebooklib import epub

from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from fastapi.responses import FileResponse, HTMLResponse

from PIL import Image, ImageDraw, ImageFont

from pytesseract import Output



BASE = Path(__file__).resolve().parent

DATA = BASE / "data"

UPLOADS = DATA / "uploads"

OUTPUTS = DATA / "outputs"

TASKS_DIR = DATA / "tasks"

TEMPLATE = BASE / "templates" / "index.html"



for folder in (UPLOADS, OUTPUTS, TASKS_DIR):

    folder.mkdir(parents=True, exist_ok=True)



MODEL = "gemini-3.5-flash-lite"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MAX_LINES_PER_REQUEST = 1



app = FastAPI(title="LomaMangaTranslator")



TASKS = {}

TASKS_LOCK = threading.RLock()

QUEUE = asyncio.Queue()

WORKER = None





def now():

    return datetime.now(timezone.utc).isoformat()





def task_file(task_id):

    return TASKS_DIR / f"{task_id}.json"





def save_task(task):

    task["updated_at"] = now()

    temp = task_file(task["id"]).with_suffix(".tmp")

    temp.write_text(

        json.dumps(task, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    temp.replace(task_file(task["id"]))

    with TASKS_LOCK:

        TASKS[task["id"]] = task.copy()





def get_task(task_id):

    with TASKS_LOCK:

        task = TASKS.get(task_id)

        return task.copy() if task else None





def update_task(task_id, **changes):

    task = get_task(task_id)

    if not task:

        return

    task.update(changes)

    save_task(task)





def all_tasks():

    with TASKS_LOCK:

        items = [item.copy() for item in TASKS.values()]

    return sorted(items, key=lambda x: x.get("created_at", ""))





def public_task(task):

    return {

        "id": task["id"],

        "name": task.get("name", "Arquivo"),

        "status": task.get("status", "queued"),

        "message": task.get("message", ""),

        "current": task.get("current", 0),

        "total": task.get("total", 0),

        "percent": task.get("percent", 0),

        "output_format": task.get("output_format", "pdf"),

        "created_at": task.get("created_at", ""),

        "updated_at": task.get("updated_at", ""),

        "error": task.get("error", ""),

    }





def load_saved_tasks():

    for path in TASKS_DIR.glob("*.json"):

        try:

            task = json.loads(path.read_text(encoding="utf-8"))

            if task.get("status") == "processing":

                task["status"] = "queued"

                task["message"] = "Retomando tarefa após reinício do servidor..."

                save_task(task)

            else:

                with TASKS_LOCK:

                    TASKS[task["id"]] = task

        except Exception as exc:

            print(f"Não foi possível recuperar {path.name}: {exc}")





def ollama_generate_ollama(prompt):
    import urllib.request

    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_ctx": 2048,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result.get("response", "")
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao comunicar com o Ollama. Verifique se ele está rodando. Detalhes: {exc}"
        ) from exc

def ollama_generate(prompt):
    """Usa Gemini quando configurado; caso contrário, usa Ollama."""
    import json
    import os
    import urllib.request
    import urllib.error

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ollama_generate_ollama(prompt)

    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent"
    )

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        },
        method="POST"
    )

    import time
    import re
    import random

    result = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                },
                method="POST"
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            if exc.code not in (429, 503, 408) or attempt == 5:
                raise RuntimeError(
                    f"Erro na API Gemini (HTTP {exc.code}): {details[:1200]}"
                ) from None

            match = re.search(r"retry in ([0-9.]+)s", details, re.I)
            delay = float(match.group(1)) if match else min(5 * (2 ** attempt), 60)
            delay = min(max(delay, 2), 120) + random.uniform(0, 1.5)
            print(
                f"Gemini limitou a requisição (HTTP {exc.code}). "
                f"Tentativa {attempt + 2}/6 em {delay:.1f}s.",
                flush=True
            )
            time.sleep(delay)
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao comunicar com a API Gemini: {exc}"
            ) from None

    if result is None:
        raise RuntimeError("Gemini não respondeu após várias tentativas.")

    candidates = result.get("candidates") or []
    if not candidates:
        feedback = result.get("promptFeedback", {})
        raise RuntimeError(
            "Gemini não retornou texto. "
            + json.dumps(feedback, ensure_ascii=False)[:800]
        )

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(
        part.get("text", "") for part in parts
        if isinstance(part.get("text"), str)
    ).strip()

    if not text:
        raise RuntimeError("Gemini retornou uma resposta vazia.")

    return text





def translate_batch(records, context=""):

    if not records:

        return {}



    results = {}

    for start in range(0, len(records), MAX_LINES_PER_REQUEST):

        chunk = records[start:start + MAX_LINES_PER_REQUEST]

        payload = [

            {"id": item["id"], "text": item["text"]}

            for item in chunk

        ]

        prompt = f"""

Você é um tradutor profissional de mangás.

Traduza do inglês para português brasileiro natural.

Preserve nomes próprios, personalidade, tom, humor e sentido.

REGRAS IMPORTANTES:
- Não traduza títulos de obras, logotipos, nomes de editoras ou nomes próprios.
- Preserve exatamente "Demon Slayer: Kimetsu no Yaiba" e "Kimetsu no Yaiba".
- Não transforme nomes japoneses em palavras traduzidas.
- Traduza apenas o texto original que realmente contém uma fala ou texto legível.
- Não invente palavras para completar trechos que não estejam visíveis.
- Traduza integralmente cada texto fornecido, sem omitir palavras ou frases.
- Mantenha cada ID e devolva uma tradução para cada item recebido.

Não invente falas nem acrescente explicações.

Cada tradução deve corresponder ao ID da fala original.

Use o contexto anterior apenas para entender a história.

Responda somente em JSON válido no formato:

{{"translations":[{{"id":1,"translation":"texto traduzido"}}]}}



Contexto anterior:

{context[-3500:] if context else "(início da história)"}



Fal falas para traduzir:

{json.dumps(payload, ensure_ascii=False)}

""".replace("Fal falas", "Falas")



        raw = ollama_generate(prompt)

        try:

            # Aceita JSON normal, JSON cercado por blocos Markdown

            # ou um objeto JSON incluído em texto adicional.

            cleaned = raw.strip()

            cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.I)

            cleaned = re.sub(r"\\s*```$", "", cleaned)



            try:

                parsed = json.loads(cleaned)

            except json.JSONDecodeError:

                match = re.search(r"\\{.*\\}", cleaned, flags=re.S)

                if not match:

                    raise

                parsed = json.loads(match.group(0))



            translations = parsed.get("translations", [])

            for item in translations:

                if "id" in item and "translation" in item:

                    results[int(item["id"])] = str(item["translation"]).strip()



        except Exception:

            # Se o modelo não respeitar o JSON, traduz cada fala

            # individualmente, sem interromper imediatamente o trabalho.

            for item in chunk:

                if item["id"] in results:

                    continue



                single_prompt = (

                    "Translate this English manga dialogue into natural Brazilian Portuguese. "

                    "Preserve character names and tone. Return only the translated dialogue, "

                    "without explanations or alternatives.\\n\\n"

                    + item["text"]

                )

                answer = ollama_generate(single_prompt).strip()

                answer = re.sub(r"^```(?:text)?\\s*", "", answer, flags=re.I)

                answer = re.sub(r"\\s*```$", "", answer).strip()



                if answer:

                    results[item["id"]] = answer



            missing_ids = [item["id"] for item in chunk if not results.get(item["id"])]

            if missing_ids:

                raise RuntimeError(

                    "O modelo não conseguiu traduzir algumas falas. IDs: "

                    + ", ".join(map(str, missing_ids))

                )



            context = "\\n".join(

                f"{item['text']} → {results[item['id']]}" for item in chunk

            )[-3500:]

            continue



        missing = [item for item in chunk if item["id"] not in results]

        if missing:

            retry_prompt = (

                "Traduza cada fala para português brasileiro. Preserve os IDs. "

                "Retorne somente JSON válido no formato "

                '{"translations":[{"id":1,"translation":"..."}]}.\n'

                + json.dumps(

                    [{"id": x["id"], "text": x["text"]} for x in missing],

                    ensure_ascii=False,

                )

            )

            try:

                retry = json.loads(ollama_generate(retry_prompt))

                for item in retry.get("translations", []):

                    results[int(item["id"])] = str(item["translation"]).strip()

            except Exception:

                pass



        if missing and any(item["id"] not in results for item in missing):

            raise RuntimeError("Não foi possível traduzir todas as falas deste lote.")



        context = "\n".join(

            f"{item['text']} → {results[item['id']]}" for item in chunk

        )[-3500:]



    return results





def pdf_images(path):

    document = fitz.open(path)

    images = []

    for page in document:

        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)

        images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))

    document.close()

    return images





def epub_images(path):

    book = epub.read_epub(str(path))

    images = []

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):

        try:

            image = Image.open(io.BytesIO(item.get_content())).convert("RGB")

            images.append(image)

        except Exception:

            continue

    if not images:

        raise RuntimeError("Não encontrei páginas em imagem dentro deste EPUB.")

    return images





def extract_lines(image, page_number):
    """OCR com ampliação e filtro conservador contra ruídos."""
    from PIL import ImageOps, ImageEnhance

    scale = 2
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.25)
    enlarged = gray.resize(
        (gray.width * scale, gray.height * scale),
        Image.Resampling.LANCZOS,
    )

    data = pytesseract.image_to_data(
        enlarged,
        lang="eng",
        config="--oem 3 --psm 11",
        output_type=Output.DICT,
    )

    lines = {}
    count = len(data["text"])

    # Palavras curtas comuns em diálogos ingleses e mangás.
    short_words = {
        "i", "a", "am", "an", "as", "at", "be", "by", "do", "go",
        "he", "if", "in", "is", "it", "me", "my", "no", "of", "oh",
        "ok", "on", "or", "so", "to", "up", "us", "we", "ya", "ah",
        "um", "uh", "hm", "ha", "oi"
    }

    import re

    for i in range(count):
        text = data["text"][i].strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        # Descarta reconhecimentos muito incertos.
        if confidence < 38:
            continue

        normalized = re.sub(r"[^A-Za-z0-9']", "", text)
        if not normalized:
            continue

        word = normalized.lower()

        # Letras soltas e palavras de duas letras exigem mais confiança,
        # mas palavras legítimas comuns continuam sendo aceitas.
        if len(word) == 1 and word not in {"i", "a"} and confidence < 75:
            continue
        if len(word) == 2 and word not in short_words and confidence < 58:
            continue

        x = int(data["left"][i] / scale)
        y = int(data["top"][i] / scale)
        w = max(1, round(data["width"][i] / scale))
        h = max(1, round(data["height"][i] / scale))

        # Ignora marcas minúsculas, sem eliminar palavras pequenas apenas
        # por terem poucas letras.
        if w < 3 or h < 3:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )

        if key not in lines:
            lines[key] = {
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "page": page_number,
            }
        else:
            item = lines[key]
            item["text"] += " " + text
            x1 = min(item["x"], x)
            y1 = min(item["y"], y)
            x2 = max(item["x"] + item["w"], x + w)
            y2 = max(item["y"] + item["h"], y + h)
            item.update(x=x1, y=y1, w=x2-x1, h=y2-y1)

    return sorted(lines.values(), key=lambda item: (item["y"], item["x"]))





def _wrap_text(draw, text, font, max_width):
    """Quebra o texto sem deixar palavras ultrapassarem a largura."""
    rows = []
    current = ""

    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
            continue

        if current:
            rows.append(current)
            current = ""

        piece = ""
        for char in word:
            candidate_piece = piece + char
            if piece and draw.textbbox(
                (0, 0), candidate_piece, font=font
            )[2] > max_width:
                rows.append(piece)
                piece = char
            else:
                piece = candidate_piece
        current = piece

    if current:
        rows.append(current)

    return rows or [""]



def fit_font(draw, text, width, height):
    """Ajusta a fonte para caber dentro da área disponível."""
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"

    width = max(1, int(width))
    height = max(1, int(height))
    max_width = max(1, int(width * 0.90))
    max_height = max(1, int(height * 0.90))
    initial_size = max(6, min(22, int(height * 0.65)))

    for size in range(initial_size, 5, -1):
        font = ImageFont.truetype(font_path, size)
        rows = _wrap_text(draw, text, font, max_width)
        line_height = max(7, int(size * 1.20))
        if rows and len(rows) * line_height <= max_height:
            if all(
                draw.textbbox((0, 0), row, font=font)[2]
                <= max_width
                for row in rows
            ):
                return font, rows, line_height, max_width, max_height

    font = ImageFont.truetype(font_path, 6)
    rows = _wrap_text(draw, text, font, max_width)
    line_height = 7
    return font, rows, line_height, max_width, max_height




def render_page(image, lines, translations):
    """Desenha cada tradução dentro da área detectada, sem ampliar a caixa."""
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)

    # As caixas recebidas do Gemini devem representar a área interna
    # do balão ou da caixa de narração, e não somente as letras originais.
    for line in lines:
        translated = str(translations.get(line["id"], "")).strip()
        if not translated:
            continue

        x, y, w, h = (
            int(line["x"]), int(line["y"]),
            int(line["w"]), int(line["h"])
        )

        x = max(0, min(image.width - 1, x))
        y = max(0, min(image.height - 1, y))
        w = max(1, min(image.width - x, w))
        h = max(1, min(image.height - y, h))

        # Limpa apenas o interior detectado, preservando as bordas externas.
        inset = max(1, min(3, int(min(w, h) * 0.025)))
        left = min(x + inset, image.width - 1)
        top = min(y + inset, image.height - 1)
        right = max(left + 1, min(image.width, x + w - inset))
        bottom = max(top + 1, min(image.height, y + h - inset))
        draw.rectangle((left, top, right, bottom), fill="white")

        area_w = max(1, right - left)
        area_h = max(1, bottom - top)
        font, rows, line_height, max_width, max_height = fit_font(
            draw, translated, area_w, area_h
        )

        # Centraliza o bloco vertical e horizontalmente.
        valid_rows = []
        for row in rows:
            if draw.textbbox((0, 0), row, font=font)[2] <= max_width:
                valid_rows.append(row)

        while valid_rows and len(valid_rows) * line_height > max_height:
            # Nunca desenha linhas que ultrapassem o balão.
            valid_rows.pop()

        if not valid_rows:
            continue

        block_h = len(valid_rows) * line_height
        start_y = top + max(0, (area_h - block_h) // 2)

        for index, row in enumerate(valid_rows):
            bbox = draw.textbbox((0, 0), row, font=font)
            row_w = bbox[2] - bbox[0]
            draw_x = left + max(0, (area_w - row_w) // 2)
            draw_y = start_y + index * line_height

            if draw_y + line_height > bottom:
                break

            draw.text(
                (draw_x, draw_y - bbox[1]),
                row,
                fill="black",
                font=font
            )

    return image



def save_pdf(images, output_path):

    if not images:

        raise RuntimeError("Não há páginas para exportar.")

    images[0].save(

        output_path,

        "PDF",

        save_all=True,

        append_images=images[1:],

        resolution=150,

    )





def save_epub(images, output_path, title):

    book = epub.EpubBook()

    book.set_identifier(str(uuid.uuid4()))

    book.set_title(title)

    book.set_language("pt-BR")

    spine = ["nav"]



    for index, image in enumerate(images, start=1):

        buffer = io.BytesIO()

        image.save(buffer, format="JPEG", quality=92)

        page_image = epub.EpubItem(

            uid=f"page{index}",

            file_name=f"images/page{index}.jpg",

            media_type="image/jpeg",

            content=buffer.getvalue(),

        )

        book.add_item(page_image)

        page = epub.EpubHtml(

            title=f"Página {index}",

            file_name=f"page{index}.xhtml",

            lang="pt-BR",

        )

        page.content = (

            '<html><body style="margin:0;text-align:center">'

            f'<img src="images/page{index}.jpg" style="width:100%;height:auto" />'

            '</body></html>'

        )

        book.add_item(page)

        book.add_item(page_image)

        spine.append(page)



    book.toc = tuple(spine[1:])

    book.spine = spine

    book.add_item(epub.EpubNcx())

    book.add_item(epub.EpubNav())

    epub.write_epub(str(output_path), book)






def analyze_page_with_gemini(image, page_number, context=""):
    """Reconhece e traduz os textos diretamente da imagem de uma página."""
    import base64
    import json
    import os
    import re
    import time
    import urllib.error
    import urllib.request
    from io import BytesIO

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY não está configurada neste terminal."
        )

    model = "gemini-3.5-flash-lite"
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent"
    )

    # Envia a página inteira, não texto extraído pelo Tesseract.
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    prompt = """
Você é um especialista em leitura visual e tradução de mangás.
Analise a página de imagem enviada e execute as duas tarefas:
1. Identifique textos legíveis que devam ser traduzidos, especialmente
   falas dentro de balões e caixas de narração.
2. Traduza cada texto do inglês para português brasileiro natural.

Regras:
- Examine a imagem inteira e preserve a ordem de leitura.
- Não invente palavras que não estejam visíveis.
- Não inclua letras do cenário, efeitos decorativos ou rabiscos como falas,
  a menos que sejam texto legível com significado.
- Preserve nomes de personagens e títulos de obras.
- Preserve exatamente "Demon Slayer: Kimetsu no Yaiba" e "Kimetsu no Yaiba".
- Não traduza logotipos, nomes de editoras nem nomes próprios.
- Cada região deve corresponder a um texto separado.
- Não junte falas de balões diferentes.
- Para cada fala, box deve delimitar a área INTERNA utilizável do balão inteiro, incluindo espaço vazio para a tradução, sem incluir a borda preta, o bico do balão ou desenhos externos. Não delimite somente as letras originais. Para caixas de narração, delimite o interior da caixa. Se não conseguir identificar uma área segura, omita essa região em vez de inventar coordenadas.
- As coordenadas devem ser números normalizados de 0 a 1000:
  [x, y, largura, altura], medidos desde o canto superior esquerdo.
- Se não houver texto traduzível, retorne uma lista vazia.
- Não inclua comentários, Markdown nem texto fora do JSON.

Contexto anterior da história:
""" + (context[-2000:] if context else "(início da história)") + """

Retorne exatamente esta estrutura JSON:
{
  "regions": [
    {
      "text": "texto original legível",
      "translation": "tradução brasileira",
      "box": [120, 220, 260, 130]
    }
  ]
}
"""

    request_body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": encoded
                    }
                }
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    payload = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        endpoint + "?key=" + api_key,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    raw = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(min(5 * (2 ** attempt), 40))
                continue
            raise RuntimeError(
                f"Gemini falhou na página {page_number} "
                f"(HTTP {exc.code}): {detail[:1200]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 4:
                time.sleep(min(5 * (2 ** attempt), 40))
                continue
            raise RuntimeError(
                f"Falha de conexão com Gemini na página {page_number}: {exc}"
            ) from exc

    if raw is None:
        raise RuntimeError(
            f"Gemini não retornou resultado para a página {page_number}."
        )

    try:
        answer = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
        answer = re.sub(r"^```(?:json)?\s*", "", answer, flags=re.I)
        answer = re.sub(r"\s*```$", "", answer)
        parsed = json.loads(answer)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Resposta JSON inválida do Gemini na página {page_number}."
        ) from exc

    width, height = image.size
    lines = []
    translations = {}

    for region in parsed.get("regions", []):
        text = str(region.get("text", "")).strip()
        translation = str(region.get("translation", "")).strip()
        box = region.get("box", [])

        if not text or not translation or not isinstance(box, list) or len(box) != 4:
            continue

        try:
            bx, by, bw, bh = [float(value) for value in box]
        except (ValueError, TypeError):
            continue

        # Coordenadas normalizadas para pixels da imagem original.
        x = max(0, min(width - 1, round(bx * width / 1000)))
        y = max(0, min(height - 1, round(by * height / 1000)))
        w = max(1, min(width - x, round(bw * width / 1000)))
        h = max(1, min(height - y, round(bh * height / 1000)))

        item_id = len(lines) + 1
        lines.append({
            "id": item_id,
            "page": page_number,
            "text": text,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })
        translations[item_id] = translation

    return lines, translations


def process_job(task_id):

    task = get_task(task_id)

    if not task:

        return



    source = Path(task["input_path"])

    output_format = task["output_format"]

    batch_size = int(task.get("batch_size", 1))

    output_path = OUTPUTS / f"{task_id}.{output_format}"



    try:

        update_task(

            task_id,

            status="processing",

            message="Carregando páginas...",

            error="",

        )



        if source.suffix.lower() == ".pdf":

            images = pdf_images(source)

        elif source.suffix.lower() == ".epub":

            images = epub_images(source)

        else:

            raise RuntimeError("Formato de arquivo não suportado.")



        total = len(images)

        update_task(

            task_id,

            total=total,

            current=0,

            percent=0,

            message=f"{total} páginas carregadas. Preparando OCR...",

        )



        rendered = []
        context = ""

        for page_index, image in enumerate(images):
            page_number = page_index + 1

            update_task(
                task_id,
                current=page_index,
                percent=int(page_index * 100 / max(total, 1)),
                message=f"Gemini analisando e traduzindo página {page_number} de {total}...",
            )

            lines, translations = analyze_page_with_gemini(
                image,
                page_number,
                context=context
            )

            rendered.append(render_page(image, lines, translations))

            if lines:
                context = (
                    context + "\n" +
                    "\n".join(
                        f"{line['text']} → {translations.get(line['id'], '')}"
                        for line in lines
                    )
                )[-3500:]

            update_task(
                task_id,
                current=page_number,
                percent=int(page_number * 100 / max(total, 1)),
                message=f"Página concluída: {page_number} de {total}",
            )

        update_task(task_id, message="Montando arquivo final...")

        if output_format == "pdf":

            save_pdf(rendered, output_path)

        else:

            save_epub(rendered, output_path, source.stem)



        update_task(

            task_id,

            status="done",

            current=total,

            total=total,

            percent=100,

            message="Tradução concluída! O arquivo está disponível para baixar.",

            output_path=str(output_path),

        )

        print(f"[LomaMangaTranslator] Concluído: {task['name']}")



    except Exception as exc:

        print(f"[LomaMangaTranslator] Erro em {task_id}: {exc}")

        update_task(

            task_id,

            status="error",

            message="A tradução falhou.",

            error=str(exc),

        )





async def queue_worker():

    while True:

        task_id = await QUEUE.get()

        try:

            task = get_task(task_id)

            if task and task.get("status") == "queued":

                await asyncio.to_thread(process_job, task_id)

        except Exception as exc:

            print(f"Erro no trabalhador da fila: {exc}")

        finally:

            QUEUE.task_done()





@app.on_event("startup")

async def startup():

    global WORKER

    load_saved_tasks()



    # Recoloca primeiro as tarefas mais antigas na fila.

    for task in all_tasks():

        if task.get("status") == "queued":

            await QUEUE.put(task["id"])



    WORKER = asyncio.create_task(queue_worker())





@app.on_event("shutdown")

async def shutdown():

    if WORKER:

        WORKER.cancel()





@app.get("/", response_class=HTMLResponse)

async def home():

    return TEMPLATE.read_text(encoding="utf-8")





@app.post("/api/translate")

async def submit_translation(

    file: UploadFile = File(...),

    output_format: str = Form("pdf"),

    batch_size: int = Form(1),

):

    original_name = Path(file.filename or "manga.pdf").name

    suffix = Path(original_name).suffix.lower()



    if suffix not in (".pdf", ".epub"):

        raise HTTPException(400, "Envie um arquivo PDF ou EPUB.")

    if output_format not in ("pdf", "epub"):

        raise HTTPException(400, "Formato de saída inválido.")

    if batch_size not in (1, 2):

        raise HTTPException(400, "O lote deve ser de 1 ou 2 páginas.")



    task_id = uuid.uuid4().hex

    input_path = UPLOADS / f"{task_id}{suffix}"



    contents = await file.read()

    if not contents:

        raise HTTPException(400, "O arquivo enviado está vazio.")

    input_path.write_bytes(contents)



    task = {

        "id": task_id,

        "name": original_name,

        "input_path": str(input_path),

        "output_format": output_format,

        "batch_size": batch_size,

        "status": "queued",

        "message": "Na fila. A tradução começará quando as tarefas anteriores terminarem.",

        "current": 0,

        "total": 0,

        "percent": 0,

        "error": "",

        "created_at": now(),

        "updated_at": now(),

    }

    save_task(task)

    await QUEUE.put(task_id)



    position = sum(1 for item in all_tasks() if item["status"] == "queued")

    return {

        "ok": True,

        "task": public_task(task),

        "queue_position": position,

    }





@app.get("/api/tasks")

async def list_tasks():

    return {"tasks": [public_task(task) for task in all_tasks()]}





@app.get("/api/progress/{task_id}")

async def progress(task_id: str):

    task = get_task(task_id)

    if not task:

        raise HTTPException(404, "Tarefa não encontrada.")

    return public_task(task)





@app.get("/api/download/{task_id}")

async def download(task_id: str):

    task = get_task(task_id)

    if not task:

        raise HTTPException(404, "Tarefa não encontrada.")

    if task.get("status") != "done":

        raise HTTPException(409, "Esta tradução ainda não terminou.")

    path = Path(task.get("output_path", ""))

    if not path.is_file():

        raise HTTPException(404, "O arquivo final não foi encontrado.")

    return FileResponse(

        path,

        filename=f"{Path(task['name']).stem}_traduzido.{task['output_format']}",

    )





@app.delete("/api/tasks/{task_id}")

async def delete_task(task_id: str):

    task = get_task(task_id)

    if not task:

        raise HTTPException(404, "Tarefa não encontrada.")

    if task.get("status") in ("queued", "processing"):

        raise HTTPException(409, "Não é possível apagar uma tradução ativa.")

    for key in ("input_path", "output_path"):

        value = task.get(key)

        if value:

            try:

                Path(value).unlink(missing_ok=True)

            except Exception:

                pass

    task_file(task_id).unlink(missing_ok=True)

    with TASKS_LOCK:

        TASKS.pop(task_id, None)

    return {"ok": True}
