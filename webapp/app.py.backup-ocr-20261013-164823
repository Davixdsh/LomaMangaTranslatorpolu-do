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



MODEL = "translategemma:4b"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MAX_LINES_PER_REQUEST = 30



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

    data = pytesseract.image_to_data(

        image,

        lang="eng",

        config="--psm 11",

        output_type=Output.DICT,

    )

    lines = {}

    count = len(data["text"])



    for i in range(count):

        text = data["text"][i].strip()

        try:

            confidence = float(data["conf"][i])

        except (ValueError, TypeError):

            confidence = -1



        if not text or confidence < 35:

            continue



        key = (

            data["block_num"][i],

            data["par_num"][i],

            data["line_num"][i],

        )

        x = int(data["left"][i])

        y = int(data["top"][i])

        w = int(data["width"][i])

        h = int(data["height"][i])



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

            x1, y1 = min(item["x"], x), min(item["y"], y)

            x2 = max(item["x"] + item["w"], x + w)

            y2 = max(item["y"] + item["h"], y + h)

            item.update(x=x1, y=y1, w=x2 - x1, h=y2 - y1)



    return sorted(lines.values(), key=lambda x: (x["y"], x["x"]))





def fit_font(draw, text, width, height):

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    if not os.path.exists(font_path):

        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"



    size = max(8, min(22, height))

    while size > 7:

        font = ImageFont.truetype(font_path, size)

        box = draw.textbbox((0, 0), text, font=font)

        if box[2] - box[0] <= max(width * 2, 20) and box[3] - box[1] <= max(height * 2, 16):

            return font

        size -= 1

    return ImageFont.truetype(font_path, 8)





def render_page(image, lines, translations):

    image = image.copy().convert("RGB")

    draw = ImageDraw.Draw(image)



    for line in lines:

        translated = translations.get(line["id"], "").strip()

        if not translated:

            continue



        x, y, w, h = line["x"], line["y"], line["w"], line["h"]

        pad = 3

        # Limpa somente a região identificada pelo OCR.

        draw.rectangle(

            (max(0, x - pad), max(0, y - pad), min(image.width, x + w + pad), min(image.height, y + h + pad)),

            fill="white",

        )

        font = fit_font(draw, translated, w, h)



        # Quebra simples de texto para reduzir sobreposição.

        words = translated.split()

        rows, current = [], ""

        max_width = max(w * 2, 30)

        for word in words:

            candidate = f"{current} {word}".strip()

            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:

                current = candidate

            else:

                if current:

                    rows.append(current)

                current = word

        if current:

            rows.append(current)



        line_height = max(9, int(h * 1.15))

        for index, row in enumerate(rows[:4]):

            draw.text((x, y + index * line_height), row, fill="black", font=font)



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

        for start in range(0, total, batch_size):

            page_batch = images[start:start + batch_size]

            all_lines = []



            for offset, image in enumerate(page_batch):

                page_number = start + offset + 1

                lines = extract_lines(image, page_number)

                for line in lines:

                    line["id"] = len(all_lines) + 1

                    all_lines.append(line)



            update_task(

                task_id,

                message=f"Reconhecendo e traduzindo páginas {start + 1}–{min(start + batch_size, total)}...",

            )



            translations = translate_batch(all_lines, context=context)

            for image in page_batch:

                page_lines = [

                    line for line in all_lines

                    if line["page"] == start + page_batch.index(image) + 1

                ]

                rendered.append(render_page(image, page_lines, translations))



            if all_lines:

                context = (

                    context + "\n" +

                    "\n".join(

                        f"{line['text']} → {translations.get(line['id'], '')}"

                        for line in all_lines

                    )

                )[-3500:]



            current = min(start + len(page_batch), total)

            update_task(

                task_id,

                current=current,

                percent=int(current * 100 / max(total, 1)),

                message=f"Traduzindo páginas: {current} de {total}",

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
