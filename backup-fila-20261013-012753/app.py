from pathlib import Path
from uuid import uuid4
from threading import Lock
import io
import re
import requests
import pymupdf
import pytesseract
import ebooklib

from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output
from ebooklib import epub
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUT = BASE / "output"
UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

app = FastAPI(title="LomaMangaTranslator")
jobs = {}
lock = Lock()

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "translategemma:4b"

def update(job, **data):
    with lock:
        jobs[job].update(data)

def translate(text):
    r = requests.post(
        OLLAMA,
        json={
            "model": MODEL,
            "stream": False,
            "prompt": (
                "Traduza este diálogo de mangá para português brasileiro. "
                "Preserve o sentido, nomes próprios, personalidade e tom do "
                "personagem. Não explique nem acrescente informações. "
                "Responda somente com a tradução.\n\n" + text
            )
        },
        timeout=600
    )
    r.raise_for_status()
    return r.json()["response"].strip()

def pdf_images(path):
    images = []
    with pymupdf.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=160, alpha=False)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return images

def epub_images(path):
    book = epub.read_epub(str(path))
    by_href = {
        item.get_name(): item
        for item in book.get_items()
    }
    images = []

    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup.find_all("img"):
            src = tag.get("src", "")
            parent = item.get_name().rsplit("/", 1)[0]
            candidates = [
                src,
                str(Path(parent) / src),
                str(Path(item.get_name()).parent / src)
            ]
            image_item = next(
                (by_href[c] for c in candidates if c in by_href),
                None
            )
            if image_item:
                try:
                    images.append(
                        Image.open(io.BytesIO(image_item.get_content())).convert("RGB")
                    )
                except Exception:
                    pass

    if not images:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            try:
                images.append(
                    Image.open(io.BytesIO(item.get_content())).convert("RGB")
                )
            except Exception:
                pass

    return images

def fit_font(draw, text, width, height):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    for size in range(26, 7, -1):
        font = ImageFont.truetype(font_path, size)
        words = text.split()
        lines, current = [], ""

        for word in words:
            test = f"{current} {word}".strip()
            if draw.textbbox((0, 0), test, font=font)[2] <= width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        line_height = size * 1.18
        if lines and len(lines) * line_height <= height:
            return font, lines, line_height

    font = ImageFont.truetype(font_path, 8)
    return font, [text], 9

def process_page(image, page_number, job_id, total_pages):
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config="--psm 11",
        output_type=Output.DICT
    )

    groups = {}
    count = len(data["text"])

    for i in range(count):
        text = data["text"][i].strip()
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if not text or confidence < 35:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )
        groups.setdefault(key, []).append(i)

    draw = ImageDraw.Draw(image)
    W, H = image.size
    lines = []

    for indexes in groups.values():
        indexes.sort(key=lambda i: data["left"][i])
        original = " ".join(data["text"][i].strip() for i in indexes).strip()
        if not original:
            continue

        x1 = min(data["left"][i] for i in indexes)
        y1 = min(data["top"][i] for i in indexes)
        x2 = max(data["left"][i] + data["width"][i] for i in indexes)
        y2 = max(data["top"][i] + data["height"][i] for i in indexes)

        # Ignora fragmentos muito pequenos ou improváveis.
        if x2 - x1 < 8 or y2 - y1 < 5:
            continue

        translated = translate(original)
        if not translated:
            continue

        # Máscara branca pequena. Não detecta perfeitamente os limites
        # do balão; por isso esta é uma primeira versão experimental.
        margin = max(3, int((y2 - y1) * 0.25))
        bx1 = max(0, x1 - margin)
        by1 = max(0, y1 - margin)
        bx2 = min(W - 1, x2 + margin)
        by2 = min(H - 1, y2 + margin)

        draw.rectangle((bx1, by1, bx2, by2), fill="white")

        available_w = max(8, bx2 - bx1 - 4)
        available_h = max(8, by2 - by1 - 2)
        font, wrapped, line_height = fit_font(
            draw, translated, available_w, available_h
        )

        y = by1 + max(0, (available_h - len(wrapped) * line_height) / 2)
        for line in wrapped:
            draw.text((bx1 + 2, y), line, fill="black", font=font)
            y += line_height

        lines.append((original, translated))

    update(
        job_id,
        current=page_number,
        total=total_pages,
        percent=int(page_number * 100 / max(1, total_pages)),
        message=f"Página {page_number} de {total_pages} processada; "
                f"{len(lines)} linhas traduzidas nesta página."
    )
    return image

def save_pdf(images, path):
    doc = pymupdf.open()
    for image in images:
        page = doc.new_page(width=image.width, height=image.height)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        page.insert_image(page.rect, stream=buffer.getvalue())
    doc.save(path)
    doc.close()

def save_epub(images, path, title):
    book = epub.EpubBook()
    book.set_identifier(uuid4().hex)
    book.set_title(title)
    book.set_language("pt-BR")
    chapters = []

    for i, image in enumerate(images, start=1):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_name = f"page_{i:04d}.png"

        image_item = epub.EpubItem(
            uid=f"img{i}",
            file_name=f"images/{image_name}",
            media_type="image/png",
            content=buffer.getvalue()
        )
        book.add_item(image_item)

        chapter = epub.EpubHtml(
            title=f"Página {i}",
            file_name=f"page_{i:04d}.xhtml",
            lang="pt-BR"
        )
        chapter.content = (
            '<html xmlns="http://www.w3.org/1999/xhtml" lang="pt-BR">'
            '<head><meta charset="utf-8"/>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
            '<style>body{margin:0;text-align:center}img{width:100%;height:auto}</style>'
            f'</head><body><img src="../images/{image_name}" alt="Página {i}"/>'
            '</body></html>'
        )
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = tuple(chapters)
    book.spine = ["nav"] + chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)

def process_job(job_id, source, original_name, input_ext, output_format):
    try:
        update(job_id, status="processing", message="Abrindo o mangá...")
        images = (
            pdf_images(source) if input_ext == ".pdf"
            else epub_images(source)
        )

        if not images:
            raise ValueError(
                "Não consegui localizar páginas em imagem. "
                "Este EPUB pode usar outro formato de organização."
            )

        total = len(images)
        processed = []

        for index, image in enumerate(images, start=1):
            update(
                job_id,
                message=f"Reconhecendo textos da página {index} de {total}..."
            )
            processed.append(
                process_page(image, index, job_id, total)
            )

        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(original_name).stem)
        ext = ".pdf" if output_format == "pdf" else ".epub"
        output = OUTPUT / f"{stem}_ptbr{ext}"

        update(job_id, message="Gerando o arquivo final...")
        if output_format == "pdf":
            save_pdf(processed, output)
        else:
            save_epub(processed, output, f"{stem} — Português")

        update(
            job_id,
            status="done",
            percent=100,
            message="Processamento concluído!",
            download=f"/api/download/{output.name}"
        )

    except Exception as e:
        update(job_id, status="error", message=str(e))
    finally:
        source.unlink(missing_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "templates" / "index.html").read_text(encoding="utf-8")

@app.post("/api/translate")
def start(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = Form(...)
):
    name = Path(file.filename or "manga").name
    input_ext = Path(name).suffix.lower()

    if input_ext not in {".pdf", ".epub"}:
        raise HTTPException(400, "Envie um PDF ou EPUB.")
    if output_format not in {"pdf", "epub"}:
        raise HTTPException(400, "Escolha PDF ou EPUB.")

    job_id = uuid4().hex
    source = UPLOADS / f"{job_id}{input_ext}"

    with source.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            out.write(chunk)

    with lock:
        jobs[job_id] = {
            "status": "queued",
            "current": 0,
            "total": 0,
            "percent": 0,
            "message": "Arquivo recebido."
        }

    background_tasks.add_task(
        process_job, job_id, source, name, input_ext, output_format
    )
    return {"job_id": job_id}

@app.get("/api/progress/{job_id}")
def progress(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Processamento não encontrado.")
        return dict(job)

@app.get("/api/download/{filename}")
def download(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(400, "Nome inválido.")
    path = OUTPUT / filename
    if not path.is_file():
        raise HTTPException(404, "Arquivo não encontrado.")
    return FileResponse(path, filename=filename)
