#!/usr/bin/env python3
"""Extrai metadados e resumos de PDFs e HTMLs num ZIP usando modelos multimodais da OpenAI."""

import argparse
import base64
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import fitz
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

PAGES_PER_PART = 20
IMG_DPI = 150
TEXT_CHAR_THRESHOLD = 50
HTML_CHARS_PER_PART = 20000

SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm"}

EXTRACTION_PROMPT = """Você está analisando trechos de um documento (possivelmente do SEI - Sistema Eletrônico de Informações).
Cada trecho vem rotulado como "(texto)" — texto extraído nativamente do documento (PDF ou HTML) — ou "(imagem)" — página renderizada (provavelmente escaneada).
Examine cuidadosamente todo o conteúdo, incluindo cabeçalhos, rodapés, assinaturas e imagens.

Extraia as seguintes informações:
- resumo: resumo objetivo do conteúdo dos trechos (3 a 6 frases, em português).
- data: data principal do documento. Use o formato YYYY-MM-DD quando possível; caso a data esteja parcial, retorne como aparece. null se não houver.
- autores: lista (array) de autores do documento (quem produziu o conteúdo). [] se não houver.
- assinantes: lista (array) de quem assina o documento. [] se não houver.
- entidade: instituição/órgão emissor (string). null se não houver.

Responda APENAS com um objeto JSON válido contendo exatamente essas chaves.
"""

CONSOLIDATE_PROMPT = """Você recebeu metadados e resumos parciais de partes de um mesmo documento, em ordem.
Consolide tudo num único registro coerente, mantendo a ordem cronológica das partes.
Retorne JSON com as chaves: resumo, data, autores, assinantes, entidade.
- resumo: 5 a 10 frases consolidando o conteúdo do documento inteiro.
- demais campos: unifique sem duplicar; use null/[] quando não houver.

Responda APENAS com JSON válido.
"""

GENERAL_PROMPT = """Você recebeu os resumos individuais de vários documentos pertencentes a um mesmo conjunto.
Produza um resumo geral coeso (5 a 10 frases, em português) descrevendo o conjunto como um todo,
identificando temas comuns, propósito do conjunto e quaisquer relações entre os documentos.
Responda apenas com o texto do resumo, sem JSON e sem marcações.
"""


def load_pdf_pages(pdf_path: Path, dpi: int = IMG_DPI) -> list[dict]:
    """Para cada página: tenta extrair texto; se vazia, renderiza como imagem.

    Retorna lista de dicts {numero, kind: "text"|"image", content: str|bytes}.
    """
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[dict] = []
    for i, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if len(text) >= TEXT_CHAR_THRESHOLD:
            pages.append({"numero": i, "kind": "text", "content": text})
        else:
            png = page.get_pixmap(matrix=matrix).tobytes("png")
            pages.append({"numero": i, "kind": "image", "content": png})
    doc.close()
    return pages


def load_html_pages(html_path: Path, max_chars: int = HTML_CHARS_PER_PART) -> list[dict]:
    """Extrai texto de um HTML e divide em 'páginas' por contagem de caracteres.

    Mantém a mesma assinatura de `load_pdf_pages` para reaproveitar o pipeline:
    retorna lista de dicts {numero, kind: "text", content: str}. Quebra em
    parágrafos sempre que possível para evitar cortes no meio de frases.
    """
    soup = BeautifulSoup(html_path.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    # Colapsa linhas em branco repetidas, mantendo separadores de parágrafo.
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    if not text:
        return []

    paragraphs = [p for p in text.split("\n\n") if p]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        plen = len(paragraph) + 2  # "\n\n"
        if current and current_len + plen > max_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [paragraph], plen
        else:
            current.append(paragraph)
            current_len += plen
    if current:
        chunks.append("\n\n".join(current))

    return [
        {"numero": i, "kind": "text", "content": chunk}
        for i, chunk in enumerate(chunks, 1)
    ]


def load_pages(file_path: Path) -> list[dict]:
    """Despacha para o loader correto com base na extensão do arquivo."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_pages(file_path)
    if suffix in {".html", ".htm"}:
        return load_html_pages(file_path)
    raise ValueError(f"Tipo de arquivo não suportado: {file_path.suffix}")


def chunk_pages(pages: list[dict], size: int) -> list[list[dict]]:
    return [pages[i:i + size] for i in range(0, len(pages), size)]


def _image_block(img: bytes) -> dict:
    b64 = base64.b64encode(img).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _build_page_blocks(pages: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    for p in pages:
        if p["kind"] == "text":
            blocks.append({
                "type": "text",
                "text": f"--- Trecho {p['numero']} (texto) ---\n{p['content']}",
            })
        else:
            blocks.append({"type": "text", "text": f"--- Trecho {p['numero']} (imagem) ---"})
            blocks.append(_image_block(p["content"]))
    return blocks


def extract_from_pages(client: OpenAI, model: str, pages: list[dict], context: str) -> dict:

    content: list[dict] = [{"type": "text", "text": f"{EXTRACTION_PROMPT}\nContexto: {context}"}]
    
    content.extend(_build_page_blocks(pages))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def consolidate_parts(client: OpenAI, model: str, parts: list[dict]) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"{CONSOLIDATE_PROMPT}\nPartes:\n{json.dumps(parts, ensure_ascii=False, indent=2)}",
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def general_summary(client: OpenAI, model: str, docs: list[dict]) -> str:
    resumos = [{"nome": d["nome"], "resumo": d.get("resumo", "")} for d in docs]

    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"{GENERAL_PROMPT}\nDocumentos:\n{json.dumps(resumos, ensure_ascii=False, indent=2)}",
        }],
    )

    return resp.choices[0].message.content.strip()


def process_document(client: OpenAI, model: str, file_path: Path, ordem: int, total: int) -> dict:
    tipo = "html" if file_path.suffix.lower() in {".html", ".htm"} else "pdf"
    print(f"  [{ordem}/{total}] Processando ({tipo}): {file_path.name}")
    pages = load_pages(file_path)
    n = len(pages)
    n_text = sum(1 for p in pages if p["kind"] == "text")
    n_img = n - n_text
    unidade = "trechos" if tipo == "html" else "páginas"
    print(f"        {unidade.capitalize()}: {n} ({n_text} texto, {n_img} imagem)")

    entry: dict = {
        "ordem": ordem,
        "nome": file_path.name,
        "tipo": tipo,
        "num_paginas": n,
        "paginas_texto": n_text,
        "paginas_imagem": n_img,
    }

    if n == 0:
        entry.update({"resumo": "", "data": None, "autores": [], "assinantes": [], "entidade": None})
        print("        Aviso: nenhum conteúdo extraído.")
        return entry

    if n <= PAGES_PER_PART:
        info = extract_from_pages(client, model, pages, f"Documento '{file_path.name}' ({tipo}), {n} {unidade}.")
        entry.update(info)
        return entry

    chunks = chunk_pages(pages, PAGES_PER_PART)
    print(f"        Documento grande: dividido em {len(chunks)} partes")
    partes: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"        Parte {i}/{len(chunks)} ({len(chunk)} {unidade})")
        ctx = f"Documento '{file_path.name}' ({tipo}), parte {i} de {len(chunks)}."
        info = extract_from_pages(client, model, chunk, ctx)
        partes.append({"parte": i, **info})
    consolidated = consolidate_parts(client, model, partes)
    entry["partes"] = partes
    entry.update(consolidated)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai resumos e metadados de PDFs e HTMLs num ZIP usando GPT multimodal."
    )
    parser.add_argument("zip_path", type=Path, help="Caminho para o arquivo .zip")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Arquivo JSON de saída (default: <nome-do-zip>.json)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Erro: OPENAI_API_KEY não definida. Configure em .env.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not args.zip_path.exists():
        sys.exit(f"Erro: arquivo não encontrado: {args.zip_path}")

    output_path = args.output or args.zip_path.with_suffix(".json")
    client = OpenAI(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(args.zip_path) as zf:
            arquivos = sorted(
                n for n in zf.namelist()
                if Path(n).suffix.lower() in SUPPORTED_SUFFIXES
            )
            n_pdf = sum(1 for n in arquivos if n.lower().endswith(".pdf"))
            n_html = len(arquivos) - n_pdf
            print(f"\n=== ZIP: {args.zip_path.name} ===")
            print(f"Documentos encontrados: {len(arquivos)} ({n_pdf} PDF, {n_html} HTML)")
            for i, name in enumerate(arquivos, 1):
                print(f"  {i}. {name}")
            print(f"Modelo: {model}\n")
            if not arquivos:
                sys.exit("Nenhum PDF ou HTML encontrado no ZIP.")
            zf.extractall(tmp_path)

        docs = [
            process_document(client, model, tmp_path / name, i, len(arquivos))
            for i, name in enumerate(arquivos, 1)
        ]

    print("\nGerando resumo geral do conjunto...")
    resumo_geral = general_summary(client, model, docs)

    output = {
        "arquivo_zip": args.zip_path.name,
        "total_documentos": len(docs),
        "documentos": docs,
        "resumo_geral": resumo_geral,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaída salva em: {output_path}")


if __name__ == "__main__":
    main()