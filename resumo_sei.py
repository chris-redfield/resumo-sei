#!/usr/bin/env python3
"""Extrai metadados e resumos de PDFs num ZIP usando modelos multimodais da OpenAI."""

import argparse
import base64
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI

PAGES_PER_PART = 20
IMG_DPI = 150
TEXT_CHAR_THRESHOLD = 50

EXTRACTION_PROMPT = """Você está analisando páginas de um documento (possivelmente do SEI - Sistema Eletrônico de Informações).
Cada página vem rotulada como "(texto)" — texto extraído nativamente do PDF — ou "(imagem)" — página renderizada (provavelmente escaneada).
Examine cuidadosamente todo o conteúdo, incluindo cabeçalhos, rodapés, assinaturas e imagens.

Extraia as seguintes informações:
- resumo: resumo objetivo do conteúdo das páginas (3 a 6 frases, em português).
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
                "text": f"--- Página {p['numero']} (texto) ---\n{p['content']}",
            })
        else:
            blocks.append({"type": "text", "text": f"--- Página {p['numero']} (imagem) ---"})
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


def process_document(client: OpenAI, model: str, pdf_path: Path, ordem: int, total: int) -> dict:
    print(f"  [{ordem}/{total}] Processando: {pdf_path.name}")
    pages = load_pdf_pages(pdf_path)
    n = len(pages)
    n_text = sum(1 for p in pages if p["kind"] == "text")
    n_img = n - n_text
    print(f"        Páginas: {n} ({n_text} texto, {n_img} imagem)")

    entry: dict = {
        "ordem": ordem,
        "nome": pdf_path.name,
        "num_paginas": n,
        "paginas_texto": n_text,
        "paginas_imagem": n_img,
    }

    if n <= PAGES_PER_PART:
        info = extract_from_pages(client, model, pages, f"Documento '{pdf_path.name}', {n} página(s).")
        entry.update(info)
        return entry

    chunks = chunk_pages(pages, PAGES_PER_PART)
    print(f"        Documento grande: dividido em {len(chunks)} partes")
    partes: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"        Parte {i}/{len(chunks)} ({len(chunk)} páginas)")
        ctx = f"Documento '{pdf_path.name}', parte {i} de {len(chunks)}."
        info = extract_from_pages(client, model, chunk, ctx)
        partes.append({"parte": i, **info})
    consolidated = consolidate_parts(client, model, partes)
    entry["partes"] = partes
    entry.update(consolidated)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai resumos e metadados de PDFs num ZIP usando GPT multimodal."
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
            pdfs = sorted(n for n in zf.namelist() if n.lower().endswith(".pdf"))
            print(f"\n=== ZIP: {args.zip_path.name} ===")
            print(f"Documentos encontrados: {len(pdfs)}")
            for i, name in enumerate(pdfs, 1):
                print(f"  {i}. {name}")
            print(f"Modelo: {model}\n")
            if not pdfs:
                sys.exit("Nenhum PDF encontrado no ZIP.")
            zf.extractall(tmp_path)

        docs = [
            process_document(client, model, tmp_path / name, i, len(pdfs))
            for i, name in enumerate(pdfs, 1)
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
