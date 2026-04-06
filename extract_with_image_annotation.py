import os
from pathlib import Path
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    HTMLFormatOption,
)
load_dotenv()  # Load GEMINI_API_KEY from .env file if present

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = os.environ["GEMINI_GEN_AI_MODEL"]

# Strip <thought>...</thought> blocks Gemini sometimes emits (chain-of-thought leak)
import re
def clean_description(text: str) -> str:
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    return text.strip()

# Gemini exposes an OpenAI-compatible endpoint, so PictureDescriptionApiOptions works as-is
picture_desc_options = PictureDescriptionApiOptions(
    url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    # Explicitly tell Gemini NOT to think out loud
    prompt=(
        "Describe this image in sentences in a single paragraph. "
        "Do not include any reasoning, thoughts, or preamble — only the final description."
    ),
    params={"model": GEMINI_MODEL},
    headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
    timeout=60,
)

# Pipeline options shared by every format that supports picture description
pdf_pipeline_opts = PdfPipelineOptions(
    do_picture_description=True,
    picture_description_options=picture_desc_options,
    enable_remote_services=True,
    generate_picture_images=True,
    images_scale=2,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF:  PdfFormatOption(pipeline_options=pdf_pipeline_opts),
        InputFormat.DOCX: WordFormatOption(),   # images inside DOCX are handled by the PDF sub-pipeline
        InputFormat.HTML: HTMLFormatOption(),
    }
)

# Map file extensions → InputFormat so we can filter gracefully
SUPPORTED = {
    ".pdf":  InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".html": InputFormat.HTML,
    ".htm":  InputFormat.HTML,
}

def to_clean_markdown(conv_res) -> str:
    """
    Export a ConversionResult to clean markdown where:
    - <!-- image --> placeholders are replaced with [IMAGE: <description>]
    - Gemini <thought> blocks are stripped from descriptions
    """
    doc = conv_res.document
    lines = []

    for item, _ in doc.iterate_items():
        from docling_core.types.doc import TextItem, PictureItem, TableItem, SectionHeaderItem

        if isinstance(item, SectionHeaderItem):
            level = "#" * item.level
            lines.append(f"{level} {item.text}\n")

        elif isinstance(item, TextItem):
            lines.append(f"{item.text}\n")

        elif isinstance(item, TableItem):
            # Export table as markdown
            lines.append(item.export_to_markdown())
            lines.append("")

        elif isinstance(item, PictureItem):
            raw_desc = ""
            # Description lives in annotations added by the picture description pipeline
            if item.annotations:
                raw_desc = " ".join(a.text for a in item.annotations if hasattr(a, "text"))
            desc = clean_description(raw_desc) if raw_desc else "No description available."
            lines.append(f"<!-- image -->\n\n[IMAGE: {desc}]\n")

    return "\n".join(lines)


def convert_file(path: str | Path) -> str:
    """Convert a single file and return clean markdown string."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported extension: {p.suffix}")
    result = converter.convert(source=str(p))
    return to_clean_markdown(result)


def convert_directory(directory: str | Path, out_dir: str | Path | None = None):
    """
    Walk *directory*, convert every supported file, and save .md files.
    Output mirrors the input structure under *out_dir* (default: same folder as source).
    """
    src_root = Path(directory)
    dst_root = Path(out_dir) if out_dir else src_root

    for p in src_root.rglob("*"):
        if p.suffix.lower() not in SUPPORTED:
            continue
        print(f"Converting: {p}")
        try:
            md = convert_file(p)
            out = dst_root / p.relative_to(src_root).with_suffix(".md")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
            print(f"  ✓ Saved → {out}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Single file → print to stdout
    md = convert_file("./data/Data Engineering.pdf")
    print(md)