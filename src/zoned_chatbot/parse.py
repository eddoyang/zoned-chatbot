from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass
class ParsedDoc:
    text: str
    page_starts: list[tuple[int, int]] # (char_offset, page_no) ascending
    page_count: int

def _converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

def parse(path: Path) -> ParsedDoc:
    doc = _converter().convert(str(path)).document

    parts: list[str] = []
    page_starts: list[tuple[int, int]] = []
    offset = 0
    last_page = None

    for item in doc.texts:
        text = (item.text or "").strip()
        
        if not text:
            continue
    
        page = item.prov[0].page_no if item.prov else last_page
        
        if page != last_page:
            page_starts.append((offset, page))
            last_page = page
        
        parts.append(text)
        offset += len(text) + 2

    return ParsedDoc(
            text ="\n\n".join(parts),
            page_starts=page_starts,
            page_count=len(doc.pages)
    )


def page_at(parsed: ParsedDoc, char_offset: int) -> int | None:
    page = None

    for start, p in parsed.page_starts:
        if start > char_offset:
            break
        
        page = p
    
    return page

