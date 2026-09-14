"""
PDF Loader using PyMuPDF (fitz).
Extracts text chunks, serializes tables to Markdown, and isolates embedded images.
"""

from typing import List
import os
import fitz  # PyMuPDF
from .base import BaseLoader, IngestedChunk, Modality


class PDFLoader(BaseLoader):
    """
    Extracts text, structured tables (as Markdown), and embedded images from PDFs.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap: float = 0.1,
        image_output_dir: str = "data/extracted_images"
    ):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.image_output_dir = image_output_dir
        os.makedirs(self.image_output_dir, exist_ok=True)

    def load(self, file_path: str) -> List[IngestedChunk]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        filename = os.path.basename(file_path)
        doc = fitz.open(file_path)
        chunks: List[IngestedChunk] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_display_num = page_num + 1

            # 1. Extract and serialize tables to Markdown
            try:
                tables = page.find_tables()
                for t_idx, table in enumerate(tables):
                    table_data = table.extract()
                    if table_data and len(table_data) > 1:
                        md_table = self._format_table_as_markdown(table_data)
                        chunks.append(
                            IngestedChunk(
                                modality=Modality.TABLE,
                                text=md_table,
                                metadata={
                                    "source_file": filename,
                                    "file_path": os.path.abspath(file_path),
                                    "page": page_display_num,
                                    "table_index": t_idx + 1,
                                    "row_count": len(table_data),
                                    "col_count": len(table_data[0]) if table_data else 0
                                }
                            )
                        )
            except Exception:
                pass

            # 2. Extract textual content
            page_text = page.get_text("text").strip()
            if page_text:
                page_chunks = self.chunk_text(page_text, max_tokens=self.max_tokens, overlap=self.overlap)
                for c_idx, c_text in enumerate(page_chunks):
                    chunks.append(
                        IngestedChunk(
                            modality=Modality.PDF,
                            text=c_text,
                            metadata={
                                "source_file": filename,
                                "file_path": os.path.abspath(file_path),
                                "page": page_display_num,
                                "chunk_index": c_idx
                            }
                        )
                    )

            # 3. Extract embedded images
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    img_filename = f"{filename}_p{page_display_num}_img{img_idx + 1}.{image_ext}"
                    img_path = os.path.join(self.image_output_dir, img_filename)

                    with open(img_path, "wb") as f_img:
                        f_img.write(image_bytes)

                    chunks.append(
                        IngestedChunk(
                            modality=Modality.IMAGE,
                            text=f"[Embedded Image from {filename} Page {page_display_num}]",
                            image_path=os.path.abspath(img_path),
                            metadata={
                                "source_file": filename,
                                "page": page_display_num,
                                "image_index": img_idx + 1,
                                "image_path": os.path.abspath(img_path)
                            }
                        )
                    )
                except Exception:
                    continue

        doc.close()
        return chunks

    @staticmethod
    def _format_table_as_markdown(rows: List[List[str]]) -> str:
        """Converts raw rows into a standard GitHub Flavored Markdown table."""
        if not rows:
            return ""

        # Clean cells
        cleaned_rows = [
            [str(cell or "").replace("\n", " ").strip() for cell in row]
            for row in rows
        ]

        header = cleaned_rows[0]
        header_line = "| " + " | ".join(header) + " |"
        separator_line = "| " + " | ".join(["---"] * len(header)) + " |"

        body_lines = []
        for row in cleaned_rows[1:]:
            # Pad row if columns don't match
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            body_lines.append("| " + " | ".join(row[:len(header)]) + " |")

        return "\n".join([header_line, separator_line] + body_lines)
