# test_extract_images.py
import sys
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import PictureItem

pdf_path = sys.argv[1]

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0
pipeline_options.do_formula_enrichment = False  # skip — not needed here

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

print("Converting...")
result = converter.convert(pdf_path)

items = [
    item for item, _ in result.document.iterate_items()
    if isinstance(item, PictureItem)
]

print(f"Found {len(items)} picture items")
for i, item in enumerate(items):
    if item.image and item.image.pil_image:
        path = f"/tmp/figure_{i}.png"
        item.image.pil_image.save(path)
        print(f"[{i}] {item.image.pil_image.size} → {path}")
    else:
        print(f"[{i}] no image data")