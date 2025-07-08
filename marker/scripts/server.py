import traceback

import click
import os

from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated
import io

from fastapi import FastAPI, Form, File, UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )



class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
            example=None,
        ),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases).",
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines).",
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', 'html', or 'chunks'.",
        ),
    ] = "markdown"
    use_llm: Annotated[
        bool, Field(description="Use an LLM to improve accuracy.")
    ] = False
    format_lines: Annotated[
        bool,
        Field(description="Reformat all lines using a local OCR model for inline math, underlines, and bold."),
    ] = False
    block_correction_prompt: Annotated[
        Optional[str],
        Field(description="Prompt used to correct each block when LLM mode is active."),
    ] = None
    strip_existing_ocr: Annotated[
        bool, Field(description="Strip existing OCR text before processing.")
    ] = False
    redo_inline_math: Annotated[
        bool,
        Field(description="Use an LLM to redo inline math for the highest quality."),
    ] = False
    disable_image_extraction: Annotated[
        bool, Field(description="Disable image extraction from the PDF.")
    ] = False
    debug: Annotated[
        bool, Field(description="Enable debug mode for verbose logging.")
    ] = False
    processors: Annotated[
        Optional[str],
        Field(description="Comma separated list of processors to use."),
    ] = None
    config_json: Annotated[
        Optional[str],
        Field(description="Path to a JSON file with additional configuration."),
    ] = None
    converter_cls: Annotated[
        Optional[str], Field(description="Converter class to use.")
    ] = None
    llm_service: Annotated[
        Optional[str], Field(description="LLM service class if use_llm is enabled.")
    ] = None

async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html", "chunks"], (
        "Invalid output format"
    )
    try:
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
            settings.OUTPUT_ENCODING
        )

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    use_llm: Optional[bool] = Form(default=False),
    format_lines: Optional[bool] = Form(default=False),
    block_correction_prompt: Optional[str] = Form(default=None),
    strip_existing_ocr: Optional[bool] = Form(default=False),
    redo_inline_math: Optional[bool] = Form(default=False),
    disable_image_extraction: Optional[bool] = Form(default=False),
    debug: Optional[bool] = Form(default=False),
    processors: Optional[str] = Form(default=None),
    config_json: Optional[str] = Form(default=None),
    converter_cls: Optional[str] = Form(default=None),
    llm_service: Optional[str] = Form(default=None),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
        use_llm=use_llm,
        format_lines=format_lines,
        block_correction_prompt=block_correction_prompt,
        strip_existing_ocr=strip_existing_ocr,
        redo_inline_math=redo_inline_math,
        disable_image_extraction=disable_image_extraction,
        debug=debug,
        processors=processors,
        config_json=config_json,
        converter_cls=converter_cls,
        llm_service=llm_service,
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )
