import requests
import shutil
import os
from typing import Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class DownloadFileInput(BaseModel):
    url: str = Field(description="The URL to download from. Supports http(s):// or file://.")
    output_path: str = Field(description="The local file path to save the downloaded data.")
    chunk_size: Optional[int] = Field(default=8192, description="Optional chunk size in bytes for streaming downloads.")

class DownloadFileOutput(BaseModel):
    output_path: str
    size_bytes: int
    status: str
    error: Optional[str] = None

class DownloadFileTool(BaseTool):
    name: str = "DownloadFile"
    description: str = (
        "Download data from a remote URL (http(s):// or file://) and save it directly to a local file. "
        "This tool is designed for efficiently downloading large HTML, images, or binary files without passing the data through the LLM context or action input. "
        "The agent MUST only provide the URL and the desired output file path. The tool will fetch the data itself. "
        "Do NOT attempt to pass the file contents or large data directly as input!\n\n"
        "**Common usage examples:**\n"
        "- Download a large HTML page:\n"
        "  {\n    'url': 'https://example.com',\n    'output_path': '/tmp/example.html'\n  }\n"
        "- Download an image:\n"
        "  {\n    'url': 'https://example.com/image.jpg',\n    'output_path': '/tmp/image.jpg'\n  }\n"
        "- Copy a local file using file:// URL:\n"
        "  {\n    'url': 'file:///home/user/bigfile.zip',\n    'output_path': '/tmp/bigfile.zip'\n  }\n"
        "This tool will handle the download or copy efficiently, regardless of file size."
    )
    args_schema: type = DownloadFileInput

    def _run(self, url: str, output_path: str, chunk_size: Optional[int] = 8192) -> DownloadFileOutput:
        import requests
        import shutil
        import os
        try:
            if url.startswith("file://"):
                src_path = url[7:]
                with open(src_path, "rb") as src, open(output_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=chunk_size)
                size = os.path.getsize(output_path)
                return DownloadFileOutput(output_path=output_path, size_bytes=size, status="success")
            elif url.startswith("http://") or url.startswith("https://"):
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                size = os.path.getsize(output_path)
                return DownloadFileOutput(output_path=output_path, size_bytes=size, status="success")
            else:
                return DownloadFileOutput(output_path=output_path, size_bytes=0, status="error", error="Only http(s):// and file:// URLs are supported.")
        except Exception as e:
            return DownloadFileOutput(output_path=output_path, size_bytes=0, status="error", error=str(e))

    async def _arun(self, url: str, output_path: str, chunk_size: Optional[int] = 8192) -> DownloadFileOutput:
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, url, output_path, chunk_size)
