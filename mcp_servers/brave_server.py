# mcp_servers/brave_server.py

import asyncio
import logging
import os
import sys
import httpx
import json
from typing import List, Optional, Dict, Any, Literal # Added Literal
from pathlib import Path # Added Path

from pydantic import BaseModel, Field, HttpUrl
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][CustomBraveServer] %(message)s')
logger = logging.getLogger(__name__)

# Ensure API Key is available
BRAVE_API_KEY = None
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from src.config import BRAVE_SEARCH_API_KEY as _cfg_key
    BRAVE_API_KEY = _cfg_key # Directly assign
    if BRAVE_API_KEY:
        logger.info(f"Successfully retrieved BRAVE_SEARCH_API_KEY from src.config (length: {len(BRAVE_API_KEY)}).")
    else:
        logger.warning("BRAVE_SEARCH_API_KEY found in src.config but is empty/None.")
except ImportError:
    logger.critical("CRITICAL: Could not import BRAVE_SEARCH_API_KEY from src.config. Server cannot function.")
    # For direct run, this is fatal. If launched by tool, server will likely fail to provide tools.
except AttributeError:
    logger.critical("CRITICAL: BRAVE_SEARCH_API_KEY not found as an attribute in src.config. Server cannot function.")

if not BRAVE_API_KEY:
    # This log might be redundant now if the except blocks for import/attribute error are hit,
    # but kept for safety in case BRAVE_SEARCH_API_KEY is in config but evaluates to False (e.g. empty string)
    logger.critical("BRAVE_SEARCH_API_KEY is not configured in src.config.py or is empty. Server cannot function effectively.")

BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1"

# --- Initialize FastMCP ---
mcp_app = FastMCP(
    name="CustomBraveServer",
    version="1.1.0", # Incremented version
    description="MCP server providing access to Brave Search API features (Web, News, Video, Image, Summarizer)."
)

# --- Helper for API Calls ---
async def _call_brave_api(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the Brave Search API asynchronously."""
    if not BRAVE_API_KEY:
        raise ValueError("Brave Search API Key is not configured.")
        
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    # Ensure boolean parameters are lowercase strings if API expects that
    for key, value in params.items():
         if isinstance(value, bool):
             params[key] = str(value).lower()

    url = f"{BRAVE_API_BASE_URL}/{endpoint}"
    
    async with httpx.AsyncClient(timeout=30.0) as client: # Increased timeout slightly
        try:
            logger.info(f"Calling Brave API endpoint: {endpoint} with params: {params}")
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status() 
            logger.debug(f"Brave API Response Status: {response.status_code}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Brave API HTTP Error: {e.response.status_code} for {url}. Response: {e.response.text[:500]}")
            raise ValueError(f"Brave API request failed ({e.response.status_code}): {e.response.text[:200]}")
        except httpx.RequestError as e:
            logger.error(f"Brave API Request Error: {e} for {url}")
            raise ConnectionError(f"Could not connect to Brave API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling Brave API for {url}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during Brave API call: {e}")

# --- Common Pydantic Models (used by multiple search types) ---
class MetaUrl(BaseModel): 
    scheme: Optional[str] = None
    netloc: Optional[str] = None
    hostname: Optional[str] = None
    favicon: Optional[str] = None
    path: Optional[str] = None

class Author(BaseModel): 
    name: Optional[str] = None
    url: Optional[str] = None

# --- Pydantic Models for Web Search ---
class WebSearchInput(BaseModel):
    q: str = Field(description="The query string for web search.")
    count: Optional[int] = Field(default=10, ge=1, description="Number of results to return.")
    offset: Optional[int] = Field(default=0, ge=0, description="Offset for pagination.")
    country: Optional[str] = Field(default=None, description="Two-letter country code (e.g., 'US').")
    search_lang: Optional[str] = Field(default=None, description="Search language code (e.g., 'en').")
    ui_lang: Optional[str] = Field(default="en-US", description="User interface language.") 
    spellcheck: Optional[bool] = Field(default=True, description="Enable spellcheck.")
    safesearch: Optional[Literal["off"]] = Field(default="off", description="Safesearch setting.") # Ensure this is off
    # result_filter: Optional[str] = Field(default=None) 
    # freshness: Optional[str] = Field(default=None)

class WebSearchResultItem(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None # Changed HttpUrl to str for flexibility if API returns non-standard URLs
    description: Optional[str] = None
    meta_url: Optional[MetaUrl] = None 
    profile: Optional[Dict[str, Any]] = None # Contains name, url, img etc.
    # Add other fields as needed, like 'page_age', 'language' etc.

class QueryContext(BaseModel): # Example definition, adjust based on actual API response for `query` field
    original: Optional[str] = None
    show_strict_warning: Optional[bool] = None
    altered_query: Optional[str] = None
    # ... other fields from API's query object

class WebResults(BaseModel): # Holds the list of main web results
    results: Optional[List[WebSearchResultItem]] = []
    # ... other fields from API's web object

class Discussions(BaseModel):
    results: Optional[List[Dict[str, Any]]] = [] # Define a more specific item model if structure is known
    # ... other fields from API's discussions object

class Infobox(BaseModel):
    results: Optional[List[Dict[str, Any]]] = [] # Define a more specific item model if structure is known
    # ... other fields from API's infobox object

class FAQ(BaseModel):
    results: Optional[List[Dict[str, Any]]] = [] # Define a more specific item model if structure is known
    # ... other fields from API's faq object

class Locations(BaseModel):
    results: Optional[List[Dict[str, Any]]] = [] # Define a more specific item model if structure is known
    # ... other fields from API's locations object

class MixedResults(BaseModel): # For the 'mixed' section
    main: Optional[List[Dict[str, Any]]] = []
    top: Optional[List[Dict[str, Any]]] = []
    side: Optional[List[Dict[str, Any]]] = []

class SummarizerKeyInfo(BaseModel):
    type: Optional[Literal["summarizer"]] = None
    key: str

class WebSearchOutput(BaseModel):
    query: Optional[QueryContext] = None
    mixed: Optional[MixedResults] = None
    type: Optional[Literal["search"]] = None
    web: Optional[WebResults] = None
    discussions: Optional[Discussions] = None
    infobox: Optional[Infobox] = None
    faq: Optional[FAQ] = None
    locations: Optional[Locations] = None
    summarizer: Optional[SummarizerKeyInfo] = Field(None, description="Contains the key for fetching the summary if available.")
    # Removed 'results: List[WebSearchResultItem]' as 'web.results' should cover it.

# --- Pydantic Models for News Search ---
class NewsSearchInput(BaseModel):
    q: str = Field(description="The search query.")
    count: Optional[int] = Field(default=5, description="Number of results.", ge=1)
    offset: Optional[int] = Field(default=0, description="Pagination offset.", ge=0)
    country: Optional[str] = Field(default=None)
    search_lang: Optional[str] = Field(default=None)
    safesearch: Optional[Literal["off"]] = Field(default="off")

class NewsResultItem(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None # str for flexibility
    description: Optional[str] = None
    age: Optional[str] = None 
    source: Optional[str] = None 
    image_url: Optional[str] = Field(default=None, alias="img") # Example mapping for thumbnail

class NewsSearchOutput(BaseModel):
    results: List[NewsResultItem] = Field(description="List of news search results.")

# --- Pydantic Models for Summarizer ---
class SummarizerInput(BaseModel):
    q: str = Field(description="The query string for summarization.")
    country: Optional[str] = Field(default=None, description="Two-letter country code (e.g., 'US').")
    search_lang: Optional[str] = Field(default=None, description="Search language code (e.g., 'en').")
    # No 'key' here, as the tool function will manage getting it.

class TextLocation(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None

class SummaryThumbnail(BaseModel):  # Using a distinct name for clarity
    src: str
    original: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

class ImageProperties(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    # Add other properties if API provides more

class SummaryImage(BaseModel):
    # Based on Brave docs: "SummaryImage (Image)" and "Image" model containing Thumbnail and ImageProperties
    url: Optional[str] = None # Common for image models
    thumbnail: Optional[SummaryThumbnail] = None 
    properties: Optional[ImageProperties] = None
    # Potentially other fields like 'title', 'source_url' if provided by API for summary images

class SummaryEntity(BaseModel):
    type: Optional[str] = Field(None, description="Type of the entity, e.g., 'person', 'place'")
    value: Optional[str] = Field(None, description="The textual value of the entity")
    image: Optional[SummaryImage] = None
    location: Optional[TextLocation] = Field(None, description="Location of entity in summary text if applicable")

class SummaryMessage(BaseModel):
    text: Optional[str] = Field(None, description="The main summarized text content")
    entities: Optional[List[SummaryEntity]] = Field(None, description="Entities found within the summary text")

class SummaryAnswer(BaseModel):
    text: Optional[str] = Field(None, description="The direct answer if the query was a question")
    location: Optional[TextLocation] = Field(None, description="Location of answer in context if applicable")

class SummaryContext(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    meta_url: Optional[MetaUrl] = None 

class SummaryEnrichments(BaseModel):
    images: Optional[List[SummaryImage]] = None
    answers: Optional[List[SummaryAnswer]] = None
    entities: Optional[List[SummaryEntity]] = Field(None, description="Broader entities related to the summary, not just in-text")
    contexts: Optional[List[SummaryContext]] = Field(None, description="Supporting contexts or sources for the summary")

class SummaryEntityInfo(BaseModel):
    # Details for a specific entity, likely requested if entity_info=1 is used
    name: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = Field(None, description="URL to a page with more info about the entity")
    image: Optional[SummaryImage] = None
    # Other relevant fields

class SummarizerSearchOutput(BaseModel):
    # Based on Brave docs: "The response will include the summarized content or answer based on the key."
    # "SummarizerSearchApiResponse" components: SummaryMessage, SummaryEnrichments, SummaryEntityInfo
    message: Optional[SummaryMessage] = None
    enrichments: Optional[SummaryEnrichments] = None
    entity_info: Optional[List[SummaryEntityInfo]] = Field(None, description="List of entity details, or single if API returns one object") 
    # The API might also return the key or other metadata, add if observed
    # key: Optional[str] = None # The key used for the request, usually not in response body for the final summary.

# --- Pydantic Models for Image Search ---
class ImageSearchInput(BaseModel):
    q: str = Field(description="The image search query.")
    count: Optional[int] = Field(default=10, description="Number of results (1-20).", ge=1, le=20)
    offset: Optional[int] = Field(default=0, description="Pagination offset (0-9).", ge=0, le=9)
    country: Optional[str] = Field(default=None, description="Two-letter country code (e.g., 'US', 'GB').")
    search_lang: Optional[str] = Field(default=None, description="Search language code (e.g., 'en').")
    safesearch: Optional[Literal["off"]] = Field(default="off", description="Safesearch level ('off' for images).") # Corrected Literal values
    # freshness: Optional[Literal["pd", "pw", "pm", "py"]] = Field(default=None) # Freshness usually not for images
    # color: Optional[str] = None # Example: specific to image search
    # size: Optional[str] = None # Example: specific to image search

class ImageResultItem(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None # URL of the page containing the image
    thumbnail_url: Optional[str] = Field(default=None, alias="thumbnail")
    image_url: Optional[str] = Field(default=None, alias="src") # Direct URL to image source
    width: Optional[int] = None
    height: Optional[int] = None
    source: Optional[str] = None # Source website

class ImageSearchOutput(BaseModel):
    results: List[ImageResultItem] = Field(description="List of image search results.")

# --- Pydantic Models for Video Search ---
class VideoSearchInput(BaseModel):
    q: str = Field(description="The query string for video search.")
    count: Optional[int] = Field(default=10, description="Number of results to return.")
    offset: Optional[int] = Field(default=0, description="Number of results to skip.")
    country: Optional[str] = Field(default=None, description="Two-letter country code (e.g., 'US').")
    spellcheck: Optional[bool] = Field(default=True, description="Whether to enable spellcheck.")
    safesearch: Optional[Literal["off"]] = Field(default="off") # Changed from "moderate"
    freshness: Optional[Literal["pd", "pw", "pm", "py"]] = Field(default=None, description="Time range: d=day, w=week, m=month, y=year")

class VideoMetaData(BaseModel):
    duration: Optional[str] = None
    views: Optional[int] = None
    creator: Optional[str] = None
    publisher: Optional[str] = None
    requires_subscription: Optional[bool] = None
    tags: Optional[List[str]] = None
    author: Optional[Author] = None 

class VideoThumbnail(BaseModel): # New model for thumbnail object
    src: str
    original: Optional[str] = None # Original might not always be present

class VideoResultItem(BaseModel):
    type: Literal["video_result"]
    url: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    age: Optional[str] = None
    page_age: Optional[str] = Field(None, alias="page_age")
    video: Optional[VideoMetaData] = None
    meta_url: Optional[MetaUrl] = None 
    thumbnail: Optional[VideoThumbnail] = None 
    # Add other fields if necessary based on API response

class VideoSearchOutput(BaseModel):
    results: List[VideoResultItem] = Field(description="List of video search results.")

# --- MCP Tool Implementations ---

@mcp_app.tool(
    name="brave_web_search",
    description="Performs a standard web search using the Brave Search API."
)
async def brave_web_search_tool(input_params: WebSearchInput) -> WebSearchOutput:
    logger.info(f"Executing web search for query: '{input_params.q}'")
    try:
        api_params = input_params.model_dump(exclude_none=True)
        api_response_json = await _call_brave_api("web/search", api_params)
        logger.info(f"RAW WEB SEARCH RESPONSE from Brave API: {json.dumps(api_response_json)}") # Log the raw response
        # Validate and parse the response using the Pydantic model
        return WebSearchOutput.model_validate(api_response_json)
    except ValueError as e:
        logger.error(f"Error in brave_web_search_tool: {e}", exc_info=True)
        # Re-raise as a generic exception or return an error model if preferred
        # For now, re-raising to see the original error in the client tool
        raise
    except Exception as e:
        logger.error(f"Unexpected error in brave_web_search_tool: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during web search: {e}")

@mcp_app.tool(
    name="brave_news_search",
    description="Performs a news search using the Brave Search API."
)
async def brave_news_search_tool(input_params: NewsSearchInput) -> NewsSearchOutput:
    logger.info(f"Executing news search for query: '{input_params.q}'")
    try:
        api_params = input_params.model_dump(exclude_none=True)
        api_response = await _call_brave_api("news/search", api_params)
        
        results = []
        if api_response and "results" in api_response: 
             for item in api_response["results"]:
                 try:
                     # Map API fields to Pydantic model fields if names differ
                     item_data = item.copy()
                     if "meta_url" in item_data and "hostname" in item_data["meta_url"]:
                         item_data["source"] = item_data["meta_url"]["hostname"]
                     if "thumbnail" in item_data and "src" in item_data["thumbnail"]:
                         item_data["image_url"] = item_data["thumbnail"]["src"]

                     results.append(NewsResultItem.model_validate(item_data))
                 except Exception as e_parse:
                     logger.warning(f"Failed to parse news result item: {item}. Error: {e_parse}")

        return NewsSearchOutput(results=results)

    except Exception as e:
        logger.error(f"Error in brave_news_search_tool: {e}", exc_info=True)
        raise ValueError(f"News search failed: {str(e)}")


@mcp_app.tool(
    name="brave_summarize",
    description="Gets an AI-generated summary for a query using the Brave Search API's Summarizer feature."
)
async def brave_summarize_tool(input_params: SummarizerInput) -> SummarizerSearchOutput:
    logger.info(f"Executing summarizer for query: '{input_params.q}'")
    
    # Step 1: Call web/search with summary=1 to get the summarizer key
    web_search_api_params = {
        "q": input_params.q,
        "summary": 1, # Crucial parameter to request a summary key
    }
    if input_params.country:
        web_search_api_params["country"] = input_params.country
    if input_params.search_lang:
        web_search_api_params["search_lang"] = input_params.search_lang
    
    try:
        logger.info(f"Summarizer Step 1: Calling web/search with params: {web_search_api_params}")
        web_search_response_raw = await _call_brave_api("web/search", web_search_api_params)
        # It's important to validate against the expected structure which includes 'summarizer' field.
        # Assuming _call_brave_api returns a dict that WebSearchOutput can validate.
        web_search_data = WebSearchOutput.model_validate(web_search_response_raw)
        
        if not web_search_data.summarizer or not web_search_data.summarizer.key:
            logger.warning(f"No summarizer key found in web search response for query: '{input_params.q}'")
            # Return a specific message or an empty SummarizerSearchOutput if no key
            # For now, raising an error or returning a specific error structure might be best.
            # Let's return a message within the structure for the tool to report.
            return SummarizerSearchOutput(message=SummaryMessage(text="No summary available for this query or query not eligible for summarization."))

        summarizer_key = web_search_data.summarizer.key
        logger.info(f"Summarizer Step 1 successful. Received key: {summarizer_key[:50]}...")

    except Exception as e:
        logger.error(f"Error in Summarizer Step 1 (web/search) for query '{input_params.q}': {e}", exc_info=True)
        # Propagate a clear error message within the SummarizerSearchOutput or raise MCPToolError
        # For now, let's assume _call_brave_api might raise an HTTP error that gets caught.
        # If it's a validation error from Pydantic, that indicates API response structure mismatch.
        raise MCPToolError(f"Summarizer Step 1 (web/search) failed: {e}")

    # Step 2: Call summarizer/search with the obtained key
    summarizer_api_params = {
        "key": summarizer_key,
        "entity_info": 1 # As per Brave docs example, include entity_info
    }
    
    try:
        logger.info(f"Summarizer Step 2: Calling summarizer/search with key: {summarizer_key[:50]}... and params {summarizer_api_params}")
        summary_response_raw = await _call_brave_api("summarizer/search", summarizer_api_params)
        
        # Validate and parse the final summary response
        final_summary_data = SummarizerSearchOutput.model_validate(summary_response_raw)
        logger.info(f"Summarizer Step 2 successful for query: '{input_params.q}'")
        return final_summary_data
        
    except Exception as e:
        logger.error(f"Error in Summarizer Step 2 (summarizer/search) for query '{input_params.q}' with key '{summarizer_key[:50]}...': {e}", exc_info=True)
        # Propagate a clear error message
        raise MCPToolError(f"Summarizer Step 2 (fetch summary) failed: {e}")

# (Other tool functions like brave_image_search_tool, etc.)
@mcp_app.tool(
    name="brave_image_search",
    description="Performs an image search using the Brave Search API."
)
async def brave_image_search_tool(input_params: ImageSearchInput) -> ImageSearchOutput:
    logger.info(f"Executing image search for query: '{input_params.q}'")
    try:
        api_params = input_params.model_dump(exclude_none=True)
        api_response = await _call_brave_api("images/search", api_params)
        
        results = []
        if api_response and "results" in api_response:
            for item in api_response["results"]:
                try:
                    # Map API fields to model, note 'thumbnail'/'src' mapping might vary
                    item_data = item.get("properties", {}) # Title, URL often here
                    item_data["thumbnail_url"] = item.get("thumbnail", {}).get("src")
                    item_data["image_url"] = item.get("image", {}).get("src") # Check API response structure
                    item_data["source"] = item.get("source")
                    
                    results.append(ImageResultItem.model_validate(item_data))
                except Exception as e_parse:
                     logger.warning(f"Failed to parse image result item: {item}. Error: {e_parse}")
        
        return ImageSearchOutput(results=results)

    except Exception as e:
        logger.error(f"Error in brave_image_search_tool: {e}", exc_info=True)
        raise ValueError(f"Image search failed: {str(e)}")


@mcp_app.tool(
    name="brave_video_search",
    description="Performs a video search using the Brave Search API."
)
async def brave_video_search_tool(input_params: VideoSearchInput) -> VideoSearchOutput:
    logger.info(f"Executing video search for query: '{input_params.q}'")
    try:
        api_params = input_params.model_dump(exclude_none=True)
        api_response = await _call_brave_api("videos/search", api_params)
        
        results = []
        if api_response and "results" in api_response:
            for item in api_response["results"]:
                 try:
                     # Map API fields (structure might vary)
                     item_data = item.copy()
                     item_data["thumbnail"] = VideoThumbnail.model_validate(item.get("thumbnail", {}))
                     item_data["video"] = VideoMetaData.model_validate(item.get("video", {}))
                     
                     results.append(VideoResultItem.model_validate(item_data))
                 except Exception as e_parse:
                     logger.warning(f"Failed to parse video result item: {item}. Error: {e_parse}")
        
        return VideoSearchOutput(results=results)

    except Exception as e:
        logger.error(f"Error in brave_video_search_tool: {e}", exc_info=True)
        raise ValueError(f"Video search failed: {str(e)}")


# --- Run Server ---
if __name__ == "__main__":
    if not BRAVE_API_KEY: # This check now relies on BRAVE_API_KEY being populated from src.config
        print("CRITICAL ERROR: BRAVE_SEARCH_API_KEY is not configured in src.config.py or is empty. Server cannot start.", file=sys.stderr)
        sys.exit(1)
    logger.info(f"Starting Custom FastMCP Brave Search Server with API Key (last 4 chars): ...{BRAVE_API_KEY[-4:] if BRAVE_API_KEY and len(BRAVE_API_KEY) >=4 else 'INVALID_OR_SHORT_KEY'}")
    try:
        mcp_app.run()
    except KeyboardInterrupt:
         logger.info("Custom FastMCP Brave Search Server shutting down.")
    except Exception as e:
         logger.critical(f"Custom FastMCP Brave Search Server exited with critical error: {e}", exc_info=True)
         sys.exit(1)