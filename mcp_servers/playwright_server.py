# mcp_servers/playwright_server.py

import asyncio
import logging
import json
import sys
import traceback
from typing import List, Optional, Dict, Any, Literal
from uuid import uuid4
import atexit 

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright, 
    BrowserType, 
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    Locator
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][MCPPlaywrightServer] %(message)s')
logger = logging.getLogger(__name__)

mcp_app = FastMCP(
    name="MCPPlaywrightServer",
    version="1.0.7",
    description="MCP server for Playwright with enhanced defensive checks for wsEndpoint connection."
)

# --- Playwright State Management ---
class PlaywrightGlobalState:
    playwright_instance: Optional[Playwright] = None
    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    page: Optional[Page] = None
    element_map: Dict[str, Locator] = {} 
    current_launch_options: Optional[Dict[str, Any]] = None
    connected_via_ws: bool = False
    is_initialized: bool = False
    initialization_lock: Optional[asyncio.Lock] = None

    async def ensure_playwright_initialized(self):
        if not self.initialization_lock:
            self.initialization_lock = asyncio.Lock()
        async with self.initialization_lock:
            if not self.is_initialized or self.playwright_instance is None:
                logger.info("Re-initializing Playwright due to missing instance or not initialized state.")
                try:
                    if self.playwright_instance and hasattr(self.playwright_instance, 'stop') and callable(self.playwright_instance.stop):
                        try: await self.playwright_instance.stop()
                        except Exception as e_stop: logger.warning(f"Error stopping previous Playwright instance: {e_stop}")
                    self.playwright_instance = await async_playwright().start()
                    self.is_initialized = True
                    logger.info("Playwright (re)initialized successfully.")
                except Exception as e:
                    self.is_initialized = False
                    logger.error(f"Failed to initialize Playwright: {e}", exc_info=True)
                    raise RuntimeError(f"Server critical error: Playwright failed to initialize: {e}")
        if not self.playwright_instance or not isinstance(self.playwright_instance, Playwright):
             err_msg = f"Playwright instance is invalid after initialization. Type: {type(self.playwright_instance)}"
             logger.error(err_msg); raise RuntimeError(err_msg)

    async def ensure_page(self) -> Page:
        await self.ensure_playwright_initialized()
        if not self.page or self.page.is_closed():
            if self.context and hasattr(self.context, 'is_closed') and not self.context.is_closed():
                try:
                    self.page = await self.context.new_page()
                    logger.info("Created new page in existing context.")
                except Exception as e:
                    raise ValueError(f"Failed to create new page. Context might be invalid: {e}")
            else:
                raise ValueError("No active browser session to create or use a page. Navigate/connect first.")
        return self.page
    
    def clear_element_map(self):
        self.element_map.clear(); logger.debug("Element map cleared.")

playwright_global_state = PlaywrightGlobalState()

# --- Pydantic Models ---
class ServerPlaywrightNavigateInput(BaseModel): url: Optional[str]=None; launchOptions: Optional[Dict[str,Any]]=None
class ServerPlaywrightNavigateOutput(BaseModel): status: Literal["success","error"]; message: str; current_url: Optional[str]=None
class ServerPlaywrightDescribeElementsInput(BaseModel): pass
class ElementDescription(BaseModel): element_id:str;tag:str;text:Optional[str]=None;attributes:Dict[str,Optional[str]];is_visible:bool
class ServerPlaywrightDescribeElementsOutput(BaseModel): status:Literal["success","error"];message:Optional[str]=None;elements:List[ElementDescription]=[]
class ServerPlaywrightElementActionInput(BaseModel): element_id:str
class ServerPlaywrightFillInput(BaseModel): element_id:str;value:str
class ServerPlaywrightSelectInput(BaseModel): element_id:str;value:str
class ServerPlaywrightActionOutput(BaseModel): status:Literal["success","error"];message:str
class ServerPlaywrightScreenshotInput(BaseModel): name:str;element_id:Optional[str]=None;encoded:bool=True;full_page:Optional[bool]=False
class ServerPlaywrightScreenshotOutput(BaseModel): status:Literal["success","error"];message:Optional[str]=None;base64_data:Optional[str]=None
class ServerPlaywrightEvaluateInput(BaseModel): script:str;element_id:Optional[str]=None
class ServerPlaywrightEvaluateOutput(BaseModel): status:Literal["success","error"];message:Optional[str]=None;result:Optional[Any]=None
class ServerPlaywrightPressInput(BaseModel): element_id: str; key: str = Field(description="Name of the key to press, e.g., 'Enter', 'ArrowDown', 'Control+A'.")
class ServerPlaywrightPressOutput(BaseModel): status:Literal["success","error"];message:str
    
# --- Cleanup Function ---
async def _async_cleanup_playwright():
    logger.info("Attempting asynchronous Playwright cleanup...")
    state = playwright_global_state
    try:
        if state.page and not state.page.is_closed(): await state.page.close(); logger.debug("Page closed.")
    except Exception as e: logger.error(f"Error closing page: {e}")
    try:
        if state.context and hasattr(state.context,'is_closed') and not state.context.is_closed(): await state.context.close(); logger.debug("Context closed.")
    except Exception as e: logger.error(f"Error closing context: {e}")
    try:
        if state.browser and state.browser.is_connected(): await state.browser.close(); logger.debug("Browser closed.")
    except Exception as e: logger.error(f"Error closing browser: {e}")
    try:
        if state.playwright_instance: await state.playwright_instance.stop(); logger.info("Playwright stopped.")
    except Exception as e: logger.error(f"Error stopping Playwright: {e}")
    state.browser=None; state.context=None; state.page=None; state.playwright_instance=None; state.is_initialized=False; state.clear_element_map()
    logger.info("Asynchronous Playwright cleanup finished.")

def cleanup_playwright_sync():
    logger.info("Executing synchronous Playwright cleanup via atexit.")
    try: asyncio.run(_async_cleanup_playwright())
    except RuntimeError as e:
        if "Event loop is closed" in str(e) or "cannot be called when another loop is running" in str(e):
            logger.warning("Main event loop issue, creating new loop for cleanup.")
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            try: loop.run_until_complete(_async_cleanup_playwright())
            finally: loop.close(); asyncio.set_event_loop(None)
        else: logger.error(f"RuntimeError during sync cleanup: {e}")
    except Exception as e: logger.error(f"General error during sync cleanup: {e}")
atexit.register(cleanup_playwright_sync)

# --- Helper to get Locator ---
async def _get_locator_by_id(element_id:str)->Locator:
    state=playwright_global_state; loc=state.element_map.get(element_id)
    if not loc: raise ValueError(f"Element ID {element_id} not found. Re-describe elements.")
    if not state.page or state.page.is_closed() or loc.page != state.page: raise ValueError("Page for element ID invalid. Re-describe elements.")
    return loc

# --- MCP Tool Implementations ---
@mcp_app.tool(name="playwright_navigate")
async def playwright_navigate(input_data: ServerPlaywrightNavigateInput) -> ServerPlaywrightNavigateOutput:
    state = playwright_global_state
    try:
        await state.ensure_playwright_initialized()
    except RuntimeError as e_init:
        logger.error(f"Playwright initialization failed in navigate: {e_init}")
        return ServerPlaywrightNavigateOutput(status="error", message=f"Server Playwright init error: {e_init}")

    ws_endpoint = input_data.launchOptions.get("wsEndpoint") if input_data.launchOptions else None
    
    try:
        should_restart_browser = False
        if state.browser: 
            if not state.browser.is_connected():
                logger.info("Existing browser is not connected. Will restart/reconnect.")
                should_restart_browser = True
            # Only evaluate for restart based on launchOptions if new options are explicitly provided.
            # If input_data.launchOptions is None, we assume the intent is to use the existing browser.
            elif input_data.launchOptions is not None: 
                current_ws_connected = state.connected_via_ws
                # ws_endpoint was derived from input_data.launchOptions at the start of the function
                new_config_is_ws = bool(ws_endpoint) 

                if new_config_is_ws != current_ws_connected:
                    logger.info(f"Switching connection type (WS vs Local). New WS: {new_config_is_ws}, Current WS: {current_ws_connected}. Browser restart required.")
                    should_restart_browser = True
                # Handles cases where:
                # 1. Both are local launches, but options differ.
                # 2. Both are WS connections, but wsEndpoint or other connect options differ.
                elif input_data.launchOptions != state.current_launch_options:
                    logger.info(f"Launch options have changed. New: {input_data.launchOptions}, Current: {state.current_launch_options}. Browser restart required.")
                    should_restart_browser = True
        # If state.browser is None, the subsequent block handles new browser creation.

        if should_restart_browser and state.browser: # state.browser check is for safety, should_restart_browser implies it existed
            logger.info("Closing existing browser for re-configuration or due to disconnection.")
            await state.browser.close()
            state.browser = None
            state.context = None # Will be recreated
            state.page = None    # Will be recreated
            state.clear_element_map()
            # state.connected_via_ws and state.current_launch_options will be reset/set when new browser is established.

        if not state.browser: # This handles initial launch AND cases where browser was closed for restart
            state.clear_element_map()
            
            # Crucially, after a new browser is successfully launched or connected:
            state.current_launch_options = input_data.launchOptions # Store the options that led to this browser instance
            # state.connected_via_ws is also set within the if/else ws_endpoint block.

        if should_restart_browser and state.browser and state.browser.is_connected():
            logger.info("Closing existing browser for re-configuration.")
            await state.browser.close(); state.browser = None; state.context = None; state.page = None; state.clear_element_map()

        if not state.browser or not state.browser.is_connected():
            state.clear_element_map()
            current_options_for_server = input_data.launchOptions.copy() if input_data.launchOptions else {}
            
            if ws_endpoint:
                logger.info(f"Attempting to connect via wsEndpoint (assumed Chromium CDP): {ws_endpoint}")
                # Options for connect_over_cdp should not include wsEndpoint itself or browser_type
                connect_options = {
                    k: v for k, v in current_options_for_server.items() 
                    if k not in ['wsEndpoint', 'browser_type']
                }
                if 'timeout' not in connect_options: 
                    connect_options['timeout'] = 30000 # Default connect timeout

                if not state.playwright_instance or not isinstance(state.playwright_instance, Playwright):
                    err_msg = "Playwright instance invalid before wsEndpoint connect."; 
                    logger.error(err_msg); 
                    raise TypeError(err_msg)

                if not hasattr(state.playwright_instance, "chromium"):
                    err_msg = f"Playwright instance ({type(state.playwright_instance)}) has no 'chromium' attribute."
                    logger.error(err_msg)
                    raise TypeError(err_msg)
                
                chromium_api = state.playwright_instance.chromium
                if not isinstance(chromium_api, BrowserType):
                    err_msg = f"Playwright's 'chromium' attribute is not a BrowserType. Got: {type(chromium_api)}."
                    logger.error(err_msg)
                    raise TypeError(err_msg)

                connect_method_ref = getattr(chromium_api, "connect_over_cdp", None)

                if not callable(connect_method_ref):
                    err_msg = (
                        f"The 'chromium.connect_over_cdp' attribute is not callable. "
                        f"Got type: {type(connect_method_ref)}. Value: {str(connect_method_ref)[:100]}"
                    )
                    logger.error(err_msg)
                    raise TypeError(err_msg)
                
                logger.debug(f"Calling chromium.connect_over_cdp(ws='{ws_endpoint}', opts={connect_options})")
                state.browser = await connect_method_ref(
                    ws_endpoint, **connect_options
                )
                state.connected_via_ws = True
                logger.info("Successfully connected via wsEndpoint.")
                
                # Ensure context and page are from the new browser
# In playwright_navigate, within the `if ws_endpoint:` block, after successful connection:
# ... (browser connection code) ...
                state.browser = await connect_method_ref( # or state.playwright_instance.chromium.connect_over_cdp
                    ws_endpoint, **connect_options
                )
                state.connected_via_ws = True
                logger.info("Successfully connected via wsEndpoint.")

                # Close any pre-existing page that this server state might hold from a previous session/browser.
                if state.page and not state.page.is_closed():
                    try:
                        logger.debug(f"Closing previous server-held page: {state.page.url}")
                        await state.page.close()
                    except Exception as e: 
                        logger.debug(f"Non-critical error closing old server page: {e}")
                
                # Determine the context to use from the connected browser.
                if state.browser.contexts:
                    state.context = state.browser.contexts[0]
                    logger.info("Using first existing context from ws-connected browser.")
                else:
                    state.context = await state.browser.new_context()
                    logger.info("No existing contexts in ws-connected browser; created new context.")
                
                # ALWAYS create a new page in this chosen context for the current operation.
                # This new page will typically be 'about:blank'.
                state.page = await state.context.new_page()
                logger.info(f"Created a new page in the ws-connected browser's context. Initial URL: {state.page.url}")
                
                # The rest of the function (including `page_to_use = await state.ensure_page()`) will then use this new page.
            else: 
                final_launch_opts = current_options_for_server
                if 'headless' not in final_launch_opts: final_launch_opts['headless'] = False; logger.info("Defaulting to headless=false.")
                else: logger.info(f"Using headless={final_launch_opts['headless']} from options.")
                browser_type = final_launch_opts.pop("browser_type", "chromium").lower()
                logger.info(f"Launching new '{browser_type}' with opts: {final_launch_opts}")

                if not state.playwright_instance or not isinstance(state.playwright_instance, Playwright):
                     err_msg = "Playwright instance invalid before launch."; logger.error(err_msg); raise TypeError(err_msg)
                browser_launcher_api = getattr(state.playwright_instance, browser_type, None)
                if not isinstance(browser_launcher_api, BrowserType):
                    err_msg = f"playwright_instance.{browser_type} not BrowserType, is {type(browser_launcher_api)}."; logger.error(err_msg); raise TypeError(err_msg)
                launch_method = getattr(browser_launcher_api, "launch", None)
                if not callable(launch_method):
                    err_msg = f"{browser_type}.launch not callable. Found: {type(launch_method)}."; logger.error(err_msg); raise TypeError(err_msg)
                
                state.browser = await launch_method(**final_launch_opts)
                state.connected_via_ws = False; logger.info("New browser launched.")
                state.context = await state.browser.new_context()
                state.page = await state.context.new_page()
            state.current_launch_options = input_data.launchOptions
        
        page_to_use = await state.ensure_page() 
        state.clear_element_map() 

        if input_data.url:
            logger.info(f"Navigating page '{page_to_use.url}' to '{input_data.url}'")
            await page_to_use.bring_to_front()
            await page_to_use.goto(input_data.url, wait_until="domcontentloaded", timeout=60000)
            logger.info(f"Navigation to {input_data.url} complete.")
        
        current_url_val = page_to_use.url
        final_message = f"Navigated to {current_url_val}" if input_data.url else f"Browser session active. Current page: {current_url_val}"
        if ws_endpoint and not input_data.url: final_message = f"Connected via WebSocket. Current page: {current_url_val}"
        return ServerPlaywrightNavigateOutput(status="success", message=final_message, current_url=current_url_val)

    except PlaywrightTimeoutError as e:
        logger.error(f"Timeout in navigate: {e}", exc_info=False)
        if state.browser and not state.browser.is_connected() and ws_endpoint: # Check specific condition
             logger.info("Cleaning up failed ws connect on timeout."); 
             try: await state.browser.close()
             except Exception: pass # Best effort
             state.browser=None
        return ServerPlaywrightNavigateOutput(status="error", message=f"Timeout: {str(e).splitlines()[0]}")
    except (PlaywrightError, TypeError, AttributeError, ValueError) as e: 
        logger.error(f"Handled error in navigate: {e}", exc_info=True)
        return ServerPlaywrightNavigateOutput(status="error", message=f"Server Error ({type(e).__name__}): {str(e).splitlines()[0]}")
    except Exception as e:
        logger.error(f"Unexpected error in playwright_navigate: {e}", exc_info=True)
        return ServerPlaywrightNavigateOutput(status="error", message=f"Unexpected Server Error: {str(e)}")

@mcp_app.tool(name="playwright_describe_elements")
async def playwright_describe_elements(input_data: ServerPlaywrightDescribeElementsInput) -> ServerPlaywrightDescribeElementsOutput:
    state = playwright_global_state
    try: await state.ensure_playwright_initialized()
    except RuntimeError as e_init: return ServerPlaywrightDescribeElementsOutput(status="error", message=f"Server Playwright init error: {e_init}")
    try:
        page = await state.ensure_page(); state.clear_element_map()
        selectors = ["a[href]", "button:not([disabled])", "input:not([type='hidden']):not([disabled])", "select:not([disabled])", "textarea:not([disabled])","[role='button']:not([aria-disabled='true'])", "[role='link']:not([aria-disabled='true'])", "[role='checkbox']:not([aria-disabled='true'])", "[role='radio']:not([aria-disabled='true'])","[role='menuitem']:not([aria-disabled='true'])", "[role='tab']:not([aria-disabled='true'])", "[role='option']:not([aria-disabled='true'])", "[onclick]", "[tabindex]:not([tabindex='-1'])"]
        combined_selector = ", ".join(selectors); described_elements: List[ElementDescription] = []; temp_id_counter = 0; MAX_ELEMENTS_TO_DESCRIBE = 75
        all_locators = page.locator(combined_selector); count = await all_locators.count(); logger.info(f"Found {count} potential elements. Filtering...")
        for i in range(count):
            if len(described_elements) >= MAX_ELEMENTS_TO_DESCRIBE: break
            loc = all_locators.nth(i)
            try:
                if not await loc.is_visible(timeout=200) or not await loc.is_enabled(timeout=200): continue
                tag_name = (await loc.evaluate("el => el.tagName.toLowerCase()",timeout=200)) or "unknown"; text_content = (await loc.text_content(timeout=500) or "").strip().replace('\n',' ').replace('\r','')[:200]
                attrs_to_get = ['id','name','class','aria-label','placeholder','title','role','type','value','href']; attrs={}
                for attr_name in attrs_to_get:
                    attr_val = await loc.get_attribute(attr_name, timeout=100)
                    if attr_val is not None: attrs[attr_name] = attr_val[:200]
                temp_id = f"auto_el_{temp_id_counter}"; temp_id_counter+=1; state.element_map[temp_id]=loc
                described_elements.append(ElementDescription(element_id=temp_id,tag=tag_name,text=text_content or None,attributes=attrs,is_visible=True,))
            except (PlaywrightTimeoutError,PlaywrightError) as e_el: logger.debug(f"Skipping element due to error/timeout: {e_el}")
        logger.info(f"Described {len(described_elements)} elements.")
        return ServerPlaywrightDescribeElementsOutput(status="success",elements=described_elements,message=f"Described {len(described_elements)} elements.")
    except ValueError as e: return ServerPlaywrightDescribeElementsOutput(status="error",message=str(e))
    except PlaywrightTimeoutError as e: return ServerPlaywrightDescribeElementsOutput(status="error",message=f"Timeout: {str(e).splitlines()[0]}")
    except PlaywrightError as e: return ServerPlaywrightDescribeElementsOutput(status="error",message=f"Playwright Error: {str(e).splitlines()[0]}")
    except Exception as e: logger.error(f"Unexpected error in describe_elements: {e}",exc_info=True); return ServerPlaywrightDescribeElementsOutput(status="error",message=f"Server Error: {str(e)}")

@mcp_app.tool(name="playwright_click")
async def playwright_click(input_data: ServerPlaywrightElementActionInput) -> ServerPlaywrightActionOutput:
    state = playwright_global_state
    try: await state.ensure_playwright_initialized()
    except RuntimeError as e_init: return ServerPlaywrightActionOutput(status="error", message=f"Server Playwright init error: {e_init}")
    try:
        locator = await _get_locator_by_id(input_data.element_id); await locator.click(timeout=30000); page = await state.ensure_page()
        try: await page.wait_for_load_state('domcontentloaded', timeout=5000)
        except PlaywrightTimeoutError: logger.debug("No immediate full navigation after click.")
        state.clear_element_map(); logger.info(f"Clicked element ID: {input_data.element_id}")
        return ServerPlaywrightActionOutput(status="success", message=f"Clicked element ID '{input_data.element_id}'. Page state may have changed.")
    except (ValueError,PlaywrightError,PlaywrightTimeoutError) as e: return ServerPlaywrightActionOutput(status="error",message=str(e).splitlines()[0])
    except Exception as e: logger.error(f"Unexpected error clicking {input_data.element_id}: {e}",exc_info=True); return ServerPlaywrightActionOutput(status="error",message=f"Server Error: {str(e)}")

@mcp_app.tool(name="playwright_fill")
async def playwright_fill(input_data: ServerPlaywrightFillInput) -> ServerPlaywrightActionOutput:
    state = playwright_global_state
    try: await state.ensure_playwright_initialized()
    except RuntimeError as e_init: return ServerPlaywrightActionOutput(status="error", message=f"Server Playwright init error: {e_init}")
    try:
        locator = await _get_locator_by_id(input_data.element_id); await locator.fill(input_data.value, timeout=30000)
        logger.info(f"Filled element ID '{input_data.element_id}'.")
        return ServerPlaywrightActionOutput(status="success", message=f"Filled element ID '{input_data.element_id}'.")
    except (ValueError,PlaywrightError,PlaywrightTimeoutError) as e: return ServerPlaywrightActionOutput(status="error",message=str(e).splitlines()[0])
    except Exception as e: logger.error(f"Unexpected error filling {input_data.element_id}: {e}",exc_info=True); return ServerPlaywrightActionOutput(status="error",message=f"Server Error: {str(e)}")

@mcp_app.tool(name="playwright_hover")
async def playwright_hover(input_data: ServerPlaywrightElementActionInput) -> ServerPlaywrightActionOutput:
    state = playwright_global_state
    try: await state.ensure_playwright_initialized()
    except RuntimeError as e_init: return ServerPlaywrightActionOutput(status="error", message=f"Server Playwright init error: {e_init}")
    try:
        locator = await _get_locator_by_id(input_data.element_id); await locator.hover(timeout=15000)
        logger.info(f"Hovered over element ID: {input_data.element_id}")
        return ServerPlaywrightActionOutput(status="success", message=f"Hovered over element ID '{input_data.element_id}'.")
    except (ValueError,PlaywrightError,PlaywrightTimeoutError) as e: return ServerPlaywrightActionOutput(status="error",message=str(e).splitlines()[0])
    except Exception as e: logger.error(f"Unexpected error hovering {input_data.element_id}: {e}",exc_info=True); return ServerPlaywrightActionOutput(status="error",message=f"Server Error: {str(e)}")

@mcp_app.tool(name="playwright_select")
async def playwright_select(input_data: ServerPlaywrightSelectInput) -> ServerPlaywrightActionOutput:
    state = playwright_global_state
    try: await state.ensure_playwright_initialized()
    except RuntimeError as e_init: return ServerPlaywrightActionOutput(status="error", message=f"Server Playwright init error: {e_init}")
    try:
        locator = await _get_locator_by_id(input_data.element_id); await locator.select_option(value=input_data.value, timeout=15000)
        state.clear_element_map(); logger.info(f"Selected option '{input_data.value}' for element ID '{input_data.element_id}'.")
        return ServerPlaywrightActionOutput(status="success", message=f"Selected option for '{input_data.element_id}'. Page state may have changed.")
    except (ValueError,PlaywrightError,PlaywrightTimeoutError) as e: return ServerPlaywrightActionOutput(status="error",message=str(e).splitlines()[0])
    except Exception as e: logger.error(f"Unexpected error selecting for {input_data.element_id}: {e}",exc_info=True); return ServerPlaywrightActionOutput(status="error",message=f"Server Error: {str(e)}")

@mcp_app.tool(name="playwright_screenshot")
async def playwright_screenshot(input_data: ServerPlaywrightScreenshotInput) -> ServerPlaywrightScreenshotOutput:
    state = playwright_global_state
    try: await state.ensure_playwright_initialized()
    except RuntimeError as e_init: return ServerPlaywrightScreenshotOutput(status="error", message=f"Server Playwright init error: {e_init}")
    try:
        page = await state.ensure_page(); screenshot_options:Dict[str,Any]={"timeout":30000,"type":"png"}
        if input_data.full_page and not input_data.element_id: screenshot_options["full_page"]=True
        screenshot_bytes: bytes
        if input_data.element_id: locator=await _get_locator_by_id(input_data.element_id); screenshot_bytes=await locator.screenshot(**screenshot_options); logger.info(f"SS of element ID: {input_data.element_id}")
        else: screenshot_bytes=await page.screenshot(**screenshot_options); logger.info(f"SS of page '{input_data.name}'.")
        if input_data.encoded: import base64; base64_data=base64.b64encode(screenshot_bytes).decode('utf-8'); data_uri=f"data:image/png;base64,{base64_data}"; return ServerPlaywrightScreenshotOutput(status="success",base64_data=data_uri,message="SS as base64.")
        else: return ServerPlaywrightScreenshotOutput(status="error",message="Binary SS not supported; use encoded=true.")
    except (ValueError,PlaywrightError,PlaywrightTimeoutError) as e: return ServerPlaywrightScreenshotOutput(status="error",message=str(e).splitlines()[0])
    except Exception as e: logger.error(f"Unexpected error taking SS '{input_data.name}': {e}",exc_info=True); return ServerPlaywrightScreenshotOutput(status="error",message=f"Server Error: {str(e)}")

@mcp_app.tool(name="playwright_evaluate")
async def playwright_evaluate(input_data: ServerPlaywrightEvaluateInput) -> ServerPlaywrightEvaluateOutput:
    state = playwright_global_state
    try: await state.ensure_playwright_initialized()
    except RuntimeError as e_init: return ServerPlaywrightEvaluateOutput(status="error", message=f"Server Playwright init error: {e_init}")
    try:
        page = await state.ensure_page(); eval_result: Any
        if input_data.element_id: locator=await _get_locator_by_id(input_data.element_id); eval_result=await locator.evaluate(input_data.script,timeout=30000); logger.info(f"Eval script on el ID '{input_data.element_id}'.")
        else: eval_result=await page.evaluate(input_data.script,timeout=30000); logger.info("Eval script on page.")
        try: json.dumps(eval_result)
        except TypeError: eval_result=str(eval_result); logger.warning(f"Eval result not JSON serializable, to string: {eval_result[:100]}")
        return ServerPlaywrightEvaluateOutput(status="success",result=eval_result,message="Script eval.")
    except (ValueError,PlaywrightError,PlaywrightTimeoutError) as e: return ServerPlaywrightEvaluateOutput(status="error",message=str(e).splitlines()[0])
    except Exception as e: logger.error(f"Unexpected error eval script: {e}",exc_info=True); return ServerPlaywrightEvaluateOutput(status="error",message=f"Server Error: {str(e)}")

@mcp_app.tool(name="playwright_press_key")
async def playwright_press_key(input_data: ServerPlaywrightPressInput) -> ServerPlaywrightActionOutput:
    state = playwright_global_state
    try:
        await state.ensure_playwright_initialized()
    except RuntimeError as e_init:
        return ServerPlaywrightActionOutput(status="error", message=f"Server Playwright init error: {e_init}")
    
    try:
        locator = await _get_locator_by_id(input_data.element_id)
        await locator.press(input_data.key, timeout=15000) # Standard timeout for interactions
        
        page = await state.ensure_page() # Get current page to check for load state
        try:
            # Check if pressing key caused navigation or significant DOM change
            await page.wait_for_load_state('domcontentloaded', timeout=5000) 
            logger.info(f"Page load state 'domcontentloaded' reached after pressing '{input_data.key}' on element ID '{input_data.element_id}'. Element map cleared.")
            state.clear_element_map() 
            message = f"Pressed key '{input_data.key}' on element ID '{input_data.element_id}'. Page state likely changed."
        except PlaywrightTimeoutError:
            logger.debug(f"No immediate full navigation after pressing '{input_data.key}' on element ID '{input_data.element_id}'. Element map NOT cleared automatically.")
            message = f"Pressed key '{input_data.key}' on element ID '{input_data.element_id}'. Page state may not have changed significantly."

        logger.info(f"Pressed key '{input_data.key}' on element ID: {input_data.element_id}")
        return ServerPlaywrightActionOutput(status="success", message=message)
    except (ValueError, PlaywrightError, PlaywrightTimeoutError) as e:
        logger.error(f"Error pressing key '{input_data.key}' on {input_data.element_id}: {e}", exc_info=False)
        return ServerPlaywrightActionOutput(status="error", message=str(e).splitlines()[0])
    except Exception as e:
        logger.error(f"Unexpected error pressing key '{input_data.key}' on {input_data.element_id}: {e}", exc_info=True)
        return ServerPlaywrightActionOutput(status="error", message=f"Server Error: {str(e)}")
        
# --- Run Server ---
if __name__ == "__main__":
    if sys.platform=="win32":
        try:
            if not isinstance(asyncio.get_event_loop_policy(),asyncio.WindowsProactorEventLoopPolicy): 
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()); logger.info("Set WindowsProactorEventLoopPolicy.")
        except Exception as e_pol: logger.warning(f"Could not set WindowsProactorEventLoopPolicy: {e_pol}")
    logger.info("Starting MCP Playwright Server...")
    try:
        mcp_app.run() 
    except KeyboardInterrupt:
        logger.info("MCP Playwright Server shutting down (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"MCP Playwright Server exited with critical error: {e}", exc_info=True)
        sys.exit(1)