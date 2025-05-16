import asyncio
from playwright.async_api import async_playwright, Playwright, Error

async def test_ws_connection(ws_endpoint: str):
    """Attempts to connect to a Playwright browser instance via WebSocket."""
    print(f"Attempting to connect to: {ws_endpoint}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(ws_endpoint, timeout=15000)
            print(f"Successfully connected to browser: {browser.version}")
            print(f"Number of contexts: {len(browser.contexts)}")
            if browser.contexts:
                print(f"Number of pages in first context: {len(browser.contexts[0].pages)}")
            await browser.close()
            print("Browser connection closed successfully.")
    except Error as e:
        print(f"Playwright Error connecting to {ws_endpoint}: {e}")
    except asyncio.TimeoutError:
        print(f"Timeout connecting to {ws_endpoint} after 15 seconds.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Replace with your actual Chrome/Edge DevTools Protocol WebSocket endpoint
    # Example: ws://127.0.0.1:9222/devtools/browser/YOUR_BROWSER_ID
    # You can get this by launching Chrome with --remote-debugging-port=9222
    # and then navigating to http://127.0.0.1:9222/json/version to find the webSocketDebuggerUrl
    
    target_ws_endpoint = input("Enter the WebSocket endpoint (e.g., ws://127.0.0.1:9222/devtools/browser/...): ").strip()

    if not target_ws_endpoint:
        print("No WebSocket endpoint provided. Exiting.")
    else:
        asyncio.run(test_ws_connection(target_ws_endpoint))