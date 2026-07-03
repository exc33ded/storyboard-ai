import os
import time
import json
import urllib.request
import urllib.parse
from . import utils

def _download_image(image_url: str, name: str) -> str:
    """Downloads an image URL into the run output dir; returns the absolute path."""
    timestamp = int(time.time())
    ext = os.path.splitext(urllib.parse.urlparse(image_url).path)[1]
    if not ext or len(ext) > 5:
        ext = ".jpg"
    filename = f"reference_{timestamp}_{urllib.parse.quote(name)[:60]}{ext}"
    output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename
    req = urllib.request.Request(image_url, headers={'User-Agent': 'StoryboardAI/1.0'})
    with urllib.request.urlopen(req, timeout=20) as response:
        with open(output_path, 'wb') as f:
            f.write(response.read())
    return os.path.abspath(output_path)


def _searxng_image_search(query: str) -> str:
    """Finds a reference image via a self-hosted SearXNG instance."""
    results = utils.searxng_search(query, categories="images", max_results=5)
    for r in results:
        img_url = r.get("img_src", "")
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        if not img_url.startswith("http"):
            continue
        try:
            path = _download_image(img_url, query)
            print(f"Reference Search (SearXNG): Image saved to {path}")
            return path
        except Exception:
            continue  # dead link — try the next result
    raise RuntimeError(f"No downloadable image results for '{query}'")


def reference_search_tool_fn(query: str) -> str:
    """
    Finds and downloads a reference photo for the given subject.
    Uses SearXNG image search if SEARXNG_URL is configured (broader coverage),
    otherwise falls back to the Wikipedia main-page image.

    Args:
        query: The subject to search for (e.g., "5th president of France", "Eiffel Tower").

    Returns:
        The absolute path to the downloaded image, or an error string if not found.
    """
    from config import SEARXNG_URL
    if SEARXNG_URL:
        try:
            return _searxng_image_search(query)
        except Exception as e:
            print(f"  [!] SearXNG image search failed ({e}). Falling back to Wikipedia.")

    try:
        # First, search Wikipedia to get the best matching title
        search_query = urllib.parse.quote(query)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&utf8=&format=json"
        
        req = urllib.request.Request(search_url, headers={'User-Agent': 'StoryboardAI/1.0'})
        with urllib.request.urlopen(req) as response:
            search_data = json.loads(response.read().decode('utf-8'))
            
        search_results = search_data.get('query', {}).get('search', [])
        if not search_results:
            return f"Error: No results found for query '{query}'"
            
        best_title = search_results[0]['title']
        print(f"Reference Search: Found matching Wikipedia article '{best_title}'")
        
        # Next, get the main image for this title
        title_encoded = urllib.parse.quote(best_title)
        image_url_api = f"https://en.wikipedia.org/w/api.php?action=query&titles={title_encoded}&prop=pageimages&format=json&pithumbsize=1000"
        
        req = urllib.request.Request(image_url_api, headers={'User-Agent': 'StoryboardAI/1.0'})
        with urllib.request.urlopen(req) as response:
            image_data = json.loads(response.read().decode('utf-8'))
            
        pages = image_data.get('query', {}).get('pages', {})
        page = next(iter(pages.values()))
        
        if 'thumbnail' not in page:
            return f"Error: No image found for article '{best_title}'"
            
        image_url = page['thumbnail']['source']
        print(f"Reference Search: Downloading image from {image_url}")
        
        # Download the image
        timestamp = int(time.time())
        ext = os.path.splitext(urllib.parse.urlparse(image_url).path)[1]
        if not ext:
            ext = ".jpg"
            
        filename = f"reference_{timestamp}_{urllib.parse.quote(best_title)}{ext}"
        
        # Save to global output dir or current dir
        output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename
        
        req = urllib.request.Request(image_url, headers={'User-Agent': 'StoryboardAI/1.0'})
        with urllib.request.urlopen(req) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
                
        print(f"Reference Search: Image saved to {output_path}")
        return os.path.abspath(output_path)
        
    except Exception as e:
        return f"Error during reference search for '{query}': {str(e)}"
