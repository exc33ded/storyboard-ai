import concurrent.futures

from config import TEXT_MODEL, TEXT_EXTRA_BODY, SEARXNG_URL
from .utils import text_client, _save_to_run_folder, searxng_search
from ddgs import DDGS


def _searxng_results(query: str, max_results: int) -> list:
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in searxng_search(query, max_results=max_results)
    ]


def _ddg_results(query: str, max_results: int) -> list:
    with DDGS() as ddgs:
        return [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in ddgs.text(query, max_results=max_results)
        ]


def _web_search(query: str, max_results: int = 8) -> str:
    """
    Free web search: queries SearXNG (if configured) and DuckDuckGo in
    parallel, then merges the results (deduped by URL) for broader coverage.
    Either engine failing is non-fatal as long as the other returns something.
    """
    engines = {"DuckDuckGo": _ddg_results}
    if SEARXNG_URL:
        engines["SearXNG"] = _searxng_results

    merged, seen = [], set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as pool:
        futures = {name: pool.submit(fn, query, max_results) for name, fn in engines.items()}
        for name, fut in futures.items():
            try:
                for r in fut.result(timeout=30):
                    key = r["url"].rstrip("/")
                    if key and key not in seen:
                        seen.add(key)
                        merged.append(r)
            except Exception as e:
                print(f"  [!] {name} search failed ({e}). Continuing with other engines.")

    return "\n\n".join(f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in merged)


def research_tool_fn(context: str) -> str:
    """
    Performs end-to-end research on the given context via web search + LLM synthesis.

    Args:
        context: The topic to research.
    Returns:
        A detailed research report string.
    """
    print(f"Starting Research for: {context}")

    try:
        search_results = _web_search(context, max_results=10)

        prompt = f"""
        Using the following web search results, write a detailed, well-organized research report about: {context}

        Search Results:
        ---
        {search_results}
        ---

        Include key facts, dates, milestones, and context useful for a documentary/storyboard script.
        """

        response = text_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            extra_body=TEXT_EXTRA_BODY,
        )
        report = response.choices[0].message.content.strip()
        _save_to_run_folder(report, "research_report.md")
        return report

    except Exception as e:
        return f"An error occurred during research: {str(e)}"


def web_grounded_research_tool_fn(context: str) -> str:
    """
    Performs fast, web-grounded research using DuckDuckGo search + LLM summarization.

    Args:
        context: The topic to research.
    Returns:
        A concise, factual summary.
    """
    print(f"Starting Web-Grounded Research for: {context}")

    try:
        search_results = _web_search(context, max_results=6)

        prompt = f"""
        Using the following web search results, provide a concise, factual summary about: {context}
        Include key dates, milestones, and important contextual facts. This will be used as a source
        for a documentary/storyboard script.

        Search Results:
        ---
        {search_results}
        ---
        """

        response = text_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            extra_body=TEXT_EXTRA_BODY,
        )
        report = response.choices[0].message.content.strip()
        _save_to_run_folder(report, "web_research_report.md")
        return report

    except Exception as e:
        return f"An error occurred during web-grounded research: {str(e)}"
