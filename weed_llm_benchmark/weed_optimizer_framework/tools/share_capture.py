"""Read a public AI share page the way a person's browser would.

ChatGPT/Claude/Gemini share pages are public, but several of them assemble the
conversation in the browser with JavaScript, so a plain HTTP fetch returns only
the page shell (measured: 56 characters for a real Gemini share link). Running a
real browser engine returns the conversation itself (4,665 characters for the
same link).

Firefox is used rather than Chrome: on the dashboard host Chrome issues network
requests that are never answered, while Firefox and urllib work normally.

No sign-in of any kind is performed — only public share URLs are opened, and the
caller is responsible for restricting which hosts may be fetched.
"""
import threading
import time

# a browser is expensive and this runs on the same box as the dashboard
_SLOT = threading.Semaphore(1)
_MAX_CHARS = 200_000
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) "
       "Gecko/20100101 Firefox/133.0")


def available() -> bool:
    """True when a browser engine is installed on this machine."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception:
        return False
    return True


def render(url: str, settle_ms: int = 6000, timeout_ms: int = 60000,
           wait_slot: float = 120.0) -> dict:
    """Open `url` in a headless browser and return its visible text.

    Returns {"ok": True, "title", "text", "chars", "secs"} or
            {"ok": False, "error"}. Never raises.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return {"ok": False, "error": "no browser engine installed (%s)" % type(e).__name__}

    if not _SLOT.acquire(timeout=wait_slot):
        return {"ok": False, "error": "the page reader was busy — try again"}
    t0 = time.time()
    try:
        with sync_playwright() as pw:
            browser = pw.firefox.launch(headless=True)
            try:
                ctx = browser.new_context(user_agent=_UA, locale="en-US",
                                          viewport={"width": 1280, "height": 2000})
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=20000)
                except Exception:
                    pass                      # some pages keep a connection open
                page.wait_for_timeout(settle_ms)
                # the document title is the product's own ("Gemini - direct access
                # to Google AI"); the conversation's title is its first heading
                title = (page.title() or "").strip()[:160]
                heading = ""
                for sel in ("h1", "h2", '[role="heading"]'):
                    try:
                        loc = page.locator(sel).first
                        if loc.count():
                            heading = (loc.inner_text(timeout=2000) or "").strip()
                            if heading:
                                break
                    except Exception:
                        pass
                text = page.inner_text("body") or ""
                final_url = page.url
            finally:
                browser.close()
    except Exception as e:
        return {"ok": False, "secs": round(time.time() - t0, 1),
                "error": "%s: %s" % (type(e).__name__, str(e)[:160])}
    finally:
        _SLOT.release()

    text = _tidy(text)
    # drop the site's navigation chrome by starting at the conversation's heading
    if heading and len(heading) > 3:
        i = text.find(heading)
        if 0 < i < 1500:
            text = text[i:]
    return {"ok": True, "title": (heading[:160] or title), "page_title": title,
            "text": text[:_MAX_CHARS], "chars": len(text),
            "url": final_url, "secs": round(time.time() - t0, 1)}


def _tidy(text: str) -> str:
    """Collapse the blank runs a rendered page produces, keep the line breaks."""
    lines, out, blanks = text.splitlines(), [], 0
    for ln in lines:
        ln = ln.rstrip()
        if ln.strip():
            blanks = 0
            out.append(ln)
        else:
            blanks += 1
            if blanks <= 1:
                out.append("")
    return "\n".join(out).strip()
