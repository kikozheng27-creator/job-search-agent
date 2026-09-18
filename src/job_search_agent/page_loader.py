"""Fetch a public job-posting URL and return clean source text.

This is an input adapter only. Matching, scoring, and recommendations stay
in JobMatcher. SSL verification is never disabled.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlparse

import requests
import truststore
from bs4 import BeautifulSoup, Tag
from requests import exceptions as request_exceptions

# Same OS-trust-store injection as AIClient. Verification stays enabled.
truststore.inject_into_ssl()

USER_AGENT = (
    "job-search-agent/0.1 "
    "(personal job-search assistant; +https://github.com)"
)
DEFAULT_TIMEOUT_SECONDS = 15
MIN_USABLE_CHARS = 80

NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "canvas",
)
STRUCTURAL_NOISE_TAGS = ("nav", "footer", "aside")
# Application UI. Job descriptions are almost never authored in these tags.
APPLICATION_TAGS = (
    "form",
    "input",
    "textarea",
    "select",
    "option",
    "button",
    "label",
)

JOB_SIGNAL_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bresponsibilit",
        r"\bqualifications?\b",
        r"\brequirements?\b",
        r"\bexperience\b",
        r"\byears?\s+(?:of\s+)?experience\b",
        r"\bsalary\b",
        r"\bcompensation\b",
        r"\bbenefits\b",
        r"\babout (?:the|this) role\b",
        r"\bwhat you(?:'|’)ll do\b",
        r"\bwe are (?:looking|seeking|hiring)\b",
        r"\bjob description\b",
        r"\bfull[-\s]time\b",
        r"\bpart[-\s]time\b",
        r"\brequired skills?\b",
        r"\bpreferred skills?\b",
        r"\bwork authorization\b",
        r"\bsponsorship\b",
        r"\bsecurity clearance\b",
        r"\beducation\b",
        r"\blocation\b",
        r"\bremote\b",
        r"\bhybrid\b",
    )
)
JOB_SIGNAL_PATTERNS = tuple(JOB_SIGNAL_PATTERNS)

PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
PROSE_MIN_CHARS = 50

JS_ONLY_MARKERS = (
    "enable javascript",
    "javascript is required",
    "please enable js",
    "you need to enable javascript",
    "this site requires javascript",
    "checking your browser",
    "just a moment",
)

FetchHtml = Callable[[str], str]


class PageFetchError(ValueError):
    """User-facing failure while fetching or extracting a job page."""


def validate_job_url(url: str) -> str:
    """Accept only http(s) URLs with a host. Returns the stripped URL."""
    candidate = url.strip()

    if not candidate:
        raise PageFetchError("URL is empty.")

    parsed = urlparse(candidate)

    if parsed.scheme not in {"http", "https"}:
        raise PageFetchError(
            f"URL must start with http:// or https://: {url}"
        )

    if not parsed.netloc:
        raise PageFetchError(f"URL is missing a host: {url}")

    return candidate


def fetch_html(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """GET the URL with redirects, a timeout, and TLS verification on."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
            verify=True,
        )
    except request_exceptions.Timeout as error:
        raise PageFetchError(f"Timed out fetching {url}.") from error
    except request_exceptions.SSLError as error:
        raise PageFetchError(
            f"TLS certificate verification failed for {url}."
        ) from error
    except request_exceptions.RequestException as error:
        raise PageFetchError(f"Could not fetch {url}: {error}") from error

    if response.status_code >= 400:
        raise PageFetchError(
            f"HTTP {response.status_code} while fetching {url}."
        )

    content_type = response.headers.get("Content-Type", "")
    body = response.text or ""

    if not _looks_like_html(content_type, body):
        display_type = content_type or "unknown"
        raise PageFetchError(
            f"URL did not return HTML (Content-Type: {display_type})."
        )

    if not body.strip():
        raise PageFetchError(f"Fetched page was empty: {url}")

    return body


def extract_job_text(html: str) -> str:
    """Strip scripts, styles, chrome, and application UI; keep posting text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(NOISE_TAGS + STRUCTURAL_NOISE_TAGS + APPLICATION_TAGS):
        tag.decompose()

    root = _content_root(soup)
    body_text = root.get_text("\n", strip=True)

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title and title.casefold() not in body_text.casefold():
        return normalize_whitespace(f"{title}\n{body_text}")

    return normalize_whitespace(body_text)


def normalize_whitespace(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", compact).strip()


def assert_usable_job_text(text: str, html: str | None = None) -> None:
    compact = re.sub(r"\s+", "", text)

    if len(compact) < MIN_USABLE_CHARS:
        raise PageFetchError(
            "Could not extract usable job-posting text from the page. "
            "It may be empty, blocked, or require JavaScript."
        )

    lowered = text.casefold()
    if any(marker in lowered for marker in JS_ONLY_MARKERS):
        if len(compact) < MIN_USABLE_CHARS * 4:
            raise PageFetchError(
                "The page appears to require JavaScript and did not "
                "contain extractable job-posting text."
            )

    if html is not None and len(html) > 4000 and len(compact) < MIN_USABLE_CHARS:
        raise PageFetchError(
            "The page did not contain extractable job-posting text; "
            "it may require JavaScript."
        )

    if not looks_like_job_posting(text):
        raise PageFetchError(
            "Extracted text does not look like a job posting. "
            "The page may be a navigation shell or require JavaScript "
            "to load the description."
        )


def looks_like_job_posting(text: str) -> bool:
    """Reject contact/nav shells even when they exceed the character floor.

    Passes if there is more than one job-content signal, more than one
    prose line, or one of each. Isolated 'Jobs' nav labels are not enough.
    """
    signals = _job_signal_count(text)
    prose = _prose_line_count(text)

    return signals >= 2 or prose >= 2 or (signals >= 1 and prose >= 1)


def _job_signal_count(text: str) -> int:
    return sum(1 for pattern in JOB_SIGNAL_PATTERNS if pattern.search(text))


def _prose_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _is_prose(line))


def _is_prose(line: str) -> bool:
    stripped = " ".join(line.split())

    if len(stripped) < PROSE_MIN_CHARS:
        return False

    if not re.search(r"[a-zA-Z]{4,}", stripped):
        return False

    if EMAIL_RE.fullmatch(stripped):
        return False

    if len(PHONE_RE.findall(stripped)) >= 1 and len(stripped) < 80:
        return False

    return True


def load_job_description_from_url(
    url: str,
    *,
    fetch: FetchHtml | None = None,
) -> str:
    """Validate, fetch, extract, and reject empty or JS-only pages."""
    getter = fetch or fetch_html
    validated = validate_job_url(url)
    html = getter(validated)
    text = extract_job_text(html)
    assert_usable_job_text(text, html)
    return text


def _looks_like_html(content_type: str, body: str) -> bool:
    lowered_type = content_type.casefold()

    if "json" in lowered_type or lowered_type.startswith("image/"):
        return False

    if "html" in lowered_type or "xhtml" in lowered_type:
        return True

    snippet = body.lstrip()[:200].casefold()
    return snippet.startswith("<!doctype html") or snippet.startswith("<html")


def _content_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "[role=main]"):
        node = soup.select_one(selector)
        if node and node.get_text(strip=True):
            return node

    return soup.body if soup.body else soup
