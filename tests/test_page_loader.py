from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from job_search_agent.page_loader import (
    PageFetchError,
    extract_job_text,
    fetch_html,
    load_job_description_from_url,
    looks_like_job_posting,
    normalize_whitespace,
    validate_job_url,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def sample_html() -> str:
    return (FIXTURES / "sample_job_page.html").read_text(encoding="utf-8")


def test_extracts_visible_job_posting_text():
    text = extract_job_text(sample_html())

    assert "Biostatistician" in text
    assert "Example Pharma" in text
    assert "Boston, MA" in text
    assert "survival analysis" in text
    assert "We do not provide visa sponsorship." in text
    assert "$90,000 - $110,000" in text
    assert "0-2 years of experience" in text


def test_scripts_and_styles_are_excluded():
    text = extract_job_text(sample_html())

    assert "do-not-extract-this-token" not in text
    assert "color: red" not in text
    assert "Please enable JavaScript to view this site." not in text


def test_navigation_and_footer_are_excluded():
    text = extract_job_text(sample_html())

    assert "Home Careers About" not in text
    assert "Privacy Policy Cookie Settings" not in text


def test_whitespace_is_normalized():
    messy = "<html><body><p>Too     many   spaces</p>\n\n\n\n<p>and   lines</p></body></html>"

    text = extract_job_text(messy)

    assert "  " not in text
    assert "\n\n\n" not in text
    assert text == normalize_whitespace(text)


def test_invalid_url_is_rejected():
    with pytest.raises(PageFetchError, match="http:// or https://"):
        validate_job_url("ftp://example.com/job")

    with pytest.raises(PageFetchError, match="empty"):
        validate_job_url("   ")

    with pytest.raises(PageFetchError, match="host"):
        validate_job_url("https://")


def test_http_error_becomes_page_fetch_error():
    response = Mock()
    response.status_code = 404
    response.headers = {"Content-Type": "text/html"}
    response.text = "<html><body>Not found</body></html>"

    with patch("job_search_agent.page_loader.requests.get", return_value=response):
        with pytest.raises(PageFetchError, match="HTTP 404"):
            fetch_html("https://example.com/missing")


def test_network_timeout_becomes_page_fetch_error():
    with patch(
        "job_search_agent.page_loader.requests.get",
        side_effect=requests.Timeout("timed out"),
    ):
        with pytest.raises(PageFetchError, match="Timed out"):
            fetch_html("https://example.com/job")


def test_connection_failure_becomes_page_fetch_error():
    with patch(
        "job_search_agent.page_loader.requests.get",
        side_effect=requests.ConnectionError("dns failed"),
    ):
        with pytest.raises(PageFetchError, match="Could not fetch"):
            fetch_html("https://example.com/job")


def test_empty_page_is_rejected():
    html = (FIXTURES / "empty_page.html").read_text(encoding="utf-8")

    with pytest.raises(PageFetchError, match="usable job-posting text"):
        load_job_description_from_url(
            "https://example.com/empty",
            fetch=lambda url: html,
        )


def test_javascript_only_page_is_rejected():
    html = (FIXTURES / "js_only_page.html").read_text(encoding="utf-8")

    with pytest.raises(PageFetchError, match="JavaScript"):
        load_job_description_from_url(
            "https://example.com/spa",
            fetch=lambda url: html,
        )


def test_successful_load_uses_injected_fetch_not_the_network():
    text = load_job_description_from_url(
        "https://example.com/jobs/1",
        fetch=lambda url: sample_html(),
    )

    assert "Biostatistician" in text
    assert "We do not provide visa sponsorship." in text


def test_non_html_content_type_is_rejected():
    response = Mock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/json"}
    response.text = '{"job": "Biostatistician"}'

    with patch("job_search_agent.page_loader.requests.get", return_value=response):
        with pytest.raises(PageFetchError, match="did not return HTML"):
            fetch_html("https://example.com/api/job")


def test_tls_verification_stays_enabled():
    with patch("job_search_agent.page_loader.requests.get") as get:
        response = Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/html"}
        response.text = sample_html()
        get.return_value = response

        fetch_html("https://example.com/job")

        assert get.call_args.kwargs["verify"] is True
        assert get.call_args.kwargs["allow_redirects"] is True
        assert get.call_args.kwargs["timeout"] == 15


def test_navigation_contact_shell_is_rejected_even_when_long_enough():
    html = (FIXTURES / "nav_shell_page.html").read_text(encoding="utf-8")
    text = extract_job_text(html)
    compact = "".join(text.split())

    assert len(compact) > 80
    assert not looks_like_job_posting(text)

    with pytest.raises(PageFetchError, match="does not look like a job posting"):
        load_job_description_from_url(
            "https://example.com/jobs?gh_jid=1",
            fetch=lambda url: html,
        )


def test_application_form_fields_are_excluded_from_a_real_job_page():
    html = (FIXTURES / "job_with_application_form.html").read_text(encoding="utf-8")
    text = extract_job_text(html)

    assert "Senior Software Engineer" in text
    assert "Example Labs" in text
    assert "Arlington, VA" in text
    assert "Python services" in text
    assert "5 years of experience" in text
    assert "Secret security clearance" in text
    assert "$140,000 - $170,000" in text
    assert "permanent unrestricted U.S. work authorization" in text

    assert "Create a Job Alert" not in text
    assert "Apply for this job" not in text
    assert "First Name" not in text
    assert "Last Name" not in text
    assert "Resume/CV" not in text
    assert "Social Security Number" not in text
    assert "Voluntary Self-Identification" not in text
    assert "Race & Ethnicity" not in text
    assert "Veteran status" not in text
    assert "Submit application" not in text

    load_job_description_from_url(
        "https://example.com/jobs/1",
        fetch=lambda url: html,
    )
