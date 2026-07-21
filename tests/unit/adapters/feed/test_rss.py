"""Tests for parsing RSS feeds into FeedEpisode records."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest

from partio.adapters.feed.rss import fetch_feed_title, parse_feed, parse_feed_title

_FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test Podcast</title>
<item>
  <title>Episode Two</title>
  <guid>guid-2</guid>
  <pubDate>Tue, 07 Sep 2021 12:00:00 GMT</pubDate>
  <enclosure url="https://cdn.example/2.mp3" type="audio/mpeg" length="200"/>
</item>
<item>
  <title>Episode One</title>
  <guid>guid-1</guid>
  <enclosure url="https://cdn.example/1.mp3" type="audio/mpeg" length="100"/>
</item>
<item>
  <title>Transcript Only</title>
  <guid>guid-0</guid>
  <enclosure url="https://cdn.example/0.pdf" type="application/pdf" length="5"/>
</item>
</channel></rss>"""


def test_parse_feed_keeps_only_audio_entries_in_order():
    """parse_feed() maps audio enclosures to episodes and skips non-audio ones."""
    episodes = parse_feed(_FEED)

    assert [ep.title for ep in episodes] == ["Episode Two", "Episode One"]
    assert episodes[0].audio_url == "https://cdn.example/2.mp3"
    assert episodes[0].guid == "guid-2"
    assert episodes[0].published == datetime(2021, 9, 7, 12, 0, tzinfo=UTC)


def test_parse_feed_tolerates_missing_pubdate():
    """An episode without a pubDate parses with published set to None."""
    episodes = parse_feed(_FEED)

    assert episodes[1].published is None


def test_parse_feed_empty_when_no_audio():
    """A feed with no audio enclosures yields no episodes."""
    feed = b'<?xml version="1.0"?><rss version="2.0"><channel><title>Empty</title></channel></rss>'

    assert parse_feed(feed) == []


# -- enclosure size ----------------------------------------------------------


def test_parse_feed_reads_the_enclosure_length():
    """The declared enclosure length is carried through as size_bytes."""
    episodes = parse_feed(_FEED)

    assert episodes[0].size_bytes == 200
    assert episodes[1].size_bytes == 100


def test_parse_feed_tolerates_a_missing_or_junk_length():
    """A non-numeric or absent length leaves size_bytes as None."""
    feed = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>T</title>
<item><title>No Length</title><guid>g1</guid>
  <enclosure url="https://cdn.example/a.mp3" type="audio/mpeg"/></item>
<item><title>Junk Length</title><guid>g2</guid>
  <enclosure url="https://cdn.example/b.mp3" type="audio/mpeg" length="unknown"/></item>
</channel></rss>"""

    episodes = parse_feed(feed)

    assert [ep.size_bytes for ep in episodes] == [None, None]


# -- feed title --------------------------------------------------------------


def test_parse_feed_title_reads_the_channel_title():
    """The channel title is extracted for labelling a remembered feed."""
    assert parse_feed_title(_FEED) == "Test Podcast"


def test_parse_feed_title_empty_when_untitled():
    """A feed with no title yields an empty string rather than raising."""
    feed = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'

    assert parse_feed_title(feed) == ""


def test_fetch_feed_title_raises_for_status():
    """A non-2xx feed response propagates instead of returning a title."""
    response = httpx.Response(404, content=b"nope", request=httpx.Request("GET", "https://x"))
    with (
        patch("partio.adapters.feed.rss.httpx.get", return_value=response),
        pytest.raises(httpx.HTTPStatusError),
    ):
        fetch_feed_title("https://x")


def test_fetch_feed_title_returns_the_parsed_title():
    """A healthy feed response yields its channel title."""
    response = httpx.Response(200, content=_FEED, request=httpx.Request("GET", "https://x"))
    with patch("partio.adapters.feed.rss.httpx.get", return_value=response):
        assert fetch_feed_title("https://x") == "Test Podcast"
