"""Tests for parsing RSS feeds into FeedEpisode records."""

from __future__ import annotations

from datetime import UTC, datetime

from partio.adapters.feed.rss import parse_feed

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
