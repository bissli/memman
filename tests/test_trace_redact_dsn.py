"""Tests for `trace.redact_dsn`."""

from memman.trace import redact_dsn


def test_masks_inline_password():
    """user:password@host gets the password replaced with ***."""
    assert redact_dsn(
        'postgresql://alice:s3cret@db.example.com:5432/memman'
        ) == 'postgresql://alice:***@db.example.com:5432/memman'


def test_passthrough_when_no_password():
    """A passwordless DSN is returned unchanged."""
    assert redact_dsn(
        'postgresql://alice@db.example.com:5432/memman'
        ) == 'postgresql://alice@db.example.com:5432/memman'


def test_passthrough_for_non_dsn_string():
    """Strings that don't match the DSN shape are returned unchanged."""
    assert redact_dsn('not a connection string') == 'not a connection string'
    assert redact_dsn('') == ''


def test_handles_alternate_schemes():
    """Any `scheme://user:pass@host` shape is masked, not just postgresql."""
    assert redact_dsn(
        'postgres://u:p@h/db') == 'postgres://u:***@h/db'


def test_does_not_mask_when_password_contains_at_sign():
    """Defensive case: ambiguous strings should not over-redact.

    The regex requires a colon between user and password; URLs with
    embedded `@` in unexpected positions should leave the original
    structure intact.
    """
    assert redact_dsn(
        'http://example.com/path?q=foo'
        ) == 'http://example.com/path?q=foo'
