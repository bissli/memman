"""Domain exception types for memman.

Keeping these in a dedicated module lets `memman.llm` and other
internal packages raise user-facing errors without importing
`click` - the CLI layer catches `ConfigError` and re-wraps it as
`click.ClickException` for clean exit behavior.
"""


class ConfigError(Exception):
    """Memman configuration is invalid or incomplete.

    Typical causes: missing API keys, unknown provider, or a required
    `INSTALLABLE_KEYS` value that the env file does not contain (run
    `memman install`). The message is user-facing (no tracebacks
    needed).
    """


class EmbedFingerprintError(Exception):
    """The active embed provider/model/dim differs from the stored one.

    The stored fingerprint lives in the DB's `meta` table. Means the
    operator changed `MEMMAN_EMBED_PROVIDER` (or related env vars)
    without re-embedding existing data, or the DB has never been
    initialized. Caller must run `memman embed reembed`.
    """


class EmbedCredentialError(Exception):
    """An embed client lacks credentials or required provider config.

    Distinct from `ConfigError`: this surfaces *at embed time* when a
    store is fingerprinted to a provider whose creds are missing in
    the current process. Lets the drain mark the row failed with a
    clean operator-facing error rather than swallowing a generic
    Exception.
    """
