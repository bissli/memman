"""Domain exception types for memman.

Keeping these in a dedicated module lets `memman.llm` and other
internal packages raise user-facing errors without importing
`click` — the CLI layer catches `ConfigError` and re-wraps it as
`click.ClickException` for clean exit behavior.
"""


class ConfigError(Exception):
    """Raised when memman configuration is invalid or incomplete.

    Typical causes: missing API keys, unknown provider, or a model
    override that is not in the ZDR inventory. The message is
    user-facing (no tracebacks needed).
    """


class EmbedFingerprintError(Exception):
    """Raised when the active embed provider/model/dim does not match
    the fingerprint stored in the DB's `meta` table.

    Means the operator changed `MEMMAN_EMBED_PROVIDER` (or related
    env vars) without re-embedding existing data, or the DB has
    never been initialized. Caller must run `memman embed reembed`.
    """
