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
