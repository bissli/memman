"""Calibrated AUTO_SEMANTIC_THRESHOLD lookup table.

Each entry is the cosine cutoff for a `(provider, model, surface)`
triple where `surface` is a closed set `{'code', 'claw'}`. Uncalibrated
triples return `None` from `thresholds.resolve` and skip semantic-edge
creation rather than apply a wrong-model default.

Contributors adding a row should propose values backed by measured
retrieval quality on at least one labeled corpus per surface.
"""

_THRESHOLDS: dict[tuple[str, str, str], float] = {
    ('ollama', 'all-minilm', 'claw'): 0.568,
    ('ollama', 'all-minilm', 'code'): 0.558,
    ('ollama', 'mxbai-embed-large', 'claw'): 0.7,
    ('ollama', 'mxbai-embed-large', 'code'): 0.748,
    ('ollama', 'nomic-embed-text', 'claw'): 0.744,
    ('ollama', 'nomic-embed-text', 'code'): 0.738,
    ('ollama', 'snowflake-arctic-embed', 'claw'): 0.752,
    ('ollama', 'snowflake-arctic-embed', 'code'): 0.704,
    ('openrouter', 'baai/bge-large-en-v1.5', 'claw'): 0.712,
    ('openrouter', 'baai/bge-large-en-v1.5', 'code'): 0.738,
    ('openrouter', 'baai/bge-m3', 'claw'): 0.687,
    ('openrouter', 'baai/bge-m3', 'code'): 0.662,
    ('openrouter', 'intfloat/e5-large-v2', 'claw'): 0.845,
    ('openrouter', 'intfloat/e5-large-v2', 'code'): 0.856,
    ('openrouter', 'intfloat/multilingual-e5-large', 'claw'): 0.962,
    ('openrouter', 'intfloat/multilingual-e5-large', 'code'): 0.967,
    ('openrouter', 'openai/text-embedding-3-large', 'claw'): 0.636,
    ('openrouter', 'openai/text-embedding-3-large', 'code'): 0.625,
    ('openrouter', 'openai/text-embedding-3-small', 'claw'): 0.613,
    ('openrouter', 'openai/text-embedding-3-small', 'code'): 0.63,
    ('openrouter', 'qwen/qwen3-embedding-8b', 'claw'): 0.783,
    ('openrouter', 'qwen/qwen3-embedding-8b', 'code'): 0.639,
    ('openrouter', 'sentence-transformers/all-MiniLM-L6-v2', 'claw'): 0.591,
    ('openrouter', 'sentence-transformers/all-MiniLM-L6-v2', 'code'): 0.569,
    ('openrouter', 'sentence-transformers/all-mpnet-base-v2', 'claw'): 0.669,
    ('openrouter', 'sentence-transformers/all-mpnet-base-v2', 'code'): 0.61,
    ('voyage', 'voyage-3', 'claw'): 0.524,
    ('voyage', 'voyage-3', 'code'): 0.591,
    ('voyage', 'voyage-3-large', 'claw'): 0.681,
    ('voyage', 'voyage-3-large', 'code'): 0.777,
    ('voyage', 'voyage-3-lite', 'claw'): 0.497,
    ('voyage', 'voyage-3-lite', 'code'): 0.645,
    ('voyage', 'voyage-code-3', 'claw'): 0.795,
    ('voyage', 'voyage-code-3', 'code'): 0.79,
    ('voyage', 'voyage-finance-2', 'claw'): 0.688,
    ('voyage', 'voyage-finance-2', 'code'): 0.654,
    ('voyage', 'voyage-law-2', 'claw'): 0.611,
    ('voyage', 'voyage-law-2', 'code'): 0.625,
    ('voyage', 'voyage-multilingual-2', 'claw'): 0.68,
    ('voyage', 'voyage-multilingual-2', 'code'): 0.602,
    }
