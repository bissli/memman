"""Protocol-shape tests: every Backend implementation must expose the
swap-method surface used by `embed/swap.py` and the swap CLI.

Existing swap tests use concrete backend instances; without these
shape tests, the Protocol could silently drift away from one backend.
"""


SWAP_METHODS = (
    'swap_lock',
    'swap_prepare',
    'iter_for_swap',
    'write_swap_batch',
    'swap_cutover',
    'swap_abort',
    )


class TestBackendExposesSwapMethods:
    """Both backends must implement every swap verb declared on Backend."""

    def test_all_swap_methods_present(self, backend):
        """Every swap verb resolves to a callable on the active backend.
        """
        for method in SWAP_METHODS:
            assert callable(getattr(backend, method, None)), (
                f'{method!r} missing on {type(backend).__name__}')
