"""Protocol-shape tests for the Backend swap-method surface.

Every Backend implementation must expose the swap methods that
`embed/swap.py` and the swap CLI call. Swap tests that use one concrete
backend cannot show the Protocol drifting away from the other.
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
    """Both backends must implement every swap verb declared on Backend.
    """

    def test_all_swap_methods_present(self, backend):
        """Every swap verb resolves to a callable on the active backend.

        Mutation: a backend class missing or misnaming one of the swap
            verbs the Protocol declares.
        Oracle: the `SWAP_METHODS` tuple, hand-listed from the Protocol.
        """
        for method in SWAP_METHODS:
            assert callable(getattr(backend, method, None)), (
                f'{method!r} missing on {type(backend).__name__}')
