import logging

import torch

log = logging.getLogger(__name__)

MIN_OPEN_FILE_LIMIT = 65536


def patch_open_file_limit() -> None:
    """Raise the soft open-file limit to at least MIN_OPEN_FILE_LIMIT.

    The soft limit can be very low (e.g. 256) in non-terminal contexts, particularly for
    macOS, causing DataLoader worker spawning to fail with EMFILE. Increasing it to a
    large enough number (MIN_OPEN_FILE_LIMIT) circumvents this issue.
    """
    try:
        # Deferred import to avoid issues on Windows
        import resource  # noqa: PLC0415

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < MIN_OPEN_FILE_LIMIT:
            new_soft = (
                MIN_OPEN_FILE_LIMIT
                if hard == resource.RLIM_INFINITY
                else min(MIN_OPEN_FILE_LIMIT, hard)
            )
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            log.debug("Raised open-file limit from %d to %d", soft, new_soft)
    except (ImportError, OSError, ValueError) as exc:
        log.warning("Could not raise open-file limit: %s", exc)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError) as exc:
        log.warning("Could not set torch sharing strategy: %s", exc)
