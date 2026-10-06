import logging
from datetime import datetime
from ftplib import FTP
from ftplib import Error as FtpError
from pathlib import Path
from tempfile import TemporaryDirectory

from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.xarray import load_one
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data.core.fieldlist import FieldList, MultiFieldList
from earthkit.data.utils.patterns import Pattern
from typing_extensions import override

from cryocast.utils import to_list

logger = logging.getLogger(__name__)


@source_registry.register("ftp")
class FTPSource(Source):
    def __init__(
        self,
        context: Context,
        *,
        url: str,
        passwd: str = "",
        user: str = "anonymous",
        timeout: float = 60.0,
    ) -> None:
        """Initialise the source."""
        self.context = context
        # Parse the FTP URL
        self.ftp_args = {"passwd": passwd, "user": user}
        self.server, self.path_pattern = url.replace("ftp://", "").split("/", 1)
        self.timeout = timeout

    @override
    def execute(self, argument: list[datetime] | GroupOfDates) -> FieldList:
        """Execute the data loading process from an FTP source."""
        # Get list of remote file paths
        remote_paths = {
            date.isoformat(): to_list(
                Pattern(self.path_pattern).substitute(date=date, allow_extra=True)
            )
            for date in argument
        }

        # Connect to the FTP server
        field_lists: list[FieldList] = []
        with (
            TemporaryDirectory() as tmpdir,
            FTP(self.server, timeout=self.timeout) as session,  # noqa: S321
        ):
            base_path = Path(tmpdir)
            session.login(**self.ftp_args)

            # Iterate over remote paths
            for iso_date, remote_path_list in remote_paths.items():
                for remote_path in remote_path_list:
                    directory, filename = remote_path.rsplit("/", 1)
                    local_path = base_path / filename
                    try:
                        # Download the remote file
                        session.cwd(("/" + directory).replace("//", "/"))
                        with local_path.open("wb") as local_file:
                            session.retrbinary(f"RETR {filename}", local_file.write)
                        field_lists.append(
                            load_one("📂", self.context, [iso_date], str(local_path))
                        )
                    except (FtpError, OSError) as exc:
                        logger.warning(
                            "Failed to download from '%s': %s",
                            remote_path,
                            exc,
                        )
                    else:
                        logger.info("Downloaded '%s'.", remote_path)

        # Combine all downloaded files into a MultiFieldList
        return MultiFieldList(field_lists)
