#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dataloader.py
#   Author: xyy15926
#   Created: 2026-04-20 09:52:58
#   Updated: 2026-04-20 22:09:52
#   Description:
# ---------------------------------------------------------

# %%
import logging
import pandas as pd

# from pathlib import Path
from functools import partial
import io

if __name__ == "__main__":
    from importlib import reload
    from flagbear.slp import databundle

    reload(databundle)

from flagbear.slp.databundle import bundle_cache

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def dumps_to_csv(data: pd.DataFrame):
    """Dump DataFrame into bytes of CSV."""
    # Encode the CSV-str into bytes with utf8 explicitly, though
    # `ZipFile.writestr` seems to encode str into bytes with utf8.
    bytes_ = data.to_csv(
        path_or_buf=None,
        sep=",",
        na_rep="",
        header=True,
        index=True,
        mode="w",
    ).encode("utf8")
    return bytes_


def loads_from_csv(
    bytes_: bytes,
    metadata: dict = None,
):
    """Load DataFrame from bytes of CSV."""
    data = pd.read_csv(
        filepath_or_buffer=io.BytesIO(bytes_),
        sep=",",
        header=0,
        index_col=0,
        encoding="utf8",
    )
    return data


csv_cache = partial(
    bundle_cache,
    dumps_data=dumps_to_csv,
    loads_data=loads_from_csv,
)
pickle_cache = bundle_cache
