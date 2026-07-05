#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exoptim.py
#   Author: xyy15926
#   Created: 2024-12-10 08:23:59
#   Updated: 2026-04-03 16:31:46
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
from collections import ChainMap
# from IPython.core.debugger import set_trace

from flagbear.llp.lex import Token
from flagbear.llp.parser import EnvParser
from dirtbear.dflater import callables

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def get_envp(env = None):
    """Get EnvParser with default inner dict.
    """
    calls = {fn: getattr(callables, fn)
             for fn in dir(callables) if callable(getattr(callables, fn))}
    calls.update(callables.consts)
    if env is None:
        return EnvParser(calls)
    return EnvParser(ChainMap(env, calls))


# %%
def compile_deps(
    rules: list[tuple],
    envp: EnvParser = None,
) -> dict[str, list]:
    """Compile the dependences of rules.

    Params:
    ----------------------------
    rules: List of Tuple of string.
      1st: Target identifier to get its dependences.
      follows: Expression string, passed to EnvParser, to determine the
        dependences of the 1st.
    envp: EnvParser to compile string.

    Return:
    ----------------------------
    Dict[key, dependences]
    """
    envp = get_envp() if envp is None else envp

    deps = {}
    for key, *exprs in rules:
        dep = []
        for expr in exprs:
            if isinstance(expr, str) and expr != "":
                for tnode in envp.compile(expr):
                    tk = tnode.val
                    if isinstance(tk, Token) and tk.type == "ID":
                        dep.append(tk.val)
        # In case that some identifiers may rely on multiple rules.
        deps.setdefault(key, set()).update(dep)

    return deps
