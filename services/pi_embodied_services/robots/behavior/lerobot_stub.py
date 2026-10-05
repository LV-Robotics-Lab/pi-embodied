"""A stand-in ``lerobot`` package for the behavior env server's Isaac Sim 6.1 venv.

OmniGibson 3.9 imports its LeRobot dataset wrappers (``omnigibson.envs.lerobot_data_wrapper``:
``LeRobotDataWrapper``, ``LeRobotPlaybackWrapper``, demonstration export and playback) at package
import, so ``import omnigibson`` needs a ``lerobot`` with ``lerobot.configs``, ``lerobot.datasets``
and ``lerobot.utils.constants``. OmniGibson's real dependency, ``lerobot[dataset]`` from
wensi-ai/lerobot@release/b1k, pins a torch / numpy stack that is not Isaac Sim 6.1's, and the env
server never exports or plays back datasets. install_isaac61.sh installs this module as
``lerobot/__init__.py`` instead: every ``lerobot.*`` submodule exists and every name in it is a
class that accepts any constructor arguments and does nothing. Nothing that actually uses LeRobot
works in this venv; install the real fork over it for that.
"""

import importlib.abc
import importlib.machinery
import sys
import types


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.startswith("lerobot."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _Stub(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Finder())
