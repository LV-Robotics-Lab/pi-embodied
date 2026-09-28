# CaP-X's LIBERO APIs (capx/integrations/franka/libero.py FrankaLiberoApi and libero_privileged.py
# FrankaLiberoPrivilegedApi @53e9966) are this server's high and privileged tiers: the programs
# call get_object_pose, sample_grasp_pose, goto_pose, open_gripper and close_gripper directly
# (manifests/libero.json; env_server.py documents how each differs from CaP-X's). --code-oracle
# prepends this file to object_swap_7.py and object_swap_7_privileged.py for one thing only:
# the programs `import viser.transforms as vtf` and never use it, so without viser in the
# sandbox an empty stand-in lets the import succeed.
import sys
import types

try:
    import viser.transforms  # noqa: F401
except ImportError:
    _viser = types.ModuleType("viser")
    _viser.transforms = types.ModuleType("viser.transforms")
    sys.modules.setdefault("viser", _viser)
    sys.modules.setdefault("viser.transforms", _viser.transforms)
