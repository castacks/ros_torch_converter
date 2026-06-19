"""Import-FALLBACK installer for the offline ROS stubs.

Installs a MetaPathFinder appended to the END of sys.meta_path, so it is consulted only AFTER
the normal finders have failed. Consequence: a real ROS install (sensor_msgs / rclpy / cv_bridge /
nav_msgs / ... on sys.path or PYTHONPATH) ALWAYS wins — these lightweight stubs are returned only
when the requested module is genuinely absent (the no-ROS venv case). The `offline_stubs/` directory
is intentionally NOT placed on sys.path, so it can never shadow a real package by path ordering.

Bootstrapped from the venv .pth via an `import`-prefixed exec line (see scripts/setup_rtc_env.sh).
`install()` runs on import and is idempotent.
"""
import os
import sys
import importlib.abc
import importlib.util

_STUB_DIR = os.path.dirname(os.path.abspath(__file__))

# Top-level names we provide offline stand-ins for. Submodules (e.g. sensor_msgs.msg, rclpy.node)
# resolve normally once the top-level stub package is located here.
_STUB_TOPLEVEL = {
    "sensor_msgs", "std_msgs", "nav_msgs", "geometry_msgs", "builtin_interfaces",
    "core_interfaces", "grid_map_msgs", "perception_interfaces",
    "cv_bridge", "message_filters", "rclpy",
}


class _StubFallbackFinder(importlib.abc.MetaPathFinder):
    """Resolve a known stub name from _STUB_DIR. Only reached after real finders fail (appended last
    on sys.meta_path), so it never overrides an installed ROS package."""

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] not in _STUB_TOPLEVEL:
            return None
        base = os.path.join(_STUB_DIR, *fullname.split("."))
        pkg_init = os.path.join(base, "__init__.py")
        if os.path.isdir(base) and os.path.exists(pkg_init):
            return importlib.util.spec_from_file_location(
                fullname, pkg_init, submodule_search_locations=[base])
        mod_py = base + ".py"
        if os.path.exists(mod_py):
            return importlib.util.spec_from_file_location(fullname, mod_py)
        return None


def install():
    if not any(isinstance(f, _StubFallbackFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFallbackFinder())  # append => lowest priority


install()
