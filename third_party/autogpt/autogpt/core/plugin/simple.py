from importlib import import_module
import importlib.util
from pathlib import Path
import tempfile
import zipfile
from typing import TYPE_CHECKING

from autogpt.core.plugin.base import (
    PluginLocation,
    PluginService,
    PluginStorageFormat,
    PluginStorageRoute,
)

if TYPE_CHECKING:
    from autogpt.core.plugin.base import PluginType


class SimplePluginService(PluginService):
    """Simple plugin service implementation.

    This service provides minimal functionality for loading plugin classes from
    either the local workspace or from installed python packages.
    """

    # Default search locations
    WORKSPACE_PLUGIN_DIR = Path(__file__).resolve().parents[3] / "plugins"
    CORE_PLUGIN_PACKAGE = "autogpt.plugins"
    EXTERNAL_PLUGIN_PACKAGE = "auto_gpt_plugins"
    @staticmethod
    def get_plugin(plugin_location: dict | PluginLocation) -> "PluginType":
        """Get a plugin from a plugin location."""
        if isinstance(plugin_location, dict):
            plugin_location = PluginLocation.parse_obj(plugin_location)
        if plugin_location.storage_format == PluginStorageFormat.WORKSPACE:
            return SimplePluginService.load_from_workspace(
                plugin_location.storage_route
            )
        elif plugin_location.storage_format == PluginStorageFormat.INSTALLED_PACKAGE:
            return SimplePluginService.load_from_installed_package(
                plugin_location.storage_route
            )
        else:
            raise NotImplementedError(
                "Plugin storage format %s is not implemented."
                % plugin_location.storage_format
            )

    ####################################
    # Low-level storage format loaders #
    ####################################
    @staticmethod
    def load_from_file_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from a file, directory, or zip archive.

        The file path may point to a Python file, a directory containing an
        ``__init__.py`` (i.e. a package), or a ``.zip`` archive with python
        sources. The first class defined in the module will be returned.
        """

        path = Path(plugin_route)
        if not path.exists():
            raise FileNotFoundError(f"Plugin path '{plugin_route}' does not exist")

        if path.is_file() and path.suffix == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(tmpdir)
                module_path = next(Path(tmpdir).rglob("*.py"), None)
                if module_path is None:
                    raise ImportError("No python modules found in plugin archive")
                spec = importlib.util.spec_from_file_location(
                    module_path.stem, module_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
        elif path.is_dir():
            init_file = path / "__init__.py"
            if not init_file.exists():
                raise FileNotFoundError(
                    f"Plugin directory '{plugin_route}' missing __init__.py"
                )
            spec = importlib.util.spec_from_file_location(path.name, init_file)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
        else:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type):
                return obj
        raise ImportError("No class found in plugin module")

    @staticmethod
    def load_from_import_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from an import path."""
        module_path, _, class_name = plugin_route.rpartition(".")
        return getattr(import_module(module_path), class_name)

    @staticmethod
    def resolve_name_to_path(
        plugin_route: PluginStorageRoute, path_type: str
    ) -> PluginStorageRoute:
        """Resolve a plugin name to a storage route.

        This searches, in order, the workspace directory, the built-in AutoGPT
        plugin package, and the optional ``auto_gpt_plugins`` package.
        """

        name = plugin_route

        # Workspace search
        workspace = SimplePluginService.WORKSPACE_PLUGIN_DIR
        candidates = [
            workspace / name,
            workspace / f"{name}.py",
            workspace / f"{name}.zip",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # Core plugin package
        core_module = f"{SimplePluginService.CORE_PLUGIN_PACKAGE}.{name}"
        try:
            if importlib.util.find_spec(core_module):
                return core_module
        except ModuleNotFoundError:
            pass

        # External plugin package
        ext_module = f"{SimplePluginService.EXTERNAL_PLUGIN_PACKAGE}.{name}"
        try:
            if importlib.util.find_spec(ext_module):
                return ext_module
        except ModuleNotFoundError:
            pass

        raise FileNotFoundError(f"Plugin '{name}' not found")

    #####################################
    # High-level storage format loaders #
    #####################################

    @staticmethod
    def load_from_workspace(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from the workspace."""
        plugin = SimplePluginService.load_from_file_path(plugin_route)
        return plugin

    @staticmethod
    def load_from_installed_package(plugin_route: PluginStorageRoute) -> "PluginType":
        plugin = SimplePluginService.load_from_import_path(plugin_route)
        return plugin
