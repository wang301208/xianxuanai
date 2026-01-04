import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.capability import register_module
from backend.capability.runtime_loader import RuntimeModuleManager
from modules.interface import ModuleInterface


def test_runtime_module_manager_load_and_unload():
    # Register a dummy module
    name = "dummy_runtime_test"
    register_module(name, lambda: {"name": name})

    mgr = RuntimeModuleManager()
    mod = mgr.load(name)
    assert mod == {"name": name}
    assert name in mgr.loaded_modules()

    mgr.unload(name)
    assert name not in mgr.loaded_modules()


def test_runtime_module_manager_update(tmp_path):
    name = "dummy_runtime_test2"
    register_module(name, lambda: {"name": name})
    mgr = RuntimeModuleManager()

    mgr.update([name])
    assert name in mgr.loaded_modules()

    mgr.update([])
    assert name not in mgr.loaded_modules()


def test_runtime_module_manager_resolves_dependencies():
    class Dep(ModuleInterface):
        initialized = False
        def initialize(self) -> None:
            self.initialized = True
        def shutdown(self) -> None:
            self.initialized = False

    class Main(ModuleInterface):
        dependencies = ["dep"]
        initialized = False
        def initialize(self) -> None:
            self.initialized = True
        def shutdown(self) -> None:
            self.initialized = False

    register_module("dep", Dep)
    register_module("main", Main)

    mgr = RuntimeModuleManager()
    main_mod = mgr.load("main")
    dep_mod = mgr.load("dep")  # should return already loaded dependency

    # both main and dependency should be loaded and initialized
    assert "dep" in mgr.loaded_modules()
    assert "main" in mgr.loaded_modules()
    assert main_mod.initialized and dep_mod.initialized


def test_runtime_module_manager_emits_module_used_events() -> None:
    from modules.events import InMemoryEventBus

    name = "dummy_runtime_used_events"
    register_module(name, lambda: {"name": name})

    bus = InMemoryEventBus()
    used_events = []
    bus.subscribe("module.used", lambda e: used_events.append(e))

    mgr = RuntimeModuleManager(bus)
    mgr.load(name)
    mgr.load(name)  # cached
    bus.join()

    assert len(used_events) >= 2
    assert used_events[0]["module"] == name
    assert used_events[0].get("cached") is False
    assert used_events[1]["module"] == name
    assert used_events[1].get("cached") is True


def test_runtime_module_manager_update_does_not_prune_on_unknown_required() -> None:
    name = "dummy_runtime_keep_loaded"
    register_module(name, lambda: {"name": name})

    mgr = RuntimeModuleManager()
    mgr.load(name)

    # When the requested names are unknown/disabled, update() should avoid
    # unloading everything as a side-effect.
    mgr.update(["this_is_not_a_registered_module_name"], prune=True)
    assert name in mgr.loaded_modules()
