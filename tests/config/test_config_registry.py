import os
import sys

import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend", "autogpt"))
)

from third_party.autogpt.autogpt.core.configuration import ConfigRegistry, SystemConfiguration, SystemSettings, UserConfigurable


class ExampleConfiguration(SystemConfiguration):
    foo: int = UserConfigurable(1, from_env="EXAMPLE_FOO")
    bar: str = UserConfigurable("bar", from_env="EXAMPLE_BAR")


class ExampleSettings(SystemSettings):
    name: str = "example"
    description: str = "Example settings"
    mode: str = UserConfigurable("dev", from_env="EXAMPLE_MODE")


def test_registry_load_and_override(monkeypatch, tmp_path):
    monkeypatch.setenv("EXAMPLE_FOO", "2")
    monkeypatch.setenv("EXAMPLE_MODE", "prod")

    registry = ConfigRegistry()
    registry.collect()

    # Load environment variables
    registry.load_from_env()
    conf = registry.get("ExampleConfiguration")
    settings = registry.get("ExampleSettings")
    assert conf and conf.foo == 2
    assert settings and settings.mode == "prod"

    # YAML overrides
    config_yaml = tmp_path / "config.yaml"
    with config_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "ExampleConfiguration": {"foo": 5, "bar": "baz"},
                "ExampleSettings": {"mode": "test"},
            },
            f,
        )
    registry.load_from_yaml(str(config_yaml))
    conf = registry.get("ExampleConfiguration")
    settings = registry.get("ExampleSettings")
    assert conf.foo == 5 and conf.bar == "baz"
    assert settings.mode == "test"

    # CLI overrides
    registry.apply_overrides(
        {
            "ExampleConfiguration": {"bar": "cli"},
            "ExampleSettings": {"mode": "cli"},
        }
    )
    conf = registry.get("ExampleConfiguration")
    settings = registry.get("ExampleSettings")
    assert conf.bar == "cli"
    assert settings.mode == "cli"
