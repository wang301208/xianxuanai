from __future__ import annotations

from modules.skills.rpc_config_generator import SkillRPCConfigGenerator


def test_rpc_config_generator_parses_http_invoke_url() -> None:
    docs = "POST http://localhost:8300/invoke\\nContent-Type: application/json"
    generator = SkillRPCConfigGenerator()
    result = generator.generate(docs)

    assert result.rpc_config is not None
    assert result.rpc_config["protocol"] == "http"
    assert result.rpc_config["endpoint"] == "http://localhost:8300"
    assert result.rpc_config["path"] == "/invoke"
    assert result.rpc_config["method"] == "POST"


def test_rpc_config_generator_parses_grpc_target_and_method() -> None:
    docs = "Use gRPC at my-svc.example:50051 method /SkillService/Invoke"
    generator = SkillRPCConfigGenerator()
    result = generator.generate(docs)

    assert result.rpc_config is not None
    assert result.rpc_config["protocol"] == "grpc"
    assert result.rpc_config["endpoint"] == "my-svc.example:50051"
    assert result.rpc_config["options"]["method"] == "/SkillService/Invoke"


def test_rpc_config_generator_uses_json_block_when_present() -> None:
    docs = """
    ```json
    {
      "rpc_config": {
        "protocol": "http",
        "endpoint": "https://api.example.com",
        "path": "/invoke",
        "method": "POST"
      }
    }
    ```
    """
    generator = SkillRPCConfigGenerator()
    result = generator.generate(docs)

    assert result.rpc_config is not None
    assert result.rpc_config["protocol"] == "http"
    assert result.rpc_config["endpoint"] == "https://api.example.com"
    assert result.rpc_config["path"] == "/invoke"

