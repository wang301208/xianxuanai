from __future__ import annotations

"""Auto-generation utilities for AI-authored skills."""

import json
import logging
import inspect
import sys
import os
from dataclasses import replace
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Mapping, Optional, Sequence, TYPE_CHECKING

from .registry import SkillRegistry, SkillSpec
from .generator import SkillCodeGenerator, SkillGenerationConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .executor import SkillSandbox

logger = logging.getLogger(__name__)


DEFAULT_HANDLER = dedent(
    """
    \"\"\"Auto-generated skill handler.\"\"\"

    from __future__ import annotations
    from typing import Any, Dict


    def handle(payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        \"\"\"Basic placeholder implementation generated automatically.

        The handler normalizes payload/context, returning a structured response that
        callers can validate against the declared output schema. Replace the
        placeholder logic with real processing as needed.
        \"\"\"

        normalized_payload: Dict[str, Any] = dict(payload or {})
        normalized_context: Dict[str, Any] = dict(context or {})

        return {
            "status": "ok",
            "received": normalized_payload,
            "context": normalized_context,
            "summary": {
                "received_keys": sorted(normalized_payload.keys()),
                "context_keys": sorted(normalized_context.keys()),
            },
        }
    """
).strip()


def _slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "skill"


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class SkillAutoBuilder:
    """Create plugin scaffolding so the agent can propose new skills."""

    def __init__(
        self,
        *,
        plugin_root: Path | str = Path("skills") / "generated",
        review_root: Path | str | None = None,
        registry: SkillRegistry | None = None,
        default_enabled: bool = False,
        code_generator: SkillCodeGenerator | None = None,
        generator_model: str | None = None,
        generator_timeout: float | None = 30.0,
        generator_client: Any | None = None,
        require_review_approval: bool = False,
        sandbox: "SkillSandbox | None" = None,
        sandbox_payloads: Sequence[Dict[str, Any]] | None = None,
        sandbox_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.plugin_root = Path(plugin_root)
        self.plugin_root.mkdir(parents=True, exist_ok=True)
        self.review_root = Path(review_root) if review_root else None
        if self.review_root:
            self.review_root.mkdir(parents=True, exist_ok=True)
        if registry is not None:
            self.registry = registry
        else:
            try:  # optional global registry
                from backend.capability.skill_registry import get_skill_registry  # type: ignore

                self.registry = get_skill_registry()
            except Exception:  # pragma: no cover - optional dependency
                self.registry = None
        self.default_enabled = default_enabled
        self.require_review_approval = require_review_approval
        self.sandbox = sandbox
        self.sandbox_payloads = list(sandbox_payloads or [])
        self.sandbox_context = dict(sandbox_context or {})
        self.code_generator = code_generator or SkillCodeGenerator(
            llm_client=generator_client,
            config=SkillGenerationConfig(
                model=generator_model,
                request_timeout=generator_timeout,
            ),
        )

    # ------------------------------------------------------------------
    def create_skill(
        self,
        spec: SkillSpec,
        *,
        handler_source: str | None = None,
        tests_source: str | None = None,
        notes: Sequence[str] | None = None,
        auto_register: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        rpc_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist a new skill scaffold and optionally register it."""

        package_name = _slugify(spec.name)
        target_dir = self.plugin_root / package_name
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "__init__.py").write_text("", encoding="utf-8")

        module_name = ".".join(target_dir.relative_to(self.plugin_root).parts + ("skill",))
        reference_payload: Any | None = None
        if isinstance(metadata, dict):
            reference_payload = (
                metadata.get("retrieval_context")
                or metadata.get("references")
                or metadata.get("reference_material")
            )

        if handler_source is None or tests_source is None:
            try:
                generation_kwargs: Dict[str, Any] = {
                    "module_import": module_name,
                    "include_tests": tests_source is None,
                }
                if reference_payload is not None:
                    try:
                        if "references" in inspect.signature(self.code_generator.generate).parameters:
                            generation_kwargs["references"] = reference_payload
                    except Exception:
                        pass
                generation = self.code_generator.generate(spec, **generation_kwargs)
                handler_source = handler_source or generation.handler_source
                tests_source = tests_source or generation.tests_source
            except Exception as err:
                logger.error(
                    "Skill generation failed for %s: %s. Falling back to defaults.",
                    spec.name,
                    err,
                )

        if handler_source is None:
            handler_source = DEFAULT_HANDLER

        manifest_spec = replace(
            spec,
            enabled=self.default_enabled and spec.enabled,
            entrypoint=spec.entrypoint or f"{module_name}:handle",
            source=str(target_dir / f"{package_name}.skill.json"),
        )

        handler_module = target_dir / "skill.py"
        handler_module.write_text(
            handler_source.strip() if handler_source else DEFAULT_HANDLER,
            encoding="utf-8",
        )

        if tests_source:
            tests_dir = target_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / f"test_{package_name}.py").write_text(
                tests_source.strip(),
                encoding="utf-8",
            )

        sandbox_report: Optional[Dict[str, Any]] = None
        if self.sandbox is not None:
            sandbox_report = self._run_sandbox_checks(
                module_name=module_name,
                handler_path=handler_module,
                spec=manifest_spec,
            )

        incoming_metadata = dict(metadata or {})
        resolved_rpc_config: Optional[Dict[str, Any]] = None
        if isinstance(rpc_config, Mapping):
            resolved_rpc_config = dict(rpc_config)

        if resolved_rpc_config is None and manifest_spec.execution_mode == "rpc":
            meta_rpc = incoming_metadata.pop("rpc_config", None)
            if isinstance(meta_rpc, Mapping):
                resolved_rpc_config = dict(meta_rpc)
            else:
                docs_text = None
                for key in ("rpc_docs", "documentation", "retrieval_context", "reference_material"):
                    value = incoming_metadata.get(key)
                    if isinstance(value, str) and value.strip():
                        docs_text = value
                        break
                if docs_text:
                    try:
                        from .rpc_config_generator import RPCConfigGenerationConfig, SkillRPCConfigGenerator

                        gen_cfg = RPCConfigGenerationConfig(
                            model=getattr(getattr(self.code_generator, "config", None), "model", None),
                            request_timeout=float(
                                getattr(getattr(self.code_generator, "config", None), "request_timeout", 15.0) or 15.0
                            ),
                        )
                        generator = SkillRPCConfigGenerator(
                            llm_client=getattr(self.code_generator, "llm_client", None),
                            config=gen_cfg,
                        )
                        result = generator.generate(docs_text, hint=manifest_spec.name)
                        if isinstance(result.rpc_config, dict) and result.rpc_config:
                            resolved_rpc_config = dict(result.rpc_config)
                        incoming_metadata.setdefault(
                            "rpc_config_generation",
                            {**(result.diagnostics or {}), "used_llm": bool(result.used_llm)},
                        )
                    except Exception as err:
                        logger.debug("RPC config inference failed for %s: %s", manifest_spec.name, err)
        incoming_review = incoming_metadata.pop("review", None)
        review_status = (
            str(incoming_review.get("status"))
            if isinstance(incoming_review, Dict)
            else None
        ) or ("pending" if self.require_review_approval else "auto_approved")
        if sandbox_report and sandbox_report.get("status") == "failed":
            review_status = "failed"
        review_record: Dict[str, Any] = {
            "status": review_status,
            "requires_approval": self.require_review_approval,
        }
        if isinstance(incoming_review, Dict):
            review_record.update({k: v for k, v in incoming_review.items() if k != "status"})
        if sandbox_report:
            review_record["sandbox_status"] = sandbox_report.get("status")

        manifest_path = target_dir / f"{package_name}.skill.json"

        proposal_path = self._persist_review_artifacts(
            target_dir=target_dir,
            package_name=package_name,
            manifest_path=manifest_path,
            review_status=review_status,
            sandbox_report=sandbox_report,
            notes=notes,
            metadata=incoming_metadata,
        )

        if proposal_path:
            review_record["proposal"] = str(proposal_path)

        manifest_metadata: Dict[str, Any] = {}
        if incoming_metadata:
            manifest_metadata.update(incoming_metadata)
        manifest_metadata["review"] = review_record
        if sandbox_report:
            manifest_metadata["sandbox"] = sandbox_report

        manifest_payload: Dict[str, Any] = {
            "name": manifest_spec.name,
            "description": manifest_spec.description,
            "execution_mode": manifest_spec.execution_mode,
            "input_schema": manifest_spec.input_schema,
            "output_schema": manifest_spec.output_schema,
            "cost": manifest_spec.cost,
            "tags": manifest_spec.tags,
            "provider": manifest_spec.provider,
            "version": manifest_spec.version,
            "enabled": manifest_spec.enabled,
            "entrypoint": manifest_spec.entrypoint,
        }
        if resolved_rpc_config:
            manifest_payload["rpc_config"] = resolved_rpc_config
        if manifest_metadata:
            manifest_payload["metadata"] = manifest_metadata

        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        should_register = auto_register and self.registry is not None
        if sandbox_report and sandbox_report.get("status") == "failed":
            should_register = False
        if self.require_review_approval and review_status not in {"approved", "auto_approved"}:
            should_register = False
            logger.info(
                "Skill '%s' awaiting approval before registration (proposal: %s)",
                spec.name,
                proposal_path or manifest_path,
            )

        if should_register:
            try:
                register_metadata: Dict[str, Any] = {"manifest": str(manifest_path)}
                register_metadata.update(manifest_metadata)
                if resolved_rpc_config:
                    register_metadata.setdefault("rpc_config", dict(resolved_rpc_config))
                self.registry.register(
                    manifest_spec,
                    handler=None,
                    replace=True,
                    metadata=register_metadata,
                )
                logger.info(
                    "Auto-registered generated skill '%s' with review status '%s'",
                    spec.name,
                    review_status,
                )
            except Exception as err:
                logger.warning("Failed to auto-register generated skill %s: %s", spec.name, err)

        logger.info("Generated skill scaffold for '%s' at %s", spec.name, target_dir)
        return target_dir

    # ------------------------------------------------------------------
    def _run_sandbox_checks(
        self,
        *,
        module_name: str,
        handler_path: Path,
        spec: SkillSpec,
    ) -> Dict[str, Any]:
        if spec.execution_mode == "rpc" and not _parse_bool(
            os.getenv("SKILL_SANDBOX_VALIDATE_RPC") or os.getenv("SKILL_RPC_SANDBOX_VALIDATE"),
            default=False,
        ):
            return {
                "status": "skipped",
                "checked_at": datetime.utcnow().isoformat() + "Z",
                "samples": [],
                "reason": "rpc_validation_disabled",
            }
        if self.sandbox is None:
            return {}

        handler_fn = self._load_handler(module_name, handler_path)
        samples = self.sandbox_payloads or [{"sample": "value"}]
        report: Dict[str, Any] = {
            "status": "pending",
            "checked_at": datetime.utcnow().isoformat() + "Z",
            "samples": [],
        }

        for payload in samples:
            try:
                result = self.sandbox.run(
                    handler_fn,
                    payload,
                    context=self.sandbox_context,
                    metadata={"name": spec.name, "execution_mode": spec.execution_mode},
                )
                report["samples"].append(
                    {
                        "payload": payload,
                        "status": "passed",
                        "result": result,
                    }
                )
            except Exception as exc:
                report["samples"].append(
                    {
                        "payload": payload,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                report["status"] = "failed"
                logger.error(
                    "Sandbox execution failed for '%s' with payload %s: %s",
                    spec.name,
                    payload,
                    exc,
                )

        report["status"] = report["status"] if report["status"] != "pending" else "passed"
        logger.info(
            "Sandbox validation for '%s' completed with status=%s", spec.name, report["status"]
        )
        return report

    # ------------------------------------------------------------------
    def _persist_review_artifacts(
        self,
        *,
        target_dir: Path,
        package_name: str,
        manifest_path: Path,
        review_status: str,
        sandbox_report: Optional[Dict[str, Any]],
        notes: Sequence[str] | None,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Path]:
        review_dir = self.review_root or target_dir
        review_dir.mkdir(parents=True, exist_ok=True)

        proposal_payload: Dict[str, Any] = {
            "name": package_name,
            "manifest": str(manifest_path),
            "review_status": review_status,
            "requires_approval": self.require_review_approval,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata or {},
        }
        if sandbox_report:
            proposal_payload["sandbox"] = sandbox_report
        if notes:
            proposal_payload["notes"] = list(notes)

        proposal_path = review_dir / f"{package_name}.proposal.json"
        proposal_path.write_text(
            json.dumps(proposal_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        review_lines = [
            f"# Skill proposal: {package_name}",
            "",
            f"* Manifest: {manifest_path}",
            f"* Review status: {review_status}",
            f"* Requires approval: {self.require_review_approval}",
            f"* Proposal created: {proposal_payload['created_at']}",
        ]
        if sandbox_report:
            review_lines.append(f"* Sandbox status: {sandbox_report.get('status')}")
        review_lines.append("")
        review_lines.append("## Reviewer notes")
        if notes:
            review_lines.extend(f"- {line}" for line in notes)
        else:
            review_lines.append("- Pending reviewer feedback")

        review_file = review_dir / f"{package_name}_REVIEW.md"
        review_file.write_text("\n".join(review_lines) + "\n", encoding="utf-8")

        logger.info(
            "Queued skill proposal for '%s' at %s (review status=%s)",
            package_name,
            proposal_path,
            review_status,
        )
        return proposal_path

    # ------------------------------------------------------------------
    def _load_handler(self, module_name: str, handler_path: Path):
        spec = spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:  # pragma: no cover - loader missing
            raise ImportError(f"Unable to load handler module from {handler_path}")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        handler = getattr(module, "handle", None)
        if handler is None:
            raise ImportError(f"Skill handler 'handle' not found in {handler_path}")
        return handler
