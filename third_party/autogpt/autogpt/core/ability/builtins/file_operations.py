"""AutoGPT 文件操作能力实现。

本模块提供了 AutoGPT 系统的核心文件操作能力，包括文件读取和写入功能。
这些能力允许 AI 代理与文件系统进行交互，读取和创建文件内容。

主要能力:
    - ReadFile: 读取并解析文件内容
    - WriteFile: 写入文本内容到文件

技术特点:
    - 使用 unstructured 库进行智能文件解析
    - 支持多种文件格式的自动识别和处理
    - 集成工作空间安全机制
    - 提供详细的错误处理和状态反馈

安全考虑:
    - 所有文件操作都限制在指定的工作空间内
    - 包含完整的前置条件检查
    - 提供详细的操作日志和错误信息
"""

import base64
import logging
import mimetypes
import os
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace import Workspace


class ReadFile(Ability):
    """文件读取能力。
    
    提供智能文件读取功能，能够自动解析多种文件格式并提取文本内容。
    使用 unstructured 库进行文档解析，支持 PDF、Word、HTML 等多种格式。
    
    功能特性:
        - 自动文件格式识别
        - 智能文本提取
        - 结构化内容解析
        - 安全的工作空间限制
        
    使用场景:
        - 读取配置文件
        - 解析文档内容
        - 提取数据文件信息
        - 分析代码文件
    """
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.ReadFile",
        ),
        packages_required=["unstructured"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = "Read and parse all text from a file."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to read.",
        ),
    }

    def _check_preconditions(self, filename: str) -> AbilityResult | None:
        message = ""
        try:
            pass
        except ImportError:
            message = "Package charset_normalizer is not installed."

        try:
            file_path = self._workspace.get_path(filename)
            if not file_path.exists():
                message = f"File {filename} does not exist."
            if not file_path.is_file():
                message = f"{filename} is not a file."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"filename": filename},
                success=False,
                message=message,
                data=None,
            )

    async def __call__(self, filename: str) -> AbilityResult:
        if result := self._check_preconditions(filename):
            return result

        file_path = self._workspace.get_path(filename)
        mime_type, _ = mimetypes.guess_type(str(file_path))
        try:
            # Special handling for images and other binary files
            if mime_type and mime_type.startswith("image"):
                from PIL import Image

                with Image.open(file_path) as img:
                    width, height = img.size
                content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
                metadata = {
                    "filename": filename,
                    "mime_type": mime_type,
                    "width": width,
                    "height": height,
                    "encoding": "base64",
                }
            elif mime_type and not mime_type.startswith("text"):
                content = base64.b64encode(file_path.read_bytes()).decode("utf-8")
                metadata = {
                    "filename": filename,
                    "mime_type": mime_type,
                    "encoding": "base64",
                }
            else:
                from unstructured.partition.auto import partition

                elements = partition(str(file_path))
                content = "\n\n".join([element.text for element in elements])
                headings = [
                    el.text
                    for el in elements
                    if getattr(el, "category", "") == "Title"
                ]
                tables = []
                for el in elements:
                    if getattr(el, "category", "") == "Table":
                        meta_dict = (
                            el.metadata.to_dict() if hasattr(el, "metadata") else {}
                        )
                        tables.append(meta_dict.get("text_as_html", el.text))
                metadata = {"filename": filename}
                if headings:
                    metadata["headings"] = headings
                if tables:
                    metadata["tables"] = tables

            new_knowledge = Knowledge(
                content=content,
                content_type=ContentType.TEXT,
                content_metadata=metadata,
            )
            success = True
            message = f"File {file_path} read successfully."
        except Exception as e:
            new_knowledge = None
            success = False
            message = str(e)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename},
            success=success,
            message=message,
            new_knowledge=new_knowledge,
        )


class WriteFile(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.WriteFile",
        ),
        packages_required=["unstructured"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = "Write text to a file."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to write.",
        ),
        "contents": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The contents of the file to write.",
        ),
    }

    def _check_preconditions(
        self, filename: str, contents: str
    ) -> AbilityResult | None:
        message = ""
        try:
            self._workspace.get_path(filename)
            if not len(contents):
                message = f"File {filename} was not given any content."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"filename": filename, "contents": contents},
                success=False,
                message=message,
                data=None,
            )

    async def __call__(self, filename: str, contents: str) -> AbilityResult:
        if result := self._check_preconditions(filename, contents):
            return result

        file_path = self._workspace.get_path(filename)
        try:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(contents)
            success = True
            message = f"File {file_path} written successfully."
        except IOError as e:
            success = False
            message = str(e)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename, "contents": contents},
            success=success,
            message=message,
        )
