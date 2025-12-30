"""AutoGPT 文件存储基础模块。

本模块定义了文件存储系统的抽象接口，提供了统一的文件操作API。
支持本地文件系统、云存储等多种存储后端的抽象化访问。

核心功能:
    - 统一的文件操作接口
    - 路径安全性验证
    - 存储根目录限制
    - 事件钩子支持

设计模式:
    - 抽象基类模式
    - 模板方法模式
    - 策略模式（不同存储后端）
    - 观察者模式（事件钩子）

安全特性:
    - 路径遍历攻击防护
    - 根目录访问限制
    - 空字节注入防护
    - 绝对路径验证
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from io import IOBase, TextIOBase
from pathlib import Path
from typing import IO, Any, BinaryIO, Callable, Literal, TextIO, overload

from autogpt.core.configuration.schema import SystemConfiguration

logger = logging.getLogger(__name__)


class FileStorageConfiguration(SystemConfiguration):
    """文件存储配置类。

    定义了文件存储系统的基本配置参数，包括根目录限制和根路径设置。

    属性:
        restrict_to_root: 是否限制文件访问在根目录内
        root: 存储系统的根目录路径
    """
    restrict_to_root: bool = True  # 是否限制访问根目录
    root: Path = Path("/")  # 存储根目录


class FileStorage(ABC):
    """文件存储抽象基类。

    定义了文件存储系统的统一接口，支持多种存储后端的实现。
    提供了完整的文件操作功能，包括读写、列表、删除等操作。

    核心特性:
        - 抽象化的存储接口
        - 路径安全性保障
        - 事件钩子机制
        - 类型安全的文件操作

    安全机制:
        - 路径遍历防护
        - 根目录限制
        - 输入验证
        - 权限控制

    扩展性:
        - 支持多种存储后端
        - 可插拔的实现
        - 灵活的配置选项
    """

    # 文件写入事件钩子，在文件写入后执行
    on_write_file: Callable[[Path], Any] | None = None
    """文件写入事件钩子。

    在文件成功写入后执行的回调函数，可用于日志记录、
    缓存更新、通知发送等后续处理。

    参数:
        Path: 相对于存储根目录的文件路径
    """

    @property
    @abstractmethod
    def root(self) -> Path:
        """The root path of the file storage."""

    @property
    @abstractmethod
    def restrict_to_root(self) -> bool:
        """Whether to restrict file access to within the storage's root path."""

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Whether the storage is local (i.e. on the same machine, not cloud-based)."""

    @abstractmethod
    def initialize(self) -> None:
        """
        Calling `initialize()` should bring the storage to a ready-to-use state.
        For example, it can create the resource in which files will be stored, if it
        doesn't exist yet. E.g. a folder on disk, or an S3 Bucket.
        """

    @overload
    @abstractmethod
    def open_file(
        self,
        path: str | Path,
        mode: Literal["w", "r"] = "r",
        binary: Literal[False] = False,
    ) -> TextIO | TextIOBase:
        """Returns a readable text file-like object representing the file."""

    @overload
    @abstractmethod
    def open_file(
        self,
        path: str | Path,
        mode: Literal["w", "r"] = "r",
        binary: Literal[True] = True,
    ) -> BinaryIO | IOBase:
        """Returns a readable binary file-like object representing the file."""

    @abstractmethod
    def open_file(
        self, path: str | Path, mode: Literal["w", "r"] = "r", binary: bool = False
    ) -> IO | IOBase:
        """Returns a readable file-like object representing the file."""

    @overload
    @abstractmethod
    def read_file(self, path: str | Path, binary: Literal[False] = False) -> str:
        """Read a file in the storage as text."""
        ...

    @overload
    @abstractmethod
    def read_file(self, path: str | Path, binary: Literal[True] = True) -> bytes:
        """Read a file in the storage as binary."""
        ...

    @abstractmethod
    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the storage."""

    @abstractmethod
    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the storage."""

    @abstractmethod
    def list_files(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the storage."""

    @abstractmethod
    def list_folders(
        self, path: str | Path = ".", recursive: bool = False
    ) -> list[Path]:
        """List all folders in a directory in the storage."""

    @abstractmethod
    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""

    @abstractmethod
    def delete_dir(self, path: str | Path) -> None:
        """Delete an empty folder in the storage."""

    @abstractmethod
    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in the storage."""

    @abstractmethod
    def rename(self, old_path: str | Path, new_path: str | Path) -> None:
        """Rename a file or folder in the storage."""

    @abstractmethod
    def copy(self, source: str | Path, destination: str | Path) -> None:
        """Copy a file or folder with all contents in the storage."""

    @abstractmethod
    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""

    @abstractmethod
    def clone_with_subroot(self, subroot: str | Path) -> FileStorage:
        """Create a new FileStorage with a subroot of the current storage."""

    def get_path(self, relative_path: str | Path) -> Path:
        """Get the full path for an item in the storage.

        Parameters:
            relative_path: The relative path to resolve in the storage.

        Returns:
            Path: The resolved path relative to the storage.
        """
        return self._sanitize_path(relative_path)

    def _sanitize_path(
        self,
        path: str | Path,
    ) -> Path:
        """Resolve the relative path within the given root if possible.

        Parameters:
            relative_path: The relative path to resolve.

        Returns:
            Path: The resolved path.

        Raises:
            ValueError: If the path is absolute and a root is provided.
            ValueError: If the path is outside the root and the root is restricted.
        """

        # Posix systems disallow null bytes in paths. Windows is agnostic about it.
        # Do an explicit check here for all sorts of null byte representations.
        if "\0" in str(path):
            raise ValueError("Embedded null byte")

        logger.debug(f"Resolving path '{path}' in storage '{self.root}'")

        relative_path = Path(path)

        # Allow absolute paths if they are contained in the storage.
        if (
            relative_path.is_absolute()
            and self.restrict_to_root
            and not relative_path.is_relative_to(self.root)
        ):
            raise ValueError(
                f"Attempted to access absolute path '{relative_path}' "
                f"in storage '{self.root}'"
            )

        full_path = self.root / relative_path
        if self.is_local:
            full_path = full_path.resolve()
        else:
            full_path = Path(os.path.normpath(full_path))

        logger.debug(f"Joined paths as '{full_path}'")

        if self.restrict_to_root and not full_path.is_relative_to(self.root):
            raise ValueError(
                f"Attempted to access path '{full_path}' "
                f"outside of storage '{self.root}'."
            )

        return full_path
