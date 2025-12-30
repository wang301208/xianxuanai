import abc
import logging
import os
import shutil
import typing
from pathlib import Path

try:  # pragma: no cover - optional dependency for GCSWorkspace
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore


logger = logging.getLogger(__name__)


class Workspace(abc.ABC):
    @abc.abstractclassmethod
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    @abc.abstractclassmethod
    def read(self, task_id: str, path: str) -> bytes:
        pass

    @abc.abstractclassmethod
    def write(self, task_id: str, path: str, data: bytes) -> None:
        pass

    @abc.abstractclassmethod
    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        pass

    @abc.abstractclassmethod
    def exists(self, task_id: str, path: str) -> bool:
        pass

    @abc.abstractclassmethod
    def list(self, task_id: str, path: str) -> typing.List[str]:
        pass


class LocalWorkspace(Workspace):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()

    def _resolve_path(self, task_id: str, path: str) -> Path:
        path = Path(str(path).lstrip("/"))
        abs_path = (self.base_path / task_id / path).resolve()
        if not abs_path.is_relative_to(self.base_path):
            logger.error("Directory traversal is not allowed! - %s", abs_path)
            raise ValueError(f"Directory traversal is not allowed! - {abs_path}")
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        return abs_path

    def read(self, task_id: str, path: str) -> bytes:
        with open(self._resolve_path(task_id, path), "rb") as f:
            return f.read()

    def write(self, task_id: str, path: str, data: bytes) -> None:
        file_path = self._resolve_path(task_id, path)
        with open(file_path, "wb") as f:
            f.write(data)

    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        resolved_path = self._resolve_path(task_id, path)
        if directory:
            if recursive:
                shutil.rmtree(resolved_path)
            else:
                os.rmdir(resolved_path)
        else:
            os.remove(resolved_path)

    def exists(self, task_id: str, path: str) -> bool:
        return self._resolve_path(task_id, path).exists()

    def list(self, task_id: str, path: str) -> typing.List[str]:
        base = self._resolve_path(task_id, path)
        if not base.exists() or not base.is_dir():
            return []
        return [str(p.relative_to(self.base_path / task_id)) for p in base.iterdir()]


class GCSWorkspace(Workspace):
    def __init__(self, bucket_name: str, base_path: str = ""):
        if storage is None:
            raise ModuleNotFoundError(
                "google-cloud-storage is required to use GCSWorkspace"
            )
        self.bucket_name = bucket_name
        self.base_path = Path(base_path).resolve() if base_path else Path().resolve()
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.get_bucket(self.bucket_name)

    def _resolve_path(self, task_id: str, path: str) -> str:
        path_obj = Path(str(path).lstrip("/"))
        abs_path = (self.base_path / task_id / path_obj).resolve()
        if not abs_path.is_relative_to(self.base_path):
            logger.error("Directory traversal is not allowed! - %s", abs_path)
            raise ValueError(f"Directory traversal is not allowed! - {abs_path}")
        return abs_path.relative_to(self.base_path).as_posix()

    def read(self, task_id: str, path: str) -> bytes:
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        if not blob.exists():
            raise FileNotFoundError()
        return blob.download_as_bytes()

    def write(self, task_id: str, path: str, data: bytes) -> None:
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        blob.upload_from_string(data)

    def delete(self, task_id: str, path: str, directory=False, recursive=False):
        if directory and not recursive:
            raise ValueError("recursive must be True when deleting a directory")
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        if not blob.exists():
            return
        if directory:
            for b in list(self.bucket.list_blobs(prefix=blob.name)):
                b.delete()
        else:
            blob.delete()

    def exists(self, task_id: str, path: str) -> bool:
        blob = self.bucket.blob(self._resolve_path(task_id, path))
        return blob.exists()

    def list(self, task_id: str, path: str) -> typing.List[str]:
        prefix = os.path.join(task_id, self.base_path, path).replace("\\", "/") + "/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        return [str(Path(b.name).relative_to(prefix[:-1])) for b in blobs]
