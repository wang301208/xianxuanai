"""AutoGPT 文件操作工具模块。

本模块提供了多种文件格式的解析和处理功能，支持文本、PDF、Word、JSON、XML、YAML、HTML 和 LaTeX 等格式。
使用策略模式实现不同文件类型的解析器，提供统一的文件内容提取接口。

支持的文件格式:
    - 文本文件: .txt, .md, .csv
    - PDF 文档: .pdf
    - Word 文档: .docx
    - 数据格式: .json, .xml, .yaml, .yml
    - 网页格式: .html, .htm, .xhtml
    - 学术格式: .tex (LaTeX)

设计模式:
    - 策略模式：不同文件格式的解析策略
    - 上下文模式：文件解析上下文管理
    - 工厂模式：解析器的创建和选择

核心功能:
    - 自动文件格式检测
    - 字符编码自动识别
    - 二进制文件检测
    - 统一的文本提取接口
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import BinaryIO

try:  # pragma: no cover - optional dependency
    import charset_normalizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent
    charset_normalizer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import docx  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent
    docx = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pypdf  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent
    pypdf = None  # type: ignore
import yaml
try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent
    BeautifulSoup = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pylatexenc.latex2text import LatexNodes2Text  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent
    LatexNodes2Text = None  # type: ignore

logger = logging.getLogger(__name__)


class ParserStrategy(ABC):
    """文件解析策略抽象基类。

    定义了所有文件解析器必须实现的接口，使用策略模式
    允许在运行时选择不同的解析算法。

    设计原则:
        - 单一职责：每个解析器只处理一种文件格式
        - 开闭原则：易于扩展新的文件格式支持
        - 接口隔离：提供简洁统一的解析接口
    """

    @abstractmethod
    def read(self, file: BinaryIO) -> str:
        """从文件中读取并解析内容为文本。

        Args:
            file: 二进制文件对象

        Returns:
            str: 解析后的文本内容

        Raises:
            具体的解析异常由子类实现定义
        """
        ...


# Basic text file reading
class TXTParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        payload = file.read()
        if charset_normalizer is None:  # pragma: no cover - optional dependency
            return payload.decode(errors="replace")

        charset_match = charset_normalizer.from_bytes(payload).best()
        encoding = getattr(charset_match, "encoding", None)
        if encoding:
            logger.debug(
                f"Reading {getattr(file, 'name', 'file')} with encoding '{encoding}'"
            )
        return str(charset_match)


# Reading text from binary file using pdf parser
class PDFParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        if pypdf is None:  # pragma: no cover - optional dependency
            raise RuntimeError("PDF parsing requires the 'pypdf' package.")
        parser = pypdf.PdfReader(file)  # type: ignore[union-attr]
        text = ""
        for page_idx in range(len(parser.pages)):
            text += parser.pages[page_idx].extract_text()
        return text


# Reading text from binary file using docs parser
class DOCXParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        if docx is None:  # pragma: no cover - optional dependency
            raise RuntimeError("DOCX parsing requires the 'python-docx' package.")
        doc_file = docx.Document(file)  # type: ignore[union-attr]
        text = ""
        for para in doc_file.paragraphs:
            text += para.text
        return text


# Reading as dictionary and returning string format
class JSONParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        data = json.load(file)
        text = str(data)
        return text


class XMLParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        if BeautifulSoup is None:  # pragma: no cover - optional dependency
            return file.read().decode(errors="replace")
        soup = BeautifulSoup(file, "xml")  # type: ignore[misc]
        text = soup.get_text()  # type: ignore[union-attr]
        return text


# Reading as dictionary and returning string format
class YAMLParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        data = yaml.load(file, Loader=yaml.SafeLoader)
        text = str(data)
        return text


class HTMLParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        if BeautifulSoup is None:  # pragma: no cover - optional dependency
            return file.read().decode(errors="replace")
        soup = BeautifulSoup(file, "html.parser")  # type: ignore[misc]
        text = soup.get_text()  # type: ignore[union-attr]
        return text


class LaTeXParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        latex = file.read().decode(errors="replace")
        if LatexNodes2Text is None:  # pragma: no cover - optional dependency
            return latex
        return LatexNodes2Text().latex_to_text(latex)  # type: ignore[operator]


class FileContext:
    def __init__(self, parser: ParserStrategy, logger: logging.Logger):
        self.parser = parser
        self.logger = logger

    def set_parser(self, parser: ParserStrategy) -> None:
        self.logger.debug(f"Setting Context Parser to {parser}")
        self.parser = parser

    def decode_file(self, file: BinaryIO) -> str:
        self.logger.debug(
            f"Reading {getattr(file, 'name', 'file')} with parser {self.parser}"
        )
        return self.parser.read(file)


extension_to_parser = {
    ".txt": TXTParser(),
    ".md": TXTParser(),
    ".markdown": TXTParser(),
    ".csv": TXTParser(),
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".json": JSONParser(),
    ".xml": XMLParser(),
    ".yaml": YAMLParser(),
    ".yml": YAMLParser(),
    ".html": HTMLParser(),
    ".htm": HTMLParser(),
    ".xhtml": HTMLParser(),
    ".tex": LaTeXParser(),
}


def is_file_binary_fn(file: BinaryIO):
    """Given a file path load all its content and checks if the null bytes is present

    Args:
        file (_type_): _description_

    Returns:
        bool: is_binary
    """
    file_data = file.read()
    file.seek(0)
    if b"\x00" in file_data:
        return True
    return False


def decode_textual_file(file: BinaryIO, ext: str, logger: logging.Logger) -> str:
    if not file.readable():
        raise ValueError(f"{repr(file)} is not readable")

    parser = extension_to_parser.get(ext.lower())
    if not parser:
        if is_file_binary_fn(file):
            raise ValueError(f"Unsupported binary file format: {ext}")
        # fallback to txt file parser (to support script and code files loading)
        parser = TXTParser()
    file_context = FileContext(parser, logger)
    return file_context.decode_file(file)
