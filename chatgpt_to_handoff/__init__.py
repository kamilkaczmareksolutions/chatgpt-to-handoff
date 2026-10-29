"""Turn a ChatGPT export or share link into HANDOFF.md or AGENTS.md."""

from __future__ import annotations

import argparse
import enum
import io
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

__all__ = [
    "ConversationSummary",
    "HandoffError",
    "list_conversations",
    "main",
    "write_handoff",
]

_NONE_EXTRACTED = "_None extracted._"
_IMAGE_PLACEHOLDER = "[image]"
_REQUIREMENT_CAP = 40
_MAX_ZIP_DEPTH = 8
_SHARE_TIMEOUT_SECONDS = 20
_CANONICAL_SHARE_PREFIX = "https://chatgpt.com/share/"
_SHARE_API_PREFIX = "https://chatgpt.com/backend-api/share/"
_SHARE_HOSTS = frozenset({"chatgpt.com", "chat.openai.com"})
_SHARE_USER_AGENT = "Mozilla/5.0 (compatible; chatgpt-to-handoff/1.0)"
_AGENTS_INSTRUCTION = (
    "This file is a handoff from a ChatGPT conversation. Follow the "
    "requirements and decisions below. Treat open questions as unresolved."
)
_GOAL_LINE = re.compile(r"(?i)^(goal|context|objective)\b")
_DECISION_LINE = re.compile(
    r"(?i)\b(decided|decision|we will|we'll|let's go with|chosen approach)\b"
)
_REQUIREMENT_LINE = re.compile(r"(?i)\b(must|should|requirement)\b")
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.+)$")
_LANGUAGE_TOKEN = re.compile(r"[A-Za-z0-9_+#.-]+")
_BARE_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """One list row. ``update_time`` is UTC ``YYYY-MM-DDTHH:MM:SSZ`` or None."""

    title: str
    id: str
    update_time: str | None


class HandoffError(Exception):
    """Input, selection, network, or output failure. The destination is unchanged.

    ``summaries`` is populated for an ambiguous id, an ambiguous title, or a
    multi-conversation source with no selector. Other failures leave it empty.
    The implementation calls ``super().__init__(message)`` and stores
    ``summaries``.
    """

    summaries: tuple[ConversationSummary, ...]

    def __init__(
        self,
        message: str,
        summaries: tuple[ConversationSummary, ...] = (),
    ) -> None:
        super().__init__(message)
        self.summaries = summaries


def write_handoff(
    source: str | os.PathLike[str],
    *,
    conversation_id: str | None = None,
    title: str | None = None,
    output: str | os.PathLike[str] | None = None,
    agents_md: bool = False,
) -> Path:
    """Write one handoff for ``source`` and return its path."""
    conversations = _load(source)
    selected = _select(
        conversations,
        conversation_id=conversation_id,
        title=title,
    )
    label = _source_label(source)
    transcript = _to_transcript(selected, label)
    kind = DocumentKind.AGENTS if agents_md else DocumentKind.HANDOFF
    text = _render_document(transcript, kind)
    if output is None:
        dest = Path("AGENTS.md" if agents_md else "HANDOFF.md")
    else:
        dest = Path(output)
    _atomic_write(dest, text)
    return dest


def list_conversations(
    source: str | os.PathLike[str],
) -> tuple[ConversationSummary, ...]:
    """Return conversations sorted by title, then id. Print nothing."""
    conversations = _load(source)
    return tuple(_to_summary(c) for c in conversations)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI adapter. ``argv`` defaults to ``sys.argv[1:]``."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="chatgpt-to-handoff")
    parser.add_argument("source")
    parser.add_argument("--id", dest="conversation_id")
    parser.add_argument("--title")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("--agents-md", action="store_true")
    args = parser.parse_args(list(argv))

    if args.list:
        if args.conversation_id or args.title or args.output or args.agents_md:
            sys.stderr.write(
                "error: --list cannot be combined with --id, --title, -o, or --agents-md\n"
            )
            return 2
        try:
            for row in list_conversations(args.source):
                sys.stdout.write(_format_summary_line(row))
        except HandoffError as exc:
            sys.stderr.write(f"error: {exc}\n")
            for summary in exc.summaries:
                sys.stderr.write(_format_summary_line(summary))
            return 1
        return 0

    try:
        write_handoff(
            args.source,
            conversation_id=args.conversation_id,
            title=args.title,
            output=args.output,
            agents_md=args.agents_md,
        )
    except HandoffError as exc:
        sys.stderr.write(f"error: {exc}\n")
        for summary in exc.summaries:
            sys.stderr.write(_format_summary_line(summary))
        return 1
    return 0


# --- domain. Wire values do not cross _parse_conversation. ---


@dataclass(frozen=True, slots=True)
class _Seal:
    """Singleton token for the three retained-text constructors."""


_SEAL = _Seal()


@dataclass(frozen=True, slots=True)
class _Sealed:
    """Base for values that only the branch pipeline may construct."""

    _seal: _Seal = field(repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise HandoffError("invalid seal")


class VisibleRole(enum.Enum):
    """Roles that may appear in a handoff. Tool is not a member."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class _AuthorRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SkipKind(enum.Enum):
    NULL = "null"
    HIDDEN = "hidden"
    EMPTY = "empty"
    INTERMEDIATE = "intermediate"
    TOOL = "tool"
    UNKNOWN_ROLE = "unknown_role"


class _Channel(enum.Enum):
    """One field so a message cannot be both absent and non-final."""

    ABSENT = "absent"
    FINAL = "final"
    OTHER = "other"


class DocumentKind(enum.Enum):
    HANDOFF = "handoff"
    AGENTS = "agents"


@dataclass(frozen=True, slots=True)
class ProseLine:
    """One line outside a fence. Matching never sees code bodies."""

    text: str


@dataclass(frozen=True, slots=True)
class CodeSpan:
    """One fenced block or one content_type ``code`` text part."""

    language: str | None
    body: str


@dataclass(frozen=True, slots=True)
class VisibleMessage(_Sealed):
    """On-branch text the handoff may contain."""

    role: VisibleRole
    segments: tuple[ProseLine | CodeSpan, ...]


@dataclass(frozen=True, slots=True)
class SkippedNode:
    """On-branch node whose text was discarded. Not stored on ActiveBranch."""

    kind: SkipKind


@dataclass(frozen=True, slots=True)
class _RawMessage:
    """Message after the JSON boundary. Non-text parts are None."""

    role: _AuthorRole | None
    content_type: str | None
    parts: tuple[str | None, ...]
    visually_hidden: bool
    channel: _Channel


@dataclass(frozen=True, slots=True)
class _TreeNode:
    """Parent pointer plus raw message. The export ``children`` array is not a field."""

    id: str
    parent_id: str | None
    message: _RawMessage | None


class _MappingTree:
    """Parent-pointer index for one conversation object."""

    __slots__ = ("_current_id", "_by_id")

    def __init__(
        self,
        current_id: str,
        by_id: Callable[[str], _TreeNode | None],
    ) -> None:
        self._current_id = current_id
        self._by_id = by_id

    def active_branch(self) -> ActiveBranch:
        """Walk parents from ``current_node`` to the root, then reverse."""
        node_id: str | None = self._current_id
        chain: list[_TreeNode] = []
        seen: set[str] = set()
        while node_id is not None:
            if node_id in seen:
                raise HandoffError("parent chain cycles")
            seen.add(node_id)
            node = self._by_id(node_id)
            if node is None:
                raise HandoffError("active branch leaves the mapping")
            chain.append(node)
            node_id = node.parent_id
        chain.reverse()
        messages: list[VisibleMessage] = []
        for node in chain:
            body = _classify(node.message)
            if isinstance(body, VisibleMessage):
                messages.append(body)
            elif isinstance(body, SkippedNode):
                continue
            else:
                _assert_never(body)
        return ActiveBranch(tuple(messages), _seal=_SEAL)


@dataclass(frozen=True, slots=True)
class ActiveBranch(_Sealed):
    """Visible messages on the current_node chain, root first."""

    messages: tuple[VisibleMessage, ...]


@dataclass(frozen=True, slots=True)
class _Conversation:
    """One export conversation reduced to its visible active branch."""

    title: str
    id: str
    conversation_id: str | None
    update_time: float | None
    branch: ActiveBranch


@dataclass(frozen=True, slots=True)
class Transcript(_Sealed):
    """Document model. ``_render_document`` accepts only this type."""

    title: str
    conversation_id: str
    source_label: str
    turns: tuple[VisibleMessage, ...]


@dataclass(frozen=True, slots=True)
class _Sections:
    """Heuristic facts. Empty string or empty tuples mean no hits."""

    goal_and_context: str
    decisions: tuple[str, ...]
    requirements: tuple[str, ...]
    open_questions: tuple[str, ...]
    code: tuple[CodeSpan, ...]


def _assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(f"unexpected value: {value!r}")


def _visible_role(role: _AuthorRole) -> VisibleRole | None:
    if role is _AuthorRole.SYSTEM:
        return VisibleRole.SYSTEM
    if role is _AuthorRole.USER:
        return VisibleRole.USER
    if role is _AuthorRole.ASSISTANT:
        return VisibleRole.ASSISTANT
    if role is _AuthorRole.TOOL:
        return None
    _assert_never(role)


def _is_conversations_json(name: str) -> bool:
    base = Path(name).name
    if not base.endswith(".json"):
        return False
    prefix = "conversations"
    if not base.startswith(prefix):
        return False
    middle = base[len(prefix) : -len(".json")]
    return middle == "" or middle.startswith("-")


def _share_id(source: str) -> str | None:
    stripped = source.strip()
    if _BARE_UUID.match(stripped):
        return stripped.lower()
    parsed = urllib_parse.urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        return None
    host = parsed.hostname
    if host not in _SHARE_HOSTS:
        return None
    path = parsed.path.rstrip("/")
    if not path.startswith("/share/"):
        return None
    uuid_part = path[len("/share/") :]
    if _BARE_UUID.match(uuid_part):
        return uuid_part.lower()
    return None


def _source_label(source: str | os.PathLike[str]) -> str:
    path = Path(source)
    if path.exists():
        return path.name
    share = _share_id(str(source))
    if share is not None:
        return f"{_CANONICAL_SHARE_PREFIX}{share}"
    return path.name


def _load(source: str | os.PathLike[str]) -> tuple[_Conversation, ...]:
    path = Path(source)
    collected: list[_Conversation] = []

    if path.exists():
        if path.is_dir():
            json_paths: list[Path] = []
            for root, dirnames, filenames in os.walk(path, followlinks=False):
                dirnames.sort()
                for name in sorted(filenames):
                    if _is_conversations_json(name):
                        json_paths.append(Path(root) / name)
            json_paths.sort(key=lambda p: str(p))
            for json_path in json_paths:
                collected.extend(_load_json_file(json_path))
        elif path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
            collected.extend(_load_zip_path(path, depth=0))
        else:
            collected.extend(_load_json_file(path))
    else:
        share = _share_id(str(source))
        if share is None:
            raise HandoffError(f"source not found: {source}")
        collected.append(_fetch_share(share))

    if not collected:
        raise HandoffError("no conversations found")
    collected.sort(key=_conversation_sort_key)
    return tuple(collected)


def _load_json_file(path: Path) -> tuple[_Conversation, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HandoffError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HandoffError(f"invalid JSON in {path}: {exc}") from exc
    return _parse_document(data, str(path))


def _load_zip_path(path: Path, depth: int) -> tuple[_Conversation, ...]:
    if depth > _MAX_ZIP_DEPTH:
        raise HandoffError("zip nesting exceeds depth limit")
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return _load_zip_members(zf, depth)
    except zipfile.LargeZipFile as exc:
        raise HandoffError(f"zip too large: {path}") from exc
    except zipfile.BadZipFile as exc:
        raise HandoffError(f"invalid zip: {path}") from exc


def _load_zip_bytes(data: bytes, origin: str, depth: int) -> tuple[_Conversation, ...]:
    if depth > _MAX_ZIP_DEPTH:
        raise HandoffError("zip nesting exceeds depth limit")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            return _load_zip_members(zf, depth, origin_prefix=origin)
    except zipfile.LargeZipFile as exc:
        raise HandoffError(f"zip too large: {origin}") from exc
    except zipfile.BadZipFile as exc:
        raise HandoffError(f"invalid zip: {origin}") from exc


def _load_zip_members(
    zf: zipfile.ZipFile,
    depth: int,
    *,
    origin_prefix: str = "",
) -> tuple[_Conversation, ...]:
    names = sorted(zf.namelist(), key=lambda n: n)
    collected: list[_Conversation] = []
    for name in names:
        if name.endswith("/"):
            continue
        if name.lower().endswith(".zip"):
            nested = zf.read(name)
            nested_origin = f"{origin_prefix}!{name}" if origin_prefix else name
            collected.extend(_load_zip_bytes(nested, nested_origin, depth + 1))
        elif _is_conversations_json(name):
            try:
                raw = zf.read(name)
            except zipfile.LargeZipFile as exc:
                raise HandoffError(f"zip member too large: {name}") from exc
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                origin = f"{origin_prefix}!{name}" if origin_prefix else name
                raise HandoffError(f"invalid JSON in {origin}: {exc}") from exc
            origin = f"{origin_prefix}!{name}" if origin_prefix else name
            collected.extend(_parse_document(data, origin))
    return tuple(collected)


def _parse_document(data: object, origin: str) -> tuple[_Conversation, ...]:
    if isinstance(data, list):
        result: list[_Conversation] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise HandoffError(f"invalid conversation entry in {origin}")
            result.append(_parse_conversation(item))
        return tuple(result)
    if isinstance(data, Mapping) and "mapping" in data:
        return (_parse_conversation(data),)
    raise HandoffError(f"unrecognized JSON document: {origin}")


def _parse_update_time(payload: Mapping[str, Any]) -> float | None:
    if "update_time" not in payload:
        return None
    ut = payload["update_time"]
    if ut is None:
        return None
    if isinstance(ut, bool):
        raise HandoffError("invalid update_time")
    if isinstance(ut, str):
        raise HandoffError("invalid update_time")
    if not isinstance(ut, (int, float)):
        raise HandoffError("invalid update_time")
    value = float(ut)
    if not math.isfinite(value):
        raise HandoffError("invalid update_time")
    return value


def _parse_conversation(payload: Mapping[str, Any]) -> _Conversation:
    title = payload.get("title")
    if not isinstance(title, str):
        raise HandoffError("conversation missing title")
    mapping = payload.get("mapping")
    if not isinstance(mapping, Mapping):
        raise HandoffError("conversation missing mapping")
    current_node = payload.get("current_node")
    if not isinstance(current_node, str) or not current_node:
        raise HandoffError("conversation missing current_node")
    raw_id = payload.get("id")
    alt = payload.get("conversation_id")
    display_id: str | None = None
    alternate: str | None = None
    if isinstance(raw_id, str) and raw_id:
        display_id = raw_id
        if isinstance(alt, str) and alt and alt != display_id:
            alternate = alt
    elif isinstance(alt, str) and alt:
        display_id = alt
    else:
        raise HandoffError("conversation missing id")
    update_time = _parse_update_time(payload)
    tree = _index_tree(mapping, current_node)
    branch = tree.active_branch()
    return _Conversation(
        title=title,
        id=display_id,
        conversation_id=alternate,
        update_time=update_time,
        branch=branch,
    )


def _index_tree(mapping: Mapping[str, Any], current_id: str) -> _MappingTree:
    nodes: dict[str, _TreeNode] = {}

    for key, raw in mapping.items():
        if not isinstance(raw, Mapping):
            continue
        node_id = raw.get("id")
        if not isinstance(node_id, str):
            node_id = key if isinstance(key, str) else None
        if not node_id:
            continue
        parent_raw = raw.get("parent")
        parent_id: str | None
        if parent_raw is None:
            parent_id = None
        elif isinstance(parent_raw, str):
            parent_id = parent_raw
        else:
            parent_id = None
        message = _raw_message(
            raw.get("message") if isinstance(raw.get("message"), Mapping) else None
        )
        nodes[node_id] = _TreeNode(id=node_id, parent_id=parent_id, message=message)

    def by_id(node_id: str) -> _TreeNode | None:
        return nodes.get(node_id)

    if by_id(current_id) is None:
        raise HandoffError("current_node not in mapping")
    return _MappingTree(current_id, by_id)


def _raw_message(message: Mapping[str, Any] | None) -> _RawMessage | None:
    if message is None:
        return None
    author = message.get("author")
    role: _AuthorRole | None = None
    if isinstance(author, Mapping):
        role_raw = author.get("role")
        if isinstance(role_raw, str):
            try:
                role = _AuthorRole(role_raw)
            except ValueError:
                role = None
    content = message.get("content")
    content_type: str | None = None
    parts: tuple[str | None, ...] = ()
    if isinstance(content, Mapping):
        ct = content.get("content_type")
        if isinstance(ct, str):
            content_type = ct
        raw_parts = content.get("parts")
        if isinstance(raw_parts, list):
            converted: list[str | None] = []
            for part in raw_parts:
                if isinstance(part, str):
                    converted.append(part)
                else:
                    converted.append(None)
            parts = tuple(converted)
    metadata = message.get("metadata")
    hidden = False
    if isinstance(metadata, Mapping):
        hidden = metadata.get("is_visually_hidden_from_conversation") is True
    channel_raw = message.get("channel")
    if channel_raw is None:
        channel = _Channel.ABSENT
    elif channel_raw == "final":
        channel = _Channel.FINAL
    else:
        channel = _Channel.OTHER
    return _RawMessage(
        role=role,
        content_type=content_type,
        parts=parts,
        visually_hidden=hidden,
        channel=channel,
    )


def _classify(message: _RawMessage | None) -> VisibleMessage | SkippedNode:
    if message is None:
        return SkippedNode(SkipKind.NULL)
    if message.visually_hidden:
        return SkippedNode(SkipKind.HIDDEN)
    if message.role is None:
        return SkippedNode(SkipKind.UNKNOWN_ROLE)
    visible = _visible_role(message.role)
    if visible is None:
        return SkippedNode(SkipKind.TOOL)
    if message.channel is _Channel.OTHER:
        return SkippedNode(SkipKind.INTERMEDIATE)
    if message.channel is not _Channel.ABSENT and message.channel is not _Channel.FINAL:
        _assert_never(message.channel)
    if message.content_type == "model_editable_context":
        has_text = any(
            isinstance(p, str) and p.strip() for p in message.parts
        )
        if not has_text:
            return SkippedNode(SkipKind.EMPTY)
    segments = _segments(message)
    has_content = any(
        isinstance(s, CodeSpan)
        or (isinstance(s, ProseLine) and s.text.strip())
        for s in segments
    )
    if not has_content:
        return SkippedNode(SkipKind.EMPTY)
    return VisibleMessage(role=visible, segments=segments, _seal=_SEAL)


def _fence_open(line: str) -> tuple[str, str, str] | None:
    match = re.match(r"^(`{3,}|~{3,})(.*)$", line)
    if not match:
        return None
    mark = match.group(1)
    info = match.group(2).strip()
    language: str | None = None
    if info:
        token = info.split()[0]
        if _LANGUAGE_TOKEN.fullmatch(token):
            language = token
    return mark, mark[0], language


def _fence_close(line: str, mark_char: str, open_len: int) -> bool:
    pattern = rf"^{re.escape(mark_char)}{{{open_len},}}\s*$"
    return re.match(pattern, line) is not None


def _segments_from_text(text: str) -> list[ProseLine | CodeSpan]:
    out: list[ProseLine | CodeSpan] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opened = _fence_open(lines[i])
        if opened is None:
            out.append(ProseLine(lines[i]))
            i += 1
            continue
        mark, mark_char, language = opened
        open_line_index = i
        open_len = len(mark)
        body_lines: list[str] = []
        i += 1
        closed = False
        while i < len(lines):
            if _fence_close(lines[i], mark_char, open_len):
                closed = True
                i += 1
                break
            body_lines.append(lines[i])
            i += 1
        if closed:
            out.append(CodeSpan(language, "\n".join(body_lines)))
        else:
            out.append(ProseLine(lines[open_line_index]))
            for bl in body_lines:
                out.append(ProseLine(bl))
    return out


def _segments(message: _RawMessage) -> tuple[ProseLine | CodeSpan, ...]:
    if message.content_type == "code":
        spans: list[ProseLine | CodeSpan] = []
        for part in message.parts:
            if isinstance(part, str):
                spans.append(CodeSpan(None, part))
        return tuple(spans)

    combined: list[ProseLine | CodeSpan] = []
    first_part = True
    for part in message.parts:
        if not first_part:
            combined.append(ProseLine(""))
        first_part = False
        if part is None:
            combined.append(ProseLine(_IMAGE_PLACEHOLDER))
        else:
            combined.extend(_segments_from_text(part))
    return tuple(combined)


def _to_transcript(conversation: _Conversation, source_label: str) -> Transcript:
    return Transcript(
        title=conversation.title,
        conversation_id=conversation.id,
        source_label=source_label,
        turns=conversation.branch.messages,
        _seal=_SEAL,
    )


def _extract_sections(transcript: Transcript) -> _Sections:
    goal_lines: list[str] = []
    decision_lines: list[str] = []
    requirement_lines: list[str] = []
    question_lines: list[str] = []
    code: list[CodeSpan] = []
    first_user_body: str | None = None
    for turn in transcript.turns:
        later_user = False
        if turn.role is VisibleRole.USER:
            if first_user_body is None:
                first_user_body = _render_turn_body(turn)
            else:
                later_user = True
        for segment in turn.segments:
            if isinstance(segment, CodeSpan):
                code.append(segment)
                continue
            if not isinstance(segment, ProseLine):
                _assert_never(segment)
            line = segment.text.strip()
            if line == "":
                continue
            if later_user and _GOAL_LINE.match(line):
                goal_lines.append(line)
            if _DECISION_LINE.search(line):
                decision_lines.append(line)
            if turn.role is VisibleRole.USER and (
                _REQUIREMENT_LINE.search(line) or _LIST_ITEM.match(line)
            ):
                requirement_lines.append(_requirement_text(line))
            if turn.role in (VisibleRole.USER, VisibleRole.ASSISTANT) and line.endswith(
                "?"
            ):
                question_lines.append(line)
    tagged = _dedupe(goal_lines)
    if tagged:
        goal_text = "\n".join(tagged)
    else:
        goal_text = (first_user_body or "").strip()
    decisions = _dedupe(decision_lines)
    requirements = _dedupe(requirement_lines)[:_REQUIREMENT_CAP]
    questions = _dedupe(question_lines)
    return _Sections(goal_text, decisions, requirements, questions, tuple(code))


def _dedupe(lines: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return tuple(out)


def _requirement_text(line: str) -> str:
    match = _LIST_ITEM.match(line.strip())
    if match:
        return match.group(1).strip()
    return line.strip()


def _fence_length(body: str) -> int:
    longest = 0
    for match in re.finditer(r"`+", body):
        longest = max(longest, len(match.group(0)))
    return max(3, longest + 1)


def _render_turn_body(message: VisibleMessage) -> str:
    parts: list[str] = []
    for segment in message.segments:
        if isinstance(segment, ProseLine):
            parts.append(segment.text)
        elif isinstance(segment, CodeSpan):
            length = _fence_length(segment.body)
            fence = "`" * length
            lang = segment.language or ""
            parts.append(f"{fence}{lang}\n{segment.body}\n{fence}")
        else:
            _assert_never(segment)
    return "\n".join(parts)


def _role_heading(role: VisibleRole) -> str:
    if role is VisibleRole.USER:
        return "User"
    if role is VisibleRole.ASSISTANT:
        return "Assistant"
    if role is VisibleRole.SYSTEM:
        return "System"
    _assert_never(role)


def _section_body(text: str) -> str:
    if text.strip():
        return text
    return _NONE_EXTRACTED


def _bullet_section(items: tuple[str, ...]) -> str:
    if not items:
        return _NONE_EXTRACTED
    return "\n".join(f"- {item}" for item in items)


def _render_code_section(spans: tuple[CodeSpan, ...]) -> str:
    if not spans:
        return _NONE_EXTRACTED
    blocks: list[str] = []
    for span in spans:
        length = _fence_length(span.body)
        fence = "`" * length
        lang = span.language or ""
        blocks.append(f"{fence}{lang}\n{span.body}\n{fence}")
    return "\n\n".join(blocks)


def _render_document(transcript: Transcript, kind: DocumentKind) -> str:
    sections = _extract_sections(transcript)
    lines: list[str] = []
    if kind is DocumentKind.HANDOFF:
        lines.append(f"# {transcript.title}")
        lines.append("")
    elif kind is DocumentKind.AGENTS:
        lines.append("# AGENTS.md")
        lines.append("")
        lines.append(_AGENTS_INSTRUCTION)
        lines.append("")
    else:
        _assert_never(kind)

    lines.append(f"- Source: {transcript.source_label}")
    lines.append(f"- Conversation: {transcript.title}")
    lines.append(f"- Id: {transcript.conversation_id}")
    lines.append("")
    lines.append("## Goal and context")
    lines.append("")
    lines.append(_section_body(sections.goal_and_context))
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append(_bullet_section(sections.decisions))
    lines.append("")
    lines.append("## Requirements")
    lines.append("")
    lines.append(_bullet_section(sections.requirements))
    lines.append("")
    lines.append("## Open questions")
    lines.append("")
    lines.append(_bullet_section(sections.open_questions))
    lines.append("")
    lines.append("## Code")
    lines.append("")
    lines.append(_render_code_section(sections.code))
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Full visible transcript</summary>")
    lines.append("")
    for turn in transcript.turns:
        lines.append(f"### {_role_heading(turn.role)}")
        lines.append("")
        lines.append(_render_turn_body(turn))
        lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def _select(
    conversations: Sequence[_Conversation],
    *,
    conversation_id: str | None,
    title: str | None,
) -> _Conversation:
    if conversation_id and title:
        raise HandoffError("cannot use --id and --title together")
    if conversation_id:
        matches = [
            c
            for c in conversations
            if c.id == conversation_id
            or (c.conversation_id is not None and c.conversation_id == conversation_id)
        ]
        if len(matches) == 1:
            return matches[0]
        summaries = tuple(_to_summary(c) for c in conversations)
        if not matches:
            raise HandoffError("no conversation matches --id", summaries)
        raise HandoffError("multiple conversations match --id", summaries)
    if title:
        folded = title.casefold()
        matches = [c for c in conversations if c.title.casefold() == folded]
        if len(matches) == 1:
            return matches[0]
        summaries = tuple(_to_summary(c) for c in conversations)
        if not matches:
            raise HandoffError("no conversation matches --title", summaries)
        raise HandoffError("multiple conversations match --title", summaries)
    if len(conversations) == 1:
        return conversations[0]
    summaries = tuple(_to_summary(c) for c in conversations)
    raise HandoffError("multiple conversations; specify --id or --title", summaries)


def _conversation_sort_key(conversation: _Conversation) -> tuple[str, str]:
    return (conversation.title, conversation.id)


def _format_update_time(value: float | None) -> str | None:
    if value is None:
        return None
    dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_summary(conversation: _Conversation) -> ConversationSummary:
    return ConversationSummary(
        title=conversation.title,
        id=conversation.id,
        update_time=_format_update_time(conversation.update_time),
    )


def _format_summary_line(summary: ConversationSummary) -> str:
    safe_title = summary.title.replace("\t", " ").replace("\n", " ")
    update = summary.update_time or ""
    return f"{safe_title}\t{summary.id}\t{update}\n"


def _atomic_write(path: Path, text: str) -> None:
    if path.exists() and path.is_dir():
        raise HandoffError(f"output path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".handoff-", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise HandoffError(f"cannot write {path}: {exc}") from exc
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _fetch_share(share_id: str) -> _Conversation:
    url = f"{_SHARE_API_PREFIX}{share_id}"
    request = urllib_request.Request(
        url,
        headers={"User-Agent": _SHARE_USER_AGENT},
    )
    try:
        with urllib_request.urlopen(request, timeout=_SHARE_TIMEOUT_SECONDS) as resp:
            code = resp.getcode()
            if code != 200:
                raise HandoffError(f"share request failed: HTTP {code}")
            raw = resp.read()
    except urllib_error.HTTPError as exc:
        raise HandoffError(f"share request failed: HTTP {exc.code}") from exc
    except urllib_error.URLError as exc:
        raise HandoffError(f"share request failed: {exc.reason}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError("share response is not JSON") from exc
    if not isinstance(data, Mapping) or "mapping" not in data:
        raise HandoffError("share response missing conversation")
    return _parse_conversation(data)
