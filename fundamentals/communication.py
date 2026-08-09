"""Messages, a visible wireless-link simulation, and authenticated encryption."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from cryptography.fernet import Fernet, InvalidToken


@dataclass(frozen=True)
class Message:
    """Application data before it becomes bytes on a link."""

    sequence: int
    sender: str
    kind: str
    value: str


@dataclass(frozen=True)
class LinkResult:
    """What happened while bytes crossed a simulated wireless link."""

    sequence: int
    delivered: bytes | None
    latency_ms: int
    dropped: bool


def encode_message(message: Message) -> bytes:
    """Encode structured data as UTF-8 JSON.  Encoding is not encryption."""

    if message.sequence < 0:
        raise ValueError("sequence must be zero or greater")
    if not message.sender or not message.kind:
        raise ValueError("sender and kind are required")
    return json.dumps(
        asdict(message),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def decode_message(payload: bytes) -> Message:
    """Decode JSON bytes and reject missing or extra fields."""

    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("payload is not valid UTF-8 JSON") from exc
    expected = {"sequence", "sender", "kind", "value"}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("message fields do not match the classroom protocol")
    if not isinstance(data["sequence"], int) or isinstance(data["sequence"], bool):
        raise ValueError("sequence must be an integer")
    if not all(isinstance(data[field], str) for field in ("sender", "kind", "value")):
        raise ValueError("sender, kind, and value must be strings")
    return Message(**data)


class WirelessLink:
    """A deterministic model of delay and packet loss, not a radio driver."""

    def __init__(
        self,
        latency_ms: int = 40,
        drop_sequences: frozenset[int] = frozenset(),
    ) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        self.latency_ms = latency_ms
        self.drop_sequences = drop_sequences

    def send(self, sequence: int, payload: bytes) -> LinkResult:
        dropped = sequence in self.drop_sequences
        return LinkResult(
            sequence=sequence,
            delivered=None if dropped else payload,
            latency_ms=self.latency_ms,
            dropped=dropped,
        )


class LockedMessageError(ValueError):
    """The key is wrong or encrypted bytes were changed."""


class SecretBox:
    """A small Fernet demo of confidentiality plus tamper detection.

    This is for messages created by this project.  It does not replace Wi-Fi
    WPA2/WPA3, HTTPS/TLS, access control, or the RoboMaster app's own protocol.
    """

    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)

    @classmethod
    def generate(cls) -> "SecretBox":
        return cls(Fernet.generate_key())

    def lock(self, message: Message) -> bytes:
        return self._fernet.encrypt(encode_message(message))

    def unlock(self, token: bytes) -> Message:
        try:
            return decode_message(self._fernet.decrypt(token))
        except InvalidToken as exc:
            raise LockedMessageError(
                "message was changed or this is not the matching secret key"
            ) from exc
