import csv
from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger


@dataclass
class ParsedTranscript:
    title: str
    conversation_id: str
    activities: list[dict]
    metadata: dict


def _parse_transcript_data(raw: dict, source_name: str) -> tuple[list[dict], dict]:
    """Normalize a decoded transcript and return (activities, metadata).

    Normalization:
    1. Convert from.role from 0/1 to "bot"/"user"
    2. Convert timestamp from epoch seconds to ISO string
    3. Assign synthetic channelData position based on array index
    4. Set valueType from name field when valueType is missing
    5. Extract metadata from SessionInfo/ConversationInfo trace events
    """
    raw_activities: list[dict] = raw.get("activities", [])

    metadata: dict = {}
    normalized: list[dict] = []

    for idx, activity in enumerate(raw_activities):
        # 1. Normalize role
        from_info = activity.get("from", {}) or {}
        role_raw = from_info.get("role")
        if role_raw == 0:
            from_info["role"] = "bot"
        elif role_raw == 1:
            from_info["role"] = "user"
        activity["from"] = from_info

        # 2. Coerce numeric timestamps (seconds OR milliseconds) to ISO.
        # Reuses the canonical helper from `timeline` so seconds-vs-ms is
        # handled correctly; the previous inline implementation assumed
        # seconds and produced year-58000 dates for ms-encoded timestamps.
        from timeline import _coerce_timestamp

        ts_iso = _coerce_timestamp(activity.get("timestamp"))
        if ts_iso:
            activity["timestamp"] = ts_iso

        # 3. Synthetic position
        channel_data = activity.get("channelData") or {}
        if "webchat:internal:position" not in channel_data:
            channel_data["webchat:internal:position"] = idx * 1000
            activity["channelData"] = channel_data

        # 4. Set valueType from name when missing
        value_type = activity.get("valueType", "")
        name = activity.get("name", "")
        if not value_type and name:
            activity["valueType"] = name

        # 5. Extract metadata from SessionInfo / ConversationInfo
        value_type = activity.get("valueType", "")
        value = activity.get("value", {}) or {}

        if value_type == "SessionInfo":
            metadata["session_info"] = value
            logger.debug(f"SessionInfo: outcome={value.get('outcome')}, turns={value.get('turnCount')}")

        if value_type == "ConversationInfo":
            metadata["conversation_info"] = value

        normalized.append(activity)

    logger.info(f"Transcript: {len(normalized)} activities from {source_name}")
    return normalized, metadata


def parse_transcript_json(path: Path) -> tuple[list[dict], dict]:
    """Parse transcript JSON, normalize activities, return (activities, metadata)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Transcript JSON must contain an object: {path.name}")
    return _parse_transcript_data(raw, path.name)


def parse_transcript_csv(path: Path) -> list[ParsedTranscript]:
    """Parse a Power Platform conversationtranscripts CSV export."""
    transcripts: list[ParsedTranscript] = []

    csv.field_size_limit(10 * 1024 * 1024)
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "content" not in reader.fieldnames:
            raise ValueError("Transcript CSV must contain a 'content' column")

        for row_number, row in enumerate(reader, start=2):
            content = (row.get("content") or "").strip()
            if not content:
                raise ValueError(f"Transcript CSV row {row_number} has no content")

            try:
                raw = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Transcript CSV row {row_number} contains invalid content JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Transcript CSV row {row_number} content must contain a JSON object")

            conversation_id = (row.get("conversationtranscriptid") or "").strip()
            title = (row.get("name") or "").strip() or conversation_id or f"conversation-{row_number - 1}"
            activities, metadata = _parse_transcript_data(raw, f"{path.name} row {row_number}")

            export_metadata_text = (row.get("metadata") or "").strip()
            if export_metadata_text:
                try:
                    export_metadata = json.loads(export_metadata_text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Transcript CSV row {row_number} contains invalid metadata JSON: {exc}") from exc
                if isinstance(export_metadata, dict):
                    metadata["export"] = export_metadata

            conversation_start_time = (row.get("conversationstarttime") or "").strip()
            if conversation_start_time:
                metadata["conversation_start_time"] = conversation_start_time
            if conversation_id:
                metadata["conversation_transcript_id"] = conversation_id

            transcripts.append(
                ParsedTranscript(
                    title=title,
                    conversation_id=conversation_id,
                    activities=activities,
                    metadata=metadata,
                )
            )

    logger.info(f"Transcript collection: {len(transcripts)} conversations from {path.name}")
    return transcripts
