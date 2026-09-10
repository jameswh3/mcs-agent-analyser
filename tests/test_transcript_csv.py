import csv
import json

import pytest

from renderer.report import render_transcript_report
from renderer.tools import build_connected_agent_links
from timeline import build_timeline
from transcript import parse_transcript_csv


def test_parse_power_platform_transcript_collection(tmp_path):
    csv_path = tmp_path / "conversationtranscripts.csv"
    rows = [
        {
            "content": json.dumps(
                {
                    "activities": [
                        {"from": {"role": 1}, "timestamp": 1_700_000_000, "text": "Hello"},
                        {
                            "from": {"role": 0},
                            "timestamp": 1_700_000_001,
                            "name": "SessionInfo",
                            "value": {"outcome": "Resolved"},
                        },
                    ]
                }
            ),
            "conversationtranscriptid": "conversation-1",
            "conversationstarttime": "2024-01-01T00:00:00Z",
            "metadata": json.dumps({"BotId": "bot-1", "BotName": "Support Bot"}),
            "name": "First conversation",
        },
        {
            "content": json.dumps({"activities": [{"from": {"role": 1}, "text": "Second"}]}),
            "conversationtranscriptid": "conversation-2",
            "conversationstarttime": "2024-01-02T00:00:00Z",
            "metadata": "",
            "name": "",
        },
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    transcripts = parse_transcript_csv(csv_path)

    assert len(transcripts) == 2
    assert transcripts[0].title == "First conversation"
    assert transcripts[0].conversation_id == "conversation-1"
    assert transcripts[0].activities[0]["from"]["role"] == "user"
    assert transcripts[0].activities[1]["from"]["role"] == "bot"
    assert transcripts[0].activities[0]["timestamp"].startswith("2023-")
    assert transcripts[0].metadata["session_info"]["outcome"] == "Resolved"
    assert transcripts[0].metadata["export"]["BotName"] == "Support Bot"
    assert transcripts[0].metadata["conversation_start_time"] == "2024-01-01T00:00:00Z"
    assert transcripts[1].title == "conversation-2"


def test_parse_transcript_collection_requires_content_column(tmp_path):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("name\nConversation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content"):
        parse_transcript_csv(csv_path)


def test_agent_blocked_transcript_renders_interaction_evidence(tmp_path):
    csv_path = tmp_path / "conversationtranscripts.csv"
    rows = [
        {
            "content": json.dumps(
                {
                    "activities": [
                        {
                            "type": "trace",
                            "valueType": "ErrorTraceData",
                            "from": {"role": 0},
                            "value": {
                                "errorCode": "AgentBlocked",
                                "errorMessage": "Your admin has blocked this agent.",
                            },
                        },
                        {
                            "type": "trace",
                            "valueType": "SessionInfo",
                            "from": {"role": 0},
                            "value": {"outcome": "None", "turnCount": 1},
                        },
                    ]
                }
            ),
            "conversationtranscriptid": "blocked-conversation",
            "conversationstarttime": "2026-09-09T21:57:21Z",
            "metadata": json.dumps({"BotName": "Case Refinement Agent"}),
            "name": "blocked-agent-entry",
        }
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    transcript = parse_transcript_csv(csv_path)[0]
    timeline = build_timeline(transcript.activities, {})
    report = render_transcript_report(transcript.title, timeline, transcript.metadata)

    assert "| Agent | Case Refinement Agent |" in report
    assert "## Agent Interaction Evidence" in report
    assert "| Runtime result | `AgentBlocked` |" in report
    assert "No user message is recorded" in report
    assert "| Caller identity | Not present in this export |" in report


def test_large_connected_agent_trace_renders_observed_delegation(tmp_path):
    csv_path = tmp_path / "conversationtranscripts.csv"
    tool_call_id = "tool-call-1"
    padding = "x" * 140_000
    activities = [
        {
            "type": "event",
            "valueType": "ToolCallTrace:Started",
            "timestamp": "2026-09-09T22:23:10Z",
            "from": {"role": 0},
            "value": {
                "toolCallId": tool_call_id,
                "toolName": "TestyMcTesterton",
                "toolDisplayName": "Testy McTesterton",
                "toolKind": "agent",
                "filledParameters": {"task": "Run a diagnostic"},
            },
        },
        {
            "type": "event",
            "valueType": "ConnectedAgentInitializeTraceData",
            "timestamp": "2026-09-09T22:23:10Z",
            "from": {"role": 0},
            "value": {
                "sessionId": tool_call_id,
                "botSchemaName": "testy_schema",
                "parentBotSchemaName": "salesy_schema",
            },
        },
        {
            "type": "event",
            "valueType": "ToolCallTrace:Completed",
            "timestamp": "2026-09-09T22:23:50Z",
            "from": {"role": 0},
            "value": {
                "toolCallId": tool_call_id,
                "toolName": "TestyMcTesterton",
                "toolDisplayName": "Testy McTesterton",
                "toolKind": "agent",
                "toolCallStatus": "Completed",
                "durationMs": 39853,
                "result": json.dumps({"agent_response": padding, "is_error": False}),
            },
        },
    ]
    row = {
        "content": json.dumps({"activities": activities}),
        "conversationtranscriptid": "salesy-conversation",
        "conversationstarttime": "2026-09-09T22:23:10Z",
        "metadata": json.dumps({"BotName": "salesy_schema"}),
        "name": "salesy-agent-call",
    }
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)

    transcript = parse_transcript_csv(csv_path)[0]
    timeline = build_timeline(transcript.activities, {})
    report = render_transcript_report(transcript.title, timeline, transcript.metadata)

    assert len(timeline.tool_calls) == 1
    assert timeline.tool_calls[0].task_dialog_id == "testy_schema"
    assert timeline.tool_calls[0].step_type == "Agent"
    assert timeline.tool_calls[0].duration_ms == 39853
    assert "### Observed Delegations" in report
    assert "| salesy_schema | Testy McTesterton | ConnectedAgent | completed | 39853 ms |" in report

    collection = [
        {
            "title": "parent-conversation_parent-bot-id",
            "metadata": {"export": {"BotName": "salesy_schema"}},
        },
        {
            "title": "parent-conversation_child-conversation-id_child-bot-id",
            "metadata": {"export": {"BotName": "testy_schema"}},
        },
    ]
    timeline.tool_calls[0].observation.structured_content["conversation_id"] = "child-conversation-id"
    links = build_connected_agent_links(timeline, collection)

    assert links[0]["agent"] == "Testy McTesterton"
    assert links[0]["target_index"] == "1"
    assert links[0]["can_open"] is True