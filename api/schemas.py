"""Pydantic request/response schemas for the Ved HTTP API."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ---- Threads ----

class CreateThreadReq(BaseModel):
    title: Optional[str] = None


class RenameThreadReq(BaseModel):
    title: str = Field(..., min_length=1)


class ThreadOut(BaseModel):
    id: str
    title: str
    created_at: float
    message_count: int


class MessageOut(BaseModel):
    role: str
    content: str


# ---- Mode ----

class SetModeReq(BaseModel):
    mode: str


class ModeOut(BaseModel):
    mode: str


# ---- Chat ----

class ChatReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    attachments: Optional[List[str]] = None  # server-side file paths (advanced)


class ApprovalReq(BaseModel):
    approved: bool
    session_id: str


class ToolCreationApprovalReq(BaseModel):
    approved: bool
    session_id: str


# ---- Memories (long-term pinned context) ----

class MemoryPinItem(BaseModel):
    user: str
    assistant: str


class MemoriesOut(BaseModel):
    items: List[MemoryPinItem]


# ---- Global files ----

class GlobalFileOut(BaseModel):
    filename: str
    chunk_count: int
    evicted: List[str] = Field(default_factory=list)


# ---- Health ----

class HealthOut(BaseModel):
    status: str


# ---- Thread-scoped files ----

class ThreadFileOut(BaseModel):
    filename: str
    chunk_count: int
    evicted: List[str] = Field(default_factory=list)
    uploaded_at: float


# ---- Script execution ----

class RunOut(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    truncated_stdout: bool
    truncated_stderr: bool


# ---- Telemetry ----

class ActiveUserOut(BaseModel):
    username: str
    session_id: str
    source: str
    mode: str
    started_at: float
    last_heartbeat: float
    meta: dict = Field(default_factory=dict)


class TelemetryOut(BaseModel):
    active_count: int
    active_users: List[ActiveUserOut]
    timeout_seconds: float
