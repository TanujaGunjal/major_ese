"""
Investigation workspace management module with SQLite persistence.
Tracks case review statuses, investigator notes, priority queues, and audit histories.
"""

import sqlite3
import json
from datetime import datetime
from uuid import uuid4
from typing import List, Dict, Any, Optional
from config.settings import SQLITE_DB_PATH, CASE_STATUS_OPTIONS, CASE_PRIORITY_OPTIONS


def get_sqlite_conn() -> sqlite3.Connection:
    """Returns a connection to the SQLite investigation database with row factory."""
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_investigation_db():
    """Initializes tables for cases, notes, and audit logs with transactional safety."""
    conn = get_sqlite_conn()
    try:
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'Medium',
                status TEXT NOT NULL DEFAULT 'New',
                assigned_investigator TEXT DEFAULT 'Unassigned',
                reason_for_review TEXT,
                summary_facts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS case_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                author TEXT NOT NULL,
                note_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases (case_id)
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                performed_by TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_unique_target
            ON cases (target_type, target_id)
            """)
    finally:
        conn.close()


# Ensure DB is created on import
init_investigation_db()


def create_or_get_case(
    target_type: str,
    target_id: str,
    priority: str = "Medium",
    reason: str = "Automated high-risk trigger",
    summary: str = "",
    investigator: str = "Unassigned"
) -> Dict[str, Any]:
    """Creates one case per target or returns its existing case without overwriting it."""
    target_type = target_type.strip().upper()
    target_id = target_id.strip().upper()
    if target_type not in {"CLAIM", "PROVIDER", "BENEFICIARY"}:
        raise ValueError("Unsupported investigation target type.")
    if not target_id:
        raise ValueError("An investigation target ID is required.")
    if priority not in CASE_PRIORITY_OPTIONS:
        raise ValueError(f"Priority must be one of: {', '.join(CASE_PRIORITY_OPTIONS)}.")
    if not reason.strip():
        raise ValueError("A reason for investigation is required.")

    conn = get_sqlite_conn()
    try:
        case_id = f"CASE-{datetime.now().strftime('%Y%m%d')}-{target_type[:3]}-{uuid4().hex[:8].upper()}"
        with conn:
            cursor = conn.execute("""
            INSERT INTO cases (
                case_id, target_type, target_id, priority, status,
                assigned_investigator, reason_for_review, summary_facts
            )
            VALUES (?, ?, ?, ?, 'New', ?, ?, ?)
            ON CONFLICT (target_type, target_id) DO NOTHING
            """, [
                case_id, target_type, target_id, priority,
                investigator.strip() or "Unassigned", reason.strip(), summary.strip(),
            ])
            created = cursor.rowcount == 1
            if created:
                conn.execute("""
                INSERT INTO audit_log (case_id, action, new_value, performed_by)
                VALUES (?, 'CASE_CREATED', 'Status: New', ?)
                """, [case_id, investigator.strip() or "Unassigned"])

            row = conn.execute(
                "SELECT * FROM cases WHERE target_type = ? AND target_id = ?",
                [target_type, target_id],
            ).fetchone()
            if row is None:
                raise sqlite3.IntegrityError("The case could not be created or reopened.")
            case = dict(row)
            case["was_created"] = created
            return case
    finally:
        conn.close()


def update_case_status(
    case_id: str,
    new_status: str,
    investigator: str = "Lead Investigator",
    note: Optional[str] = None,
    priority: Optional[str] = None,
    reason: Optional[str] = None,
    summary: Optional[str] = None,
) -> bool:
    """Updates case review fields and records changes and notes transactionally."""
    if new_status not in CASE_STATUS_OPTIONS:
        return False
    if priority is not None and priority not in CASE_PRIORITY_OPTIONS:
        return False

    conn = get_sqlite_conn()
    try:
        with conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", [case_id]
            ).fetchone()
            if row is None:
                return False

            old_case = dict(row)
            investigator = investigator.strip() or "Unassigned"
            changes = {
                "status": new_status,
                "assigned_investigator": investigator,
            }
            if priority is not None:
                changes["priority"] = priority
            if reason is not None:
                changes["reason_for_review"] = reason.strip()
            if summary is not None:
                changes["summary_facts"] = summary.strip()

            for field, new_value in changes.items():
                old_value = old_case.get(field)
                if old_value == new_value:
                    continue
                conn.execute(
                    f"UPDATE cases SET {field} = ?, updated_at = CURRENT_TIMESTAMP WHERE case_id = ?",
                    [new_value, case_id],
                )
                if field == "assigned_investigator":
                    continue
                conn.execute("""
                INSERT INTO audit_log (case_id, action, old_value, new_value, performed_by)
                VALUES (?, ?, ?, ?, ?)
                """, [
                    case_id,
                    f"{field.upper()}_CHANGE",
                    str(old_value) if old_value is not None else "",
                    str(new_value) if new_value is not None else "",
                    investigator,
                ])

            if note and note.strip():
                conn.execute("""
                INSERT INTO case_notes (case_id, author, note_text)
                VALUES (?, ?, ?)
                """, [case_id, investigator, note.strip()])
            return True
    finally:
        conn.close()


def add_case_note(case_id: str, author: str, note_text: str) -> bool:
    """Appends a new investigator note to an existing case."""
    if not note_text.strip():
        raise ValueError("An investigation note cannot be empty.")
    conn = get_sqlite_conn()
    try:
        with conn:
            updated = conn.execute(
                "UPDATE cases SET updated_at = CURRENT_TIMESTAMP WHERE case_id = ?",
                [case_id],
            )
            if updated.rowcount == 0:
                return False
            conn.execute("""
            INSERT INTO case_notes (case_id, author, note_text)
            VALUES (?, ?, ?)
            """, [case_id, author.strip() or "Unassigned", note_text.strip()])
        return True
    finally:
        conn.close()


def list_cases(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    target_type: Optional[str] = None,
    search_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves cases filtered by status, priority, or search term."""
    conn = get_sqlite_conn()
    conditions = []
    params = []

    if status and status != "All":
        conditions.append("status = ?")
        params.append(status)
    if priority and priority != "All":
        conditions.append("priority = ?")
        params.append(priority)
    if target_type and target_type != "All":
        conditions.append("target_type = ?")
        params.append(target_type)
    if search_query and search_query.strip():
        conditions.append("(case_id LIKE ? OR target_id LIKE ? OR reason_for_review LIKE ?)")
        term = f"%{search_query.strip()}%"
        params.extend([term, term, term])

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    sql = f"SELECT * FROM cases {where_clause} ORDER BY updated_at DESC"

    try:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        return rows
    finally:
        conn.close()


def get_case_full_history(case_id: str) -> Dict[str, Any]:
    """Returns case record, all attached notes, and full audit logs."""
    conn = get_sqlite_conn()
    try:
        case_row = conn.execute(
            "SELECT * FROM cases WHERE case_id = ?", [case_id]
        ).fetchone()
        if not case_row:
            return {}

        notes = [
            dict(r) for r in conn.execute(
                "SELECT * FROM case_notes WHERE case_id = ? ORDER BY created_at ASC, id ASC",
                [case_id],
            ).fetchall()
        ]
        audit = [
            dict(r) for r in conn.execute(
                "SELECT * FROM audit_log WHERE case_id = ? ORDER BY timestamp ASC, id ASC",
                [case_id],
            ).fetchall()
        ]
        return {"case": dict(case_row), "notes": notes, "audit": audit}
    finally:
        conn.close()
