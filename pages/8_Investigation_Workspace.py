"""Persistent case management for claim and provider investigations."""

import json
import sqlite3

import pandas as pd
import streamlit as st

from config.settings import CASE_PRIORITY_OPTIONS, CASE_STATUS_OPTIONS
from src.claims_repository import get_claim_by_id, get_provider_profile
from src.investigation import (
    add_case_note,
    create_or_get_case,
    get_case_full_history,
    list_cases,
    update_case_status,
)
from src.preprocessing import (
    convert_currency,
    currency_configuration,
)
from src.styles import apply_custom_styles, render_header


def _available(value):
    return value is not None and not pd.isna(value)


def _json_value(value):
    if not _available(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _money_export(value):
    if not _available(value):
        return {"source_usd": None, "display_inr": None}
    amount = float(value)
    return {
        "source_usd": amount,
        "display_inr": convert_currency(amount, "USD", "INR"),
    }


def _get_target_record(case):
    if case["target_type"] == "CLAIM":
        record = get_claim_by_id(case["target_id"])
        if record is None:
            return None
        fields = (
            "ClaimID", "BeneID", "Provider", "ClaimType", "ClaimStartDt",
            "ClaimEndDt", "ClaimDurationDays",
        )
        details = {
            field: _json_value(record.get(field))
            for field in fields
        }
        details["Reimbursement"] = _money_export(
            record.get("InscClaimAmtReimbursed")
        )
        details["Deductible"] = _money_export(record.get("DeductibleAmtPaid"))
        return details

    if case["target_type"] == "PROVIDER":
        record = get_provider_profile(case["target_id"])
        if record is None:
            return None
        fields = (
            "Provider", "FraudProbability", "RiskLevel", "TotalClaims",
            "InpatientClaims", "OutpatientClaims", "UniquePatients",
        )
        details = {
            field: _json_value(record.get(field))
            for field in fields
        }
        for field in (
            "TotalReimbursement", "AverageReimbursement",
            "InpatientReimbursement", "OutpatientReimbursement",
        ):
            details[field] = _money_export(record.get(field))
        return details
    return None


apply_custom_styles()
render_header(
    "Investigation Workspace",
    "Create and manage persistent review cases for actual claims or providers.",
)

with st.expander("Create or open a case", expanded=True):
    with st.form("create_case_form"):
        target_type_default = st.session_state.get(
            "workspace_target_type",
            "CLAIM" if st.session_state.get("selected_claim_id") else "PROVIDER",
        )
        target_type_index = 1 if target_type_default == "PROVIDER" else 0
        kind = st.selectbox(
            "Investigation target",
            ["CLAIM", "PROVIDER"],
            index=target_type_index,
        )
        if kind == "CLAIM":
            default_target = (
                st.session_state.get("workspace_target_id")
                if target_type_default == "CLAIM"
                else ""
            ) or st.session_state.get("selected_claim_id", "")
        else:
            default_target = (
                st.session_state.get("workspace_target_id")
                if target_type_default == "PROVIDER"
                else ""
            ) or st.session_state.get("selected_provider_id", "")
        target_id = st.text_input(
            "Claim ID or Provider ID",
            value=default_target,
            placeholder="Enter the exact ID from the source records",
        )
        priority = st.selectbox(
            "Priority",
            CASE_PRIORITY_OPTIONS,
            index=CASE_PRIORITY_OPTIONS.index("Medium"),
        )
        assigned_investigator = st.text_input("Assigned investigator")
        reason = st.text_input("Reason for investigation", value="")
        summary = st.text_area("Supporting observations", value="")
        create_submitted = st.form_submit_button(
            "Create or open case", type="primary", use_container_width=True
        )

if create_submitted:
    target_id = target_id.strip().upper()
    if not target_id:
        st.error(f"Enter a {kind.lower()} ID.")
    elif not reason.strip():
        st.error("Enter a reason for investigation.")
    else:
        target_exists = (
            get_claim_by_id(target_id)
            if kind == "CLAIM"
            else get_provider_profile(target_id)
        )
        if target_exists is None:
            st.error(
                f"No matching {kind.lower()} record exists for {target_id}. "
                "Check the ID against the source data."
            )
        else:
            try:
                case = create_or_get_case(
                    kind,
                    target_id,
                    priority=priority,
                    reason=reason,
                    summary=summary,
                    investigator=assigned_investigator.strip() or "Unassigned",
                )
            except sqlite3.Error as exc:
                st.error(f"The case could not be saved to the investigation database: {exc}")
            else:
                st.session_state.active_case_id = case["case_id"]
                st.session_state.workspace_target_type = kind
                st.session_state.workspace_target_id = target_id
                action = "Created" if case["was_created"] else "Opened existing"
                st.success(f"{action} case {case['case_id']}.")

st.subheader("Find a saved case")
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    case_search = st.text_input("Search by case, target, or reason")
with filter_col2:
    case_status_filter = st.selectbox("Filter by status", ["All", *CASE_STATUS_OPTIONS])
with filter_col3:
    case_priority_filter = st.selectbox("Filter by priority", ["All", *CASE_PRIORITY_OPTIONS])
case_type_filter = st.selectbox("Filter by target type", ["All", "CLAIM", "PROVIDER"])
cases = list_cases(
    status=case_status_filter,
    priority=case_priority_filter,
    target_type=case_type_filter,
    search_query=case_search,
)

if not cases:
    st.info("No saved cases match these filters. Create a case above to begin an investigation.")
    st.stop()

case_table = pd.DataFrame(cases).rename(columns={
    "case_id": "Case ID",
    "target_type": "Target type",
    "target_id": "Target ID",
    "priority": "Priority",
    "status": "Review status",
    "assigned_investigator": "Investigator",
    "reason_for_review": "Reason for investigation",
    "updated_at": "Last updated",
})
st.dataframe(
    case_table[[
        "Case ID", "Target type", "Target ID", "Priority", "Review status",
        "Investigator", "Reason for investigation", "Last updated",
    ]],
    use_container_width=True,
    hide_index=True,
)

case_ids = [case["case_id"] for case in cases]
default_case_id = st.session_state.get("active_case_id")
selected_case_id = st.selectbox(
    "Open case history",
    case_ids,
    index=case_ids.index(default_case_id) if default_case_id in case_ids else 0,
)
st.session_state.active_case_id = selected_case_id
history = get_case_full_history(selected_case_id)
if not history:
    st.error("The selected case could not be loaded from the investigation database.")
    st.stop()

case = history["case"]
st.subheader(f"Case {case['case_id']}")
st.caption(
    f"{case['target_type']} {case['target_id']} · Created {case['created_at']} · "
    f"Last updated {case['updated_at']}"
)
st.markdown("**Reason for investigation**")
st.write(case.get("reason_for_review") or "Not available")
st.markdown("**Supporting observations**")
st.write(case.get("summary_facts") or "Not available")

with st.form("update_case_form"):
    update_col1, update_col2 = st.columns(2)
    status_index = (
        CASE_STATUS_OPTIONS.index(case["status"])
        if case["status"] in CASE_STATUS_OPTIONS
        else 0
    )
    priority_index = (
        CASE_PRIORITY_OPTIONS.index(case["priority"])
        if case["priority"] in CASE_PRIORITY_OPTIONS
        else 0
    )
    with update_col1:
        new_status = st.selectbox(
            "Review status",
            CASE_STATUS_OPTIONS,
            index=status_index,
        )
        new_priority = st.selectbox(
            "Priority",
            CASE_PRIORITY_OPTIONS,
            index=priority_index,
        )
        investigator = st.text_input(
            "Assigned investigator",
            value=case.get("assigned_investigator") or "",
        )
    with update_col2:
        updated_reason = st.text_input(
            "Reason for investigation",
            value=case.get("reason_for_review") or "",
        )
        updated_summary = st.text_area(
            "Supporting observations",
            value=case.get("summary_facts") or "",
        )
        update_note = st.text_area("Dated note for this update (optional)")
    update_submitted = st.form_submit_button("Save case updates")

if update_submitted:
    if not updated_reason.strip():
        st.error("A reason for investigation is required.")
    else:
        try:
            saved = update_case_status(
                selected_case_id,
                new_status,
                investigator=investigator.strip() or "Unassigned",
                note=update_note,
                priority=new_priority,
                reason=updated_reason,
                summary=updated_summary,
            )
        except sqlite3.Error as exc:
            st.error(f"Case updates could not be saved: {exc}")
        else:
            if saved:
                st.success("Case updates saved.")
                st.rerun()
            else:
                st.error("Case updates were not saved. Check the selected case, status, and priority.")

with st.form("add_case_note_form"):
    note_col1, note_col2 = st.columns([1, 3])
    with note_col1:
        note_author = st.text_input(
            "Note author",
            value=case.get("assigned_investigator") or "",
        )
    with note_col2:
        note_text = st.text_area("Investigator note")
    note_submitted = st.form_submit_button("Add dated note")

if note_submitted:
    if not note_text.strip():
        st.error("Enter a note before saving.")
    else:
        try:
            saved = add_case_note(
                selected_case_id,
                note_author.strip() or "Unassigned",
                note_text.strip(),
            )
        except (sqlite3.Error, ValueError) as exc:
            st.error(f"The note could not be saved: {exc}")
        else:
            if saved:
                st.success("Investigator note saved.")
                st.rerun()
            else:
                st.error("The note was not saved because this case no longer exists.")

st.subheader("Case history")
if history["notes"]:
    notes = pd.DataFrame(history["notes"]).rename(columns={
        "created_at": "Date and time",
        "author": "Investigator",
        "note_text": "Note",
    })
    st.dataframe(
        notes[["Date and time", "Investigator", "Note"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No investigator notes have been added to this case yet.")

if history["audit"]:
    audit = pd.DataFrame(history["audit"]).rename(columns={
        "timestamp": "Date and time",
        "action": "Change",
        "old_value": "Previous value",
        "new_value": "New value",
        "performed_by": "Investigator",
    })
    st.dataframe(
        audit[[
            "Date and time", "Change", "Previous value", "New value", "Investigator",
        ]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No case changes have been recorded.")

report = {
    "case": history["case"],
    "target_record": _get_target_record(case),
    "notes": history["notes"],
    "audit": history["audit"],
    "currency": currency_configuration(),
    "interpretation": (
        "A case status is a workflow label and is not proof of fraud. "
        "INR values are presentation-only conversions; source amounts remain USD."
    ),
}
st.download_button(
    "Export case summary (JSON)",
    json.dumps(report, indent=2, default=str).encode("utf-8"),
    file_name=f"{selected_case_id}_investigation.json",
    mime="application/json",
)
