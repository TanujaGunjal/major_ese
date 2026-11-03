"""
Feature engineering module for provider-level modeling and claim screening.
Implements the 10 provider aggregate features and transparent screening rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.preprocessing import format_money_trace


def compute_provider_features(claims_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the authoritative 10 provider features from raw claims:
    1. TotalClaims
    2. TotalReimbursement
    3. AverageReimbursement
    4. UniquePatients
    5. InpatientClaims
    6. InpatientReimbursement
    7. InpatientAvgReimbursement
    8. OutpatientClaims
    9. OutpatientReimbursement
    10. OutpatientAvgReimbursement
    """
    inpatient = claims_df[claims_df["ClaimType"] == "Inpatient"]
    outpatient = claims_df[claims_df["ClaimType"] == "Outpatient"]

    overall = claims_df.groupby("Provider").agg(
        TotalClaims=("ClaimID", "count"),
        TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
        AverageReimbursement=("InscClaimAmtReimbursed", "mean"),
        UniquePatients=("BeneID", "nunique")
    ).reset_index()

    inp_agg = inpatient.groupby("Provider").agg(
        InpatientClaims=("ClaimID", "count"),
        InpatientReimbursement=("InscClaimAmtReimbursed", "sum"),
        InpatientAvgReimbursement=("InscClaimAmtReimbursed", "mean")
    ).reset_index()

    out_agg = outpatient.groupby("Provider").agg(
        OutpatientClaims=("ClaimID", "count"),
        OutpatientReimbursement=("InscClaimAmtReimbursed", "sum"),
        OutpatientAvgReimbursement=("InscClaimAmtReimbursed", "mean")
    ).reset_index()

    res = overall.merge(inp_agg, on="Provider", how="left")
    res = res.merge(out_agg, on="Provider", how="left")
    res = res.fillna(0.0)

    return res


def screen_individual_claim(claim_dict: Dict[str, Any], historical_stats: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Evaluates transparent screening indicators for a single claim record:
    - Reimbursement outlier
    - Duration anomaly
    - Duplicate billing indicators
    - Procedure/Diagnosis complexity
    """
    reimbursement_value = claim_dict.get("InscClaimAmtReimbursed")
    reimb = (
        float(reimbursement_value)
        if reimbursement_value is not None and pd.notna(reimbursement_value)
        else None
    )
    claim_type = claim_dict.get("ClaimType")
    historical_stats = historical_stats or {}
    flags = []
    stored_rules = [
        ("Flag_ReimbursementOutlier", "OUTLIER_REIMBURSEMENT", "Reimbursement outlier", "Medium",
         "Reimbursement was marked by the source screening pipeline."),
        ("Flag_DuplicateBilling", "DUPLICATE_BILLING", "Repeated beneficiary/provider/date cluster", "Review",
         "The stored screening pipeline identified a duplicate-looking beneficiary/provider/claim pattern."),
        ("Flag_DurationAnomaly", "DURATION_INCONSISTENCY", "Date/duration anomaly", "High",
         "The stored screening pipeline identified a negative claim duration or length of stay."),
        ("Flag_ComplexProcedure", "COMPLEX_PROCEDURES", "Complex outpatient procedure", "Low",
         "The outpatient claim meets the recorded procedure-count rule."),
        ("Flag_HighFreqBeneficiary", "HIGH_FREQ_BENEFICIARY", "High-frequency beneficiary", "Medium",
         "The beneficiary was included in the high-frequency group defined by the data preparation pipeline.")
    ]
    for column, code, title, severity, description in stored_rules:
        observed = "Not available"
        baseline = "Stored source rule"
        observed_source_usd = None
        baseline_source_usd = None
        stored_trigger = (
            bool(claim_dict[column])
            if column in claim_dict and pd.notna(claim_dict[column])
            else None
        )
        if (
            code == "OUTLIER_REIMBURSEMENT"
            and historical_stats.get("reimbursement_p99") is not None
            and pd.notna(historical_stats.get("reimbursement_p99"))
        ):
            p99 = float(historical_stats["reimbursement_p99"])
            comparison_known = reimb is not None and claim_type is not None
            calculated_trigger = comparison_known and reimb > p99
            triggered = stored_trigger if stored_trigger is not None else calculated_trigger
            relationship = (
                "above" if reimb > p99
                else "equal to" if reimb == p99
                else "below"
            ) if comparison_known else "not comparable with the available values"
            baseline = (
                f"{format_money_trace(p99)} {_safe_text(claim_type)} reimbursement p99 "
                "(current dataset baseline; original stored threshold not available)"
            )
            description = f"Reimbursement is {relationship} the current dataset's {_safe_text(claim_type)} p99 baseline."
            if stored_trigger:
                review_context = (
                    "This is at the high end of the current claim-type distribution and may merit review."
                    if relationship in ("above", "equal to")
                    else "This differs from the current baseline and merits rule-version reconciliation."
                )
                description = (
                    "The stored source screening flag is set. "
                    f"Reimbursement is {relationship} the current dataset's {_safe_text(claim_type)} p99 baseline. "
                    f"{review_context} The original stored threshold is not available."
                )
            elif stored_trigger is None and not comparison_known:
                triggered = False
                baseline = "Not available: reimbursement or claim type is missing"
        elif code == "DUPLICATE_BILLING" and historical_stats.get("duplicate_count") is not None:
            duplicate_count = int(historical_stats["duplicate_count"])
            triggered = stored_trigger if stored_trigger is not None else duplicate_count > 1
            baseline = "At most one record for beneficiary/provider/start-date key"
            description = f"This matching key occurs on {duplicate_count} claim records; repeated keys may be legitimate and require comparison."
        elif code == "HIGH_FREQ_BENEFICIARY" and historical_stats.get("beneficiary_claim_count") is not None:
            beneficiary_count = int(historical_stats["beneficiary_claim_count"])
            triggered = stored_trigger if stored_trigger is not None else beneficiary_count >= 15
            baseline = "At least 15 claims in the available dataset window (2008-11-27 through 2009-12-31)"
            description = (
                f"The beneficiary has {beneficiary_count} claims in the available dataset window; "
                "compare the claim history because the count meets the high-frequency review rule."
            )
        elif code == "DURATION_INCONSISTENCY":
            start_date = claim_dict.get("ClaimStartDt")
            end_date = claim_dict.get("ClaimEndDt")
            admission = claim_dict.get("AdmissionDt")
            discharge = claim_dict.get("DischargeDt")
            duration = claim_dict.get("ClaimDurationDays")
            stay = claim_dict.get("LengthOfStayDays")
            calculated_trigger = (
                (duration is not None and pd.notna(duration) and float(duration) < 0)
                or (stay is not None and pd.notna(stay) and float(stay) < 0)
                or (pd.notna(start_date) and pd.notna(end_date) and start_date > end_date)
                or (pd.notna(admission) and pd.notna(discharge) and admission > discharge)
            )
            triggered = stored_trigger if stored_trigger is not None else calculated_trigger
            baseline = "End >= start; discharge >= admission; durations >= 0"
            if stored_trigger:
                description = "The stored source screening flag identifies a date or duration inconsistency."
        elif code == "COMPLEX_PROCEDURES":
            procedure_count = claim_dict.get("ProcedureCount")
            calculated_trigger = (
                claim_type == "Outpatient"
                and procedure_count is not None
                and pd.notna(procedure_count)
                and int(procedure_count) >= 3
            )
            triggered = stored_trigger if stored_trigger is not None else calculated_trigger
            baseline = "At least 3 outpatient procedure codes"
            if stored_trigger:
                description = (
                    "The stored source screening flag identifies a high outpatient procedure count. "
                    "Procedure count alone is descriptive, not evidence of wrongdoing."
                )
        elif code == "DUPLICATE_BILLING":
            triggered = bool(stored_trigger)
            baseline = "At most one record for beneficiary/provider/start-date key"
            if triggered:
                description = "The stored source flag is set; matching-key count is not available."
        elif code == "HIGH_FREQ_BENEFICIARY":
            triggered = bool(stored_trigger)
            baseline = "At least 15 claims in the available dataset window (2008-11-27 through 2009-12-31)"
            if triggered:
                description = "The stored source flag is set; beneficiary claim count is not available."
        elif code == "OUTLIER_REIMBURSEMENT":
            triggered = bool(stored_trigger)
            baseline = "Reimbursement baseline is not available"
            if triggered:
                description = "The stored source screening flag is set; its comparison baseline is not available."
        elif stored_trigger is not None:
            triggered = stored_trigger
        else:
            triggered = False
        if triggered:
            if code == "OUTLIER_REIMBURSEMENT":
                observed = format_money_trace(reimb) if reimb is not None else "Not available"
                p99 = historical_stats.get("reimbursement_p99")
                observed_source_usd = reimb
                baseline_source_usd = float(p99) if p99 is not None else None
            elif code == "DUPLICATE_BILLING":
                matches = historical_stats.get("duplicate_count")
                observed = f"{int(matches)} matching claim records" if matches is not None else "Not available"
            elif code == "DURATION_INCONSISTENCY":
                start_date = claim_dict.get("ClaimStartDt")
                end_date = claim_dict.get("ClaimEndDt")
                admission = claim_dict.get("AdmissionDt")
                discharge = claim_dict.get("DischargeDt")
                duration = claim_dict.get("ClaimDurationDays")
                stay = claim_dict.get("LengthOfStayDays")
                observed = (
                    f"Claim start {_safe_text(start_date)}; end {_safe_text(end_date)}; "
                    f"admission {_safe_text(admission)}; discharge {_safe_text(discharge)}; "
                    f"duration {duration if duration is not None and pd.notna(duration) else 'Not available'}; "
                    f"length of stay {stay if stay is not None and pd.notna(stay) else 'Not available'}"
                )
            elif code == "COMPLEX_PROCEDURES":
                count = claim_dict.get("ProcedureCount")
                observed = f"{int(count)} procedures" if count is not None and pd.notna(count) else "Not available"
            elif code == "HIGH_FREQ_BENEFICIARY":
                count = historical_stats.get("beneficiary_claim_count")
                observed = f"{int(count)} beneficiary claims" if count is not None else "Not available"
            flags.append({
                "code": code,
                "title": title,
                "severity": severity,
                "description": description,
                "observed": observed,
                "baseline": baseline,
                "observed_source_usd": observed_source_usd,
                "baseline_source_usd": baseline_source_usd,
            })

    # Provider context flag
    prov_risk = claim_dict.get("ProviderRiskLevel")
    provider_score = claim_dict.get("ProviderFraudProbability")
    if prov_risk == "High":
        score_text = (
            f"{float(provider_score):.1%} provider score"
            if provider_score is not None and pd.notna(provider_score)
            else "Not available"
        )
        flags.append({
            "code": "HIGH_RISK_PROVIDER",
            "title": "Elevated Provider Risk Context",
            "severity": "High",
            "description": "The provider-level model assigns a High risk tier (probability >= 0.70); this is context, not a claim-level screening result.",
            "observed": score_text,
            "baseline": "High provider risk tier >= 70% (provider-level)",
            "context_only": True
        })

    claim_flags = [flag for flag in flags if not flag.get("context_only", False)]
    return {
        "flags": flags,
        "flag_count": len(claim_flags),
        "is_flagged": len(claim_flags) > 0
    }


def _safe_text(value: Any) -> str:
    return str(value) if value is not None and pd.notna(value) else "Not available"
