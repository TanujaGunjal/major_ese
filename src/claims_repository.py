"""
Claims repository utilizing DuckDB for sub-second queries across 558,000+ claim records.
Handles claim search, beneficiary timelines, provider lookups, and aggregate metrics.
"""

import duckdb
import pandas as pd
from typing import Dict, Any, List, Optional
from src.data_loader import get_duckdb_connection


def get_executive_metrics(
    claim_type: Optional[str] = None,
    risk_level: Optional[str] = None,
    min_reimb: Optional[float] = None,
    max_reimb: Optional[float] = None,
    provider_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Computes overall system metrics across all claims, providers, and beneficiaries,
    dynamically filtering when filters are applied.
    """
    con = get_duckdb_connection(read_only=True)
    conditions = []
    params = []

    if claim_type and claim_type != "All":
        conditions.append("ClaimType = ?")
        params.append(claim_type)
    if risk_level and risk_level != "All":
        conditions.append("ProviderRiskLevel = ?")
        params.append(risk_level)
    if provider_id and provider_id.strip():
        conditions.append("Provider ILIKE ?")
        params.append(f"%{provider_id.strip()}%")
    if min_reimb is not None and min_reimb > 0:
        conditions.append("InscClaimAmtReimbursed >= ?")
        params.append(min_reimb)
    if max_reimb is not None and max_reimb < 125000:
        conditions.append("InscClaimAmtReimbursed <= ?")
        params.append(max_reimb)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"""
    SELECT 
        CAST(COUNT(*) AS BIGINT) AS TotalClaims,
        CAST(COUNT(DISTINCT BeneID) AS BIGINT) AS UniqueBeneficiaries,
        CAST(COUNT(DISTINCT Provider) AS BIGINT) AS UniqueProviders,
        COALESCE(SUM(InscClaimAmtReimbursed), 0.0) AS TotalReimbursement,
        COALESCE(AVG(InscClaimAmtReimbursed), 0.0) AS AvgReimbursement,
        CAST(SUM(CASE WHEN ClaimType = 'Inpatient' THEN 1 ELSE 0 END) AS BIGINT) AS InpatientClaims,
        CAST(SUM(CASE WHEN ClaimType = 'Outpatient' THEN 1 ELSE 0 END) AS BIGINT) AS OutpatientClaims,
        CAST(SUM(CASE WHEN ScreeningFlagsCount > 0 THEN 1 ELSE 0 END) AS BIGINT) AS FlaggedClaims,
        CAST(SUM(CASE WHEN AIReviewRecommendation = 'Manual Investigation' THEN 1 ELSE 0 END) AS BIGINT) AS ManualInvestigationCount,
        CAST(SUM(CASE WHEN ProviderRiskLevel = 'High' THEN 1 ELSE 0 END) AS BIGINT) AS HighRiskClaims
    FROM claims
    {where_clause}
    """
    df = con.execute(query, params).fetchdf()

    # Get provider-level label counts
    prov_query = """
    SELECT 
        CAST(SUM(CASE WHEN PotentialFraud = 'Yes' THEN 1 ELSE 0 END) AS BIGINT) AS KnownFraudProviders,
        CAST(SUM(CASE WHEN RiskLevel = 'High' THEN 1 ELSE 0 END) AS BIGINT) AS HighRiskProviders,
        CAST(SUM(CASE WHEN RiskLevel = 'Medium' THEN 1 ELSE 0 END) AS BIGINT) AS MediumRiskProviders,
        CAST(SUM(CASE WHEN RiskLevel = 'Low' THEN 1 ELSE 0 END) AS BIGINT) AS LowRiskProviders
    FROM providers
    """
    prov_df = con.execute(prov_query).fetchdf()

    metrics = df.to_dict(orient="records")[0]
    metrics.update(prov_df.to_dict(orient="records")[0])
    
    # Ensure counts are clean integers (not floats with .0)
    int_keys = [
        "TotalClaims", "UniqueBeneficiaries", "UniqueProviders",
        "InpatientClaims", "OutpatientClaims", "FlaggedClaims",
        "ManualInvestigationCount", "HighRiskClaims", "KnownFraudProviders",
        "HighRiskProviders", "MediumRiskProviders", "LowRiskProviders"
    ]
    for k in int_keys:
        if k in metrics:
            metrics[k] = int(metrics[k] or 0)

    return metrics


def search_claims(
    claim_id: Optional[str] = None,
    bene_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    claim_type: Optional[str] = None,
    min_reimb: Optional[float] = None,
    max_reimb: Optional[float] = None,
    risk_level: Optional[str] = None,
    review_rec: Optional[str] = None,
    flagged_only: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    claim_ids: Optional[List[str]] = None,
    exclude_claim_ids: Optional[List[str]] = None,
    sort_by: str = "InscClaimAmtReimbursed",
    ascending: bool = False,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Searches claims table with dynamic filters, pagination, and sorting.
    """
    con = get_duckdb_connection(read_only=True)
    conditions = []
    params = []

    if claim_id:
        conditions.append("ClaimID ILIKE ?")
        params.append(f"%{claim_id.strip()}%")
    if bene_id:
        conditions.append("BeneID ILIKE ?")
        params.append(f"%{bene_id.strip()}%")
    if provider_id:
        conditions.append("Provider ILIKE ?")
        params.append(f"%{provider_id.strip()}%")
    if claim_type and claim_type != "All":
        conditions.append("ClaimType = ?")
        params.append(claim_type)
    if min_reimb is not None:
        conditions.append("InscClaimAmtReimbursed >= ?")
        params.append(min_reimb)
    if max_reimb is not None:
        conditions.append("InscClaimAmtReimbursed <= ?")
        params.append(max_reimb)
    if risk_level and risk_level != "All":
        conditions.append("ProviderRiskLevel = ?")
        params.append(risk_level)
    if review_rec and review_rec != "All":
        conditions.append("AIReviewRecommendation = ?")
        params.append(review_rec)
    claim_screening_expression = """
        COALESCE(ScreeningFlagsCount, 0) > 0
    """
    if flagged_only:
        conditions.append(claim_screening_expression)
    if date_from:
        conditions.append("ClaimStartDt >= CAST(? AS DATE)")
        params.append(date_from)
    if date_to:
        conditions.append("ClaimStartDt <= CAST(? AS DATE)")
        params.append(date_to)
    if claim_ids is not None:
        if claim_ids:
            conditions.append("ClaimID IN (" + ",".join("?" for _ in claim_ids) + ")")
            params.extend(claim_ids)
        else:
            conditions.append("1 = 0")
    if exclude_claim_ids:
        conditions.append("ClaimID NOT IN (" + ",".join("?" for _ in exclude_claim_ids) + ")")
        params.extend(exclude_claim_ids)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    order_direction = "ASC" if ascending else "DESC"

    # Whitelist sort column to avoid SQL injection
    allowed_sort_cols = [
        "InscClaimAmtReimbursed", "ClaimStartDt", "ClaimID",
        "BeneID", "Provider", "ScreeningFlagsCount", "ClaimDurationDays"
    ]
    if sort_by not in allowed_sort_cols:
        sort_by = "InscClaimAmtReimbursed"
    limit = min(max(int(limit), 1), 500)
    offset = max(int(offset), 0)

    # Count query
    count_sql = f"SELECT COUNT(*) AS total FROM claims{where_clause}"
    total_records = con.execute(count_sql, params).fetchone()[0]

    # Data query
    data_sql = f"""
    SELECT 
        ClaimID, BeneID, Provider, ClaimType, ClaimStartDt, ClaimEndDt,
        InscClaimAmtReimbursed, DeductibleAmtPaid, ClaimDurationDays, LengthOfStayDays,
        DiagnosisCount, ProcedureCount, ScreeningFlagsCount,
    ({claim_screening_expression}) AS StoredScreeningAlert,
    ProviderFraudProbability, ProviderRiskLevel, AIReviewRecommendation
    FROM claims
    {where_clause}
    ORDER BY {sort_by} {order_direction}
    LIMIT {limit} OFFSET {offset}
    """
    records_df = con.execute(data_sql, params).fetchdf()

    return {
        "total": total_records,
        "records": records_df,
        "page": (offset // limit) + 1,
        "pages": max(1, (total_records + limit - 1) // limit)
    }


def get_claim_by_id(claim_id: str) -> Optional[Dict[str, Any]]:
    """Fetches complete record of an individual claim by its unique ClaimID."""
    con = get_duckdb_connection(read_only=True)
    try:
        df = con.execute("SELECT * FROM claims WHERE ClaimID = ?", [claim_id.strip()]).fetchdf()
    finally:
        con.close()
    if df.empty:
        return None
    return df.to_dict(orient="records")[0]


def get_claim_screening_stats(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Fetches claim-type and record-key baselines used to explain stored screening flags."""
    con = get_duckdb_connection(read_only=True)
    stats: Dict[str, Any] = {
        "reimbursement_p99": None,
        "duplicate_count": None,
        "beneficiary_claim_count": None,
    }
    try:
        claim_type = claim.get("ClaimType")
        if claim_type is not None and pd.notna(claim_type):
            stats["reimbursement_p99"] = con.execute(
                """
                SELECT quantile_cont(InscClaimAmtReimbursed, 0.99)
                FROM claims
                WHERE ClaimType = ?
                """,
                [claim_type],
            ).fetchone()[0]

        bene_id = claim.get("BeneID")
        provider_id = claim.get("Provider")
        start_date = claim.get("ClaimStartDt")
        if (
            bene_id is not None and pd.notna(bene_id)
            and provider_id is not None and pd.notna(provider_id)
            and start_date is not None and pd.notna(start_date)
        ):
            stats["duplicate_count"] = con.execute(
                """
                SELECT COUNT(*)
                FROM claims
                WHERE BeneID = ? AND Provider = ? AND ClaimStartDt = ?
                """,
                [bene_id, provider_id, start_date],
            ).fetchone()[0]

        if bene_id is not None and pd.notna(bene_id):
            stats["beneficiary_claim_count"] = con.execute(
                "SELECT COUNT(*) FROM claims WHERE BeneID = ?",
                [bene_id],
            ).fetchone()[0]
    finally:
        con.close()
    return stats


def get_beneficiary_profile(bene_id: str) -> Optional[Dict[str, Any]]:
    """Fetches demographic and chronic health profile for a beneficiary."""
    con = get_duckdb_connection(read_only=True)
    try:
        df = con.execute("SELECT * FROM beneficiaries WHERE BeneID = ?", [bene_id.strip()]).fetchdf()
    finally:
        con.close()
    if df.empty:
        return None
    return df.to_dict(orient="records")[0]


def get_claims_by_beneficiary(bene_id: str) -> pd.DataFrame:
    """Fetches all historical claims for a given beneficiary sorted chronologically."""
    con = get_duckdb_connection(read_only=True)
    query = """
    SELECT 
        ClaimID, Provider, ClaimType, ClaimStartDt, ClaimEndDt,
        InscClaimAmtReimbursed, DeductibleAmtPaid, ClaimDurationDays, LengthOfStayDays,
        ClmAdmitDiagnosisCode,
        ClmDiagnosisCode_1, ClmDiagnosisCode_2, ClmDiagnosisCode_3,
        ClmDiagnosisCode_4, ClmDiagnosisCode_5, ClmDiagnosisCode_6,
        ClmDiagnosisCode_7, ClmDiagnosisCode_8, ClmDiagnosisCode_9,
        ClmDiagnosisCode_10,
        DiagnosisCount, ProcedureCount, ScreeningFlagsCount,
        ProviderFraudProbability, ProviderRiskLevel, AIReviewRecommendation
    FROM claims
    WHERE BeneID = ?
    ORDER BY ClaimStartDt ASC, ClaimID ASC
    """
    try:
        return con.execute(query, [bene_id.strip()]).fetchdf()
    finally:
        con.close()


def get_provider_profile(provider_id: str) -> Optional[Dict[str, Any]]:
    """Fetches complete provider record with ML risk scores and engineered features."""
    con = get_duckdb_connection(read_only=True)
    try:
        df = con.execute("SELECT * FROM providers WHERE Provider = ?", [provider_id.strip()]).fetchdf()
    finally:
        con.close()
    if df.empty:
        return None
    return df.to_dict(orient="records")[0]


def get_claims_by_provider(provider_id: str, limit: int = 100, offset: int = 0) -> pd.DataFrame:
    """Fetches sample of claims submitted by a given provider."""
    con = get_duckdb_connection(read_only=True)
    limit = min(max(int(limit), 1), 500)
    offset = max(int(offset), 0)
    query = """
    SELECT 
        ClaimID, BeneID, ClaimType, ClaimStartDt, ClaimEndDt,
        InscClaimAmtReimbursed, DeductibleAmtPaid, ClaimDurationDays,
        ClmAdmitDiagnosisCode, ClmDiagnosisCode_1, ClmDiagnosisCode_2,
        DiagnosisCount, ProcedureCount, ScreeningFlagsCount
    FROM claims
    WHERE Provider = ?
    ORDER BY InscClaimAmtReimbursed DESC, ClaimStartDt DESC, ClaimID ASC
    LIMIT ? OFFSET ?
    """
    try:
        return con.execute(query, [provider_id.strip(), limit, offset]).fetchdf()
    finally:
        con.close()


def get_top_providers(by: str = "TotalReimbursement", limit: int = 10) -> pd.DataFrame:
    """Fetches top providers ranked by volume or reimbursement."""
    con = get_duckdb_connection(read_only=True)
    allowed_cols = ["TotalReimbursement", "TotalClaims", "UniquePatients", "FraudProbability"]
    if by not in allowed_cols:
        by = "TotalReimbursement"
    query = f"""
    SELECT 
        Provider, PotentialFraud, RiskLevel, FraudProbability,
        TotalClaims, TotalReimbursement, AverageReimbursement, UniquePatients,
        InpatientClaims, OutpatientClaims
    FROM providers
    ORDER BY {by} DESC
    LIMIT {limit}
    """
    return con.execute(query).fetchdf()


def get_screening_claims(
    rule: str = "All",
    claim_type: str = "All",
    risk_level: str = "All",
    min_reimbursement: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    claim_id: Optional[str] = None,
    bene_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    case_ids: Optional[List[str]] = None,
    exclude_case_ids: Optional[List[str]] = None,
    limit: int = 500,
    offset: int = 0
) -> Dict[str, Any]:
    """Returns one evidence row per stored Medicare screening-rule trigger."""
    rule_columns = {
        "Reimbursement outlier": ("OUTLIER_REIMBURSEMENT", "Flag_ReimbursementOutlier"),
        "Repeated beneficiary/provider/date cluster": ("DUPLICATE_BILLING", "Flag_DuplicateBilling"),
        "Date/duration anomaly": ("DURATION_INCONSISTENCY", "Flag_DurationAnomaly"),
        "Complex outpatient procedure": ("COMPLEX_PROCEDURES", "Flag_ComplexProcedure"),
        "High-frequency beneficiary": ("HIGH_FREQ_BENEFICIARY", "Flag_HighFreqBeneficiary"),
    }
    if rule != "All" and rule not in rule_columns:
        raise ValueError(f"Unknown claim screening rule: {rule}")

    conditions = []
    params: List[Any] = []
    if claim_type != "All":
        conditions.append("ClaimType = ?")
        params.append(claim_type)
    if risk_level != "All":
        conditions.append("ProviderRiskLevel = ?")
        params.append(risk_level)
    if min_reimbursement is not None:
        conditions.append("InscClaimAmtReimbursed >= ?")
        params.append(float(min_reimbursement))
    if date_from:
        conditions.append("ClaimStartDt >= CAST(? AS DATE)")
        params.append(date_from)
    if date_to:
        conditions.append("ClaimStartDt <= CAST(? AS DATE)")
        params.append(date_to)
    if claim_id:
        conditions.append("ClaimID ILIKE ?")
        params.append(f"%{claim_id.strip()}%")
    if bene_id:
        conditions.append("BeneID ILIKE ?")
        params.append(f"%{bene_id.strip()}%")
    if provider_id:
        conditions.append("Provider ILIKE ?")
        params.append(f"%{provider_id.strip()}%")
    if case_ids is not None:
        if case_ids:
            conditions.append("ClaimID IN (" + ",".join("?" for _ in case_ids) + ")")
            params.extend(case_ids)
        else:
            conditions.append("1 = 0")
    if exclude_case_ids:
        conditions.append("ClaimID NOT IN (" + ",".join("?" for _ in exclude_case_ids) + ")")
        params.extend(exclude_case_ids)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    prepared = """
        WITH prepared AS (
            SELECT c.*,
                quantile_cont(InscClaimAmtReimbursed, 0.99)
                    OVER (PARTITION BY ClaimType) AS ClaimTypeP99,
                COUNT(*) OVER (PARTITION BY BeneID, Provider, ClaimStartDt) AS MatchingKeyCount,
                COUNT(*)                 OVER (PARTITION BY BeneID) AS BeneficiaryClaimCount
            FROM claims c
        ),
        matching AS (
            SELECT * FROM prepared
    """ + where_clause + """
        ),
    """
    selected_rules = [
        (label, rule_code, flag_column)
        for label, (rule_code, flag_column) in rule_columns.items()
        if rule == "All" or rule == label
    ]
    select_columns = """
        ClaimID, BeneID, Provider, ClaimType, ClaimStartDt, ClaimEndDt,
        InscClaimAmtReimbursed, DeductibleAmtPaid, ClaimDurationDays,
        LengthOfStayDays, DiagnosisCount, ProcedureCount, ClmAdmitDiagnosisCode,
        ClmDiagnosisCode_1, ClmDiagnosisCode_2, ClaimTypeP99, MatchingKeyCount,
        BeneficiaryClaimCount, Flag_ReimbursementOutlier,
        Flag_DuplicateBilling, Flag_DurationAnomaly, Flag_ComplexProcedure,
        Flag_HighFreqBeneficiary, ProviderRiskLevel
    """
    trigger_selects = [
        f"""
        SELECT {select_columns}, '{rule_code}' AS RuleCode, '{label}' AS Rule
        FROM matching
        WHERE COALESCE({flag_column}, FALSE)
        """
        for label, rule_code, flag_column in selected_rules
    ]
    ctes = prepared + "trigger_rows AS (" + " UNION ALL ".join(trigger_selects) + ")"
    limit = min(max(int(limit), 1), 500)
    offset = max(int(offset), 0)
    con = get_duckdb_connection(read_only=True)
    try:
        total, occurrences = con.execute(
            ctes + " SELECT COUNT(DISTINCT ClaimID), COUNT(*) FROM trigger_rows",
            params,
        ).fetchone()
        query = ctes + """
            SELECT * FROM trigger_rows
            ORDER BY ClaimStartDt DESC, ClaimID ASC, RuleCode ASC
            LIMIT ? OFFSET ?
        """
        records = con.execute(query, [*params, limit, offset]).fetchdf()
    finally:
        con.close()
    total = int(total or 0)
    occurrences = int(occurrences or 0)
    return {
        "total": total,
        "rule_trigger_occurrences": occurrences,
        "records": records,
        "page": (offset // limit) + 1,
        "pages": max(1, (occurrences + limit - 1) // limit)
    }


def get_screening_rule_audit() -> pd.DataFrame:
    """Compares stored rule flags against the documented recomputation."""
    con = get_duckdb_connection(read_only=True)
    query = """
        WITH prepared AS (
            SELECT c.*,
                quantile_cont(InscClaimAmtReimbursed, 0.99)
                    OVER (PARTITION BY ClaimType) AS ClaimTypeP99,
                COUNT(*) OVER (PARTITION BY BeneID, Provider, ClaimStartDt) AS MatchingKeyCount,
                COUNT(*) OVER (PARTITION BY BeneID) AS BeneficiaryClaimCount
            FROM claims c
        ),
        comparison AS (
            SELECT
                (InscClaimAmtReimbursed > ClaimTypeP99) AS ReimbursementOutlier,
                (MatchingKeyCount > 1) AS DuplicateKeyCluster,
                (
                    ClaimDurationDays < 0 OR LengthOfStayDays < 0
                    OR (ClaimStartDt IS NOT NULL AND ClaimEndDt IS NOT NULL AND ClaimStartDt > ClaimEndDt)
                    OR (AdmissionDt IS NOT NULL AND DischargeDt IS NOT NULL AND AdmissionDt > DischargeDt)
                ) AS DateOrDurationAnomaly,
                (ClaimType = 'Outpatient' AND ProcedureCount >= 3) AS ComplexOutpatientProcedure,
                (BeneficiaryClaimCount >= 15) AS HighFrequencyBeneficiary,
                Flag_ReimbursementOutlier AS StoredReimbursementOutlier,
                Flag_DuplicateBilling AS StoredDuplicateKeyCluster,
                Flag_DurationAnomaly AS StoredDateOrDurationAnomaly,
                Flag_ComplexProcedure AS StoredComplexOutpatientProcedure,
                Flag_HighFreqBeneficiary AS StoredHighFrequencyBeneficiary
            FROM prepared
        )
        SELECT 'Reimbursement outlier' AS Rule,
            SUM(CAST(ReimbursementOutlier AS INTEGER)) AS RecomputedClaims,
            SUM(CAST(StoredReimbursementOutlier AS INTEGER)) AS StoredClaims,
            SUM(CAST(ReimbursementOutlier != StoredReimbursementOutlier AS INTEGER)) AS Disagreements
        FROM comparison
        UNION ALL
        SELECT 'Same beneficiary/provider/start-date cluster',
            SUM(CAST(DuplicateKeyCluster AS INTEGER)),
            SUM(CAST(StoredDuplicateKeyCluster AS INTEGER)),
            SUM(CAST(DuplicateKeyCluster != StoredDuplicateKeyCluster AS INTEGER))
        FROM comparison
        UNION ALL
        SELECT 'Date or duration anomaly',
            SUM(CAST(DateOrDurationAnomaly AS INTEGER)),
            SUM(CAST(StoredDateOrDurationAnomaly AS INTEGER)),
            SUM(CAST(DateOrDurationAnomaly != StoredDateOrDurationAnomaly AS INTEGER))
        FROM comparison
        UNION ALL
        SELECT 'Complex outpatient procedure',
            SUM(CAST(ComplexOutpatientProcedure AS INTEGER)),
            SUM(CAST(StoredComplexOutpatientProcedure AS INTEGER)),
            SUM(CAST(ComplexOutpatientProcedure != StoredComplexOutpatientProcedure AS INTEGER))
        FROM comparison
        UNION ALL
        SELECT 'High-frequency beneficiary (>=15 claims in dataset window)',
            SUM(CAST(HighFrequencyBeneficiary AS INTEGER)),
            SUM(CAST(StoredHighFrequencyBeneficiary AS INTEGER)),
            SUM(CAST(HighFrequencyBeneficiary != StoredHighFrequencyBeneficiary AS INTEGER))
        FROM comparison
    """
    result = con.execute(query).fetchdf()
    con.close()
    return result.fillna(0)
