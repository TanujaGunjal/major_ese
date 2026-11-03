import duckdb
import time

con = duckdb.connect("data/claims_fraud.duckdb", read_only=True)

# Test 1: Search by ClaimID
t0 = time.time()
res1 = con.execute("SELECT ClaimID, BeneID, Provider, ClaimType, InscClaimAmtReimbursed, AIReviewRecommendation FROM claims WHERE ClaimID = 'CLM46614'").fetchdf()
t1 = time.time()
print(f"Test 1 (ClaimID lookup): {len(res1)} rows in {(t1-t0)*1000:.2f} ms")
print(res1)

# Test 2: Search by BeneID
t0 = time.time()
res2 = con.execute("SELECT ClaimID, Provider, ClaimType, InscClaimAmtReimbursed FROM claims WHERE BeneID = 'BENE11001'").fetchdf()
t1 = time.time()
print(f"\nTest 2 (Beneficiary lookup): {len(res2)} rows in {(t1-t0)*1000:.2f} ms")

# Test 3: Search by Provider
t0 = time.time()
res3 = con.execute("SELECT COUNT(*) as claims, SUM(InscClaimAmtReimbursed) as total_reimb FROM claims WHERE Provider = 'PRV51001'").fetchdf()
t1 = time.time()
print(f"\nTest 3 (Provider claims agg): {len(res3)} rows in {(t1-t0)*1000:.2f} ms")
print(res3)

# Test 4: Overall overview metrics
t0 = time.time()
res4 = con.execute("""
    SELECT 
        COUNT(*) as TotalClaims,
        COUNT(DISTINCT BeneID) as UniqueBeneficiaries,
        COUNT(DISTINCT Provider) as UniqueProviders,
        SUM(InscClaimAmtReimbursed) as TotalReimbursement,
        AVG(InscClaimAmtReimbursed) as AvgReimbursement,
        SUM(CASE WHEN ClaimType = 'Inpatient' THEN 1 ELSE 0 END) as InpatientClaims,
        SUM(CASE WHEN ClaimType = 'Outpatient' THEN 1 ELSE 0 END) as OutpatientClaims,
        SUM(CASE WHEN ScreeningFlagsCount > 0 THEN 1 ELSE 0 END) as FlaggedClaims
    FROM claims
""").fetchdf()
t1 = time.time()
print(f"\nTest 4 (Executive metrics over 558,211 claims): computed in {(t1-t0)*1000:.2f} ms")
print(res4.to_dict(orient="records")[0])
