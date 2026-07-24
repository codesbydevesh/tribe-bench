"""NeuroCheck claims database: it loads, validates clean, and has no dup IDs.

This is the one shipped asset the resource paper rests on, so the DB staying
valid is a regression guard worth having.
"""

from neurocheck.claims import load_claims, validate_claims


def test_claims_db_validates_clean():
    errors = validate_claims()
    assert errors == [], f"claims.yaml has validation errors: {errors}"


def test_claims_count_at_least_50():
    assert len(load_claims()) >= 50


def test_claim_ids_unique():
    ids = [c.id for c in load_claims()]
    assert len(ids) == len(set(ids)), "duplicate claim IDs in claims.yaml"
