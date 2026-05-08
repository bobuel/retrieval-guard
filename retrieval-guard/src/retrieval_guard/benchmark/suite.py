"""
Structural near-miss test suite for retrieval generalization benchmarking.

Test categories (per Redis April 2026 research):
  - Negation flips:   "The drug is effective" vs "The drug is NOT effective"
  - Role reversals:   "Alice hired Bob" vs "Bob hired Alice"
  - Spatial flips:    "The key is above the box" vs "The key is below the box"
  - Binding errors:   "The red car hit the blue truck" vs "The blue car hit the red truck"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Category = Literal["negation", "role_reversal", "spatial", "binding"]


@dataclass
class NearMissPair:
    query: str
    positive: str          # semantically correct match
    hard_negative: str     # structurally similar but semantically opposite
    category: Category
    id: str = ""


# ---------------------------------------------------------------------------
# Built-in test cases
# ---------------------------------------------------------------------------

BUILTIN_PAIRS: list[NearMissPair] = [
    # --- Negation ---
    NearMissPair(
        id="neg_001",
        category="negation",
        query="Is the medication effective for treating migraines?",
        positive="The medication is effective for treating migraines.",
        hard_negative="The medication is not effective for treating migraines.",
    ),
    NearMissPair(
        id="neg_002",
        category="negation",
        query="Was the experiment successful?",
        positive="The experiment was successful.",
        hard_negative="The experiment was not successful.",
    ),
    NearMissPair(
        id="neg_003",
        category="negation",
        query="Is the system secure?",
        positive="The system is secure and has passed all audits.",
        hard_negative="The system is not secure and has failed all audits.",
    ),
    NearMissPair(
        id="neg_004",
        category="negation",
        query="Does the contract allow subletting?",
        positive="The contract allows subletting with landlord approval.",
        hard_negative="The contract does not allow subletting under any circumstances.",
    ),
    NearMissPair(
        id="neg_005",
        category="negation",
        query="Is the product FDA approved?",
        positive="The product has received FDA approval.",
        hard_negative="The product has not received FDA approval.",
    ),

    # --- Role reversals ---
    NearMissPair(
        id="role_001",
        category="role_reversal",
        query="Who acquired TechCorp?",
        positive="MegaCorp acquired TechCorp last quarter.",
        hard_negative="TechCorp acquired MegaCorp last quarter.",
    ),
    NearMissPair(
        id="role_002",
        category="role_reversal",
        query="Who filed the lawsuit?",
        positive="The employee filed a lawsuit against the company.",
        hard_negative="The company filed a lawsuit against the employee.",
    ),
    NearMissPair(
        id="role_003",
        category="role_reversal",
        query="Who trained whom?",
        positive="The senior engineer trained the junior engineer.",
        hard_negative="The junior engineer trained the senior engineer.",
    ),
    NearMissPair(
        id="role_004",
        category="role_reversal",
        query="Who owes money to whom?",
        positive="The client owes payment to the contractor.",
        hard_negative="The contractor owes payment to the client.",
    ),

    # --- Spatial ---
    NearMissPair(
        id="spatial_001",
        category="spatial",
        query="Where is the server relative to the firewall?",
        positive="The server is behind the firewall.",
        hard_negative="The server is in front of the firewall.",
    ),
    NearMissPair(
        id="spatial_002",
        category="spatial",
        query="Is the temperature above or below the threshold?",
        positive="The temperature is above the critical threshold.",
        hard_negative="The temperature is below the critical threshold.",
    ),
    NearMissPair(
        id="spatial_003",
        category="spatial",
        query="Is the backup stored upstream or downstream?",
        positive="The backup is stored upstream of the processing node.",
        hard_negative="The backup is stored downstream of the processing node.",
    ),

    # --- Binding ---
    NearMissPair(
        id="binding_001",
        category="binding",
        query="Which vehicle was at fault in the incident?",
        positive="The red car collided with the blue truck.",
        hard_negative="The blue car collided with the red truck.",
    ),
    NearMissPair(
        id="binding_002",
        category="binding",
        query="What caused the system failure?",
        positive="The primary database corrupted the secondary cache.",
        hard_negative="The secondary cache corrupted the primary database.",
    ),
    NearMissPair(
        id="binding_003",
        category="binding",
        query="Which component infected which?",
        positive="The compromised API key exposed the storage bucket.",
        hard_negative="The compromised storage bucket exposed the API key.",
    ),
]


def get_suite(categories: list[Category] | None = None) -> list[NearMissPair]:
    """Return test pairs, optionally filtered by category."""
    if categories is None:
        return BUILTIN_PAIRS
    return [p for p in BUILTIN_PAIRS if p.category in categories]
