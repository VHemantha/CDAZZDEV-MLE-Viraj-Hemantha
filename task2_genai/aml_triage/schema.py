"""Pydantic contracts for alerts (model input) and triage decisions (model output),
plus the deterministic renderer that turns an alert into the user-turn text."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schemas for AML alerts and triage
# outputs with a deterministic alert-to-text renderer', Date: 2026-10-07

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config

Typology = Literal[tuple(config.TYPOLOGIES)]  # type: ignore[valid-type]
Risk = Literal[tuple(config.RISK_LEVELS)]  # type: ignore[valid-type]
Action = Literal[tuple(config.ACTIONS)]  # type: ignore[valid-type]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    direction: Literal["in", "out"]
    type: str = Field(min_length=2, max_length=40)
    amount: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=4)
    counterparty: str = Field(min_length=2, max_length=80)
    counterparty_country: str = Field(min_length=2, max_length=40)


class Customer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    segment: str
    occupation_or_business: str
    account_age_months: int = Field(ge=0, le=600)
    kyc_risk_rating: Literal["low", "medium", "high"]
    declared_monthly_turnover: float = Field(ge=0)
    turnover_currency: str = Field(min_length=3, max_length=4)
    residence_country: str


class Alert(BaseModel):
    model_config = ConfigDict(extra="ignore")
    alert_id: str = ""
    rule_triggered: str = Field(min_length=5)
    customer: Customer
    transactions: list[Transaction] = Field(min_length=config.MIN_TXNS, max_length=config.MAX_TXNS + 2)
    prior_alerts_12m: int = Field(ge=0, le=50)
    analyst_context: str = Field(default="", max_length=400)


class Triage(BaseModel):
    """The structured decision the student must learn to produce."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    typology: Typology
    risk_level: Risk
    red_flags: list[str] = Field(max_length=6)
    mitigating_factors: list[str] = Field(max_length=5)
    recommended_action: Action
    rationale: str = Field(min_length=20, max_length=700)

    @field_validator("typology", "risk_level", "recommended_action", mode="before")
    @classmethod
    def _norm(cls, v: object) -> object:
        return v.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(v, str) else v


class Case(BaseModel):
    case_id: str
    seed: dict
    teacher: str = ""
    alert: Alert
    triage: Triage


def render_alert(alert: Alert) -> str:
    """Deterministic, analyst-style alert text. Using one renderer for every case
    keeps the input format stable so the model learns the task, not the layout."""
    c = alert.customer
    lines = [
        f"ALERT {alert.alert_id} | Rule triggered: {alert.rule_triggered}",
        f"Customer: {c.segment} - {c.occupation_or_business}; resident in {c.residence_country}; "
        f"account age {c.account_age_months} months; KYC risk rating {c.kyc_risk_rating}; "
        f"declared monthly turnover {c.declared_monthly_turnover:,.0f} {c.turnover_currency}.",
        f"Prior alerts in last 12 months: {alert.prior_alerts_12m}.",
        "Transactions:",
        "date       | dir | type | amount | counterparty (country)",
    ]
    for t in alert.transactions:
        lines.append(f"{t.date} | {t.direction:<3} | {t.type} | {t.amount:,.2f} {t.currency} | "
                     f"{t.counterparty} ({t.counterparty_country})")
    if alert.analyst_context:
        lines.append(f"Context: {alert.analyst_context}")
    return "\n".join(lines)


def render_triage(triage: Triage) -> str:
    """Canonical assistant turn: compact JSON with a fixed key order."""
    return json.dumps(triage.model_dump(), ensure_ascii=False, indent=1)
