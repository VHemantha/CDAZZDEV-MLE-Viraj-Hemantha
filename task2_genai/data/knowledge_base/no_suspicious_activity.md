# no_suspicious_activity

**Definition.** The alert is a false positive: activity is consistent with the customer's profile, declared turnover and a documented, plausible explanation.

**Key indicators**
- activity matches declared turnover/seasonality
- counterparties known and consistent with business
- documentation in KYC or analyst context explains the trigger

**How to distinguish.** Most monitoring alerts are false positives. Closing correctly matters as much as escalating: unnecessary SARs waste investigative capacity.

**Typical action.** close_no_action with risk_level low; request_information if one document would confirm the explanation.
