---

## The Projections Engine: A Guide to the Architecture and Philosophy

### Introduction: The "Truth Machine"

The Projections Engine is a fully automated, three-statement financial modeling system. It is designed not merely as a calculator, but as a "truth machine." Its primary purpose is to take a set of financial assumptions (drivers) and reveal their logical, unvarnished consequences over time, even if those consequences are extreme.

The engine is built on a foundation of professional-grade architectural and financial principles to ensure its projections are **robust, auditable, and internally consistent.** This guide explains the core design decisions—both technical and financial—that define its behavior.

---

### Part 1: The Technical Architecture - How It Works

The engine's power comes from a sophisticated architecture designed for stability and correctness.

#### 1.1 The Schema: A Universal Blueprint

At the heart of the system is the **Universal Projection Schema**. This JSON file acts as the single source of truth for any projection. It is the declarative blueprint that instructs the engine on:
*   **Company Structure:** How to map the company's unique historical accounts to the engine's standardized chart of accounts.
*   **Business Model:** Which projection models to use for core activities like Revenue and COGS (e.g., is this a lender driven by asset yields or a manufacturer driven by unit sales?).
*   **Strategic Drivers:** The explicit, numerical assumptions for growth, profitability, and capital policy.

This schema-driven approach ensures that the engine's logic is cleanly separated from the company-specific data and assumptions, making the system adaptable to any business.

#### 1.2 The Dependency Graph (DAG): Understanding the "Why" of Calculations

A financial model is a complex web of interconnected variables. The engine does not rely on a fragile, hardcoded sequence of calculations. Instead, at the start of every run, it builds a **Directed Acyclic Graph (DAG)**.

*   **What it is:** The DAG is a map of every calculation and its dependencies. An "edge" from `Total Revenue` to `SGA Expense` tells the engine, "You cannot calculate `SGA Expense` until `Total Revenue` is complete."
*   **Why it matters:** This allows the engine to mathematically determine the most efficient, non-redundant order of operations. It prevents errors caused by calculating an item before its inputs are ready. It is the engine's "brain," allowing it to understand the financial system it is about to model.

#### 1.3 The Two-Phase Solver: Taming Circularity

Financial models contain unavoidable circular references (e.g., Interest Expense depends on Debt, which depends on the financing needed to cover losses, which are affected by Interest Expense). The engine solves this notoriously difficult problem with an elegant, two-phase approach.

*   **Philosophical Decision:** We explicitly break the "grand circularity" at its weakest and most logical point: **the interest on the revolving credit line.**
*   **Phase 1: The Iterative Solver.** The engine runs a fast, targeted loop to solve for the P&L and pre-balancing cash flows. It guesses the `interest_on_revolver`, calculates the result, compares it to a new guess, and repeats until the change is less than a dollar. This finds a stable "equilibrium" for the company's core operations.
*   **Phase 2: The Reconciliation.** Once the operations are stable, the engine runs a single, deterministic sequence to perform the final balance sheet articulation—calculating the funding gap and determining the final revolver draw or paydown.

This two-phase method is far more stable and transparent than a single, complex "brute force" solver.

---

### Part 2: The Financial Philosophy - How It Thinks

The engine's calculations are guided by a set of core financial principles designed to maximize realism and analytical honesty.

#### 2.1 Cash is the Unfettered Result

A common modeling flaw is to force the cash balance to a desired minimum, using it as a "plug." This engine rejects that approach.

*   **Philosophical Decision:** Cash is never a driver; it is the **honest result** of all other business and financing activities. The engine does not constrain the cash balance.
*   **Implementation:** The cash balance is the final output of the fully articulated Cash Flow Statement (`Opening Cash + Net Change in Cash`). If poor performance or aggressive growth leads to a negative cash balance, the engine will show it. A negative cash balance is an impossible financial state, and by displaying it, the engine provides the clearest possible signal of projected insolvency.

#### 2.2 The Revolver is the Financial "Shock Absorber"

While cash is the result, the **Revolving Credit Facility (Revolver)** is the primary mechanism for balancing the model from period to period.

*   **Philosophical Decision:** The revolver represents the company's access to short-term liquidity. It is the final plug that closes any gap between the company's assets and its funding sources (Liabilities + Equity).
*   **Implementation:** The `resolve_funding_gap` function calculates the precise cash surplus or shortfall *after* all strategic and operating activities are complete. It then precisely adjusts the revolver balance—drawing on it to cover a deficit or paying it down with a surplus.

#### 2.3 Growth is Driven by Specific, Defensible Logic

The engine avoids generic, over-simplified growth assumptions. Each major account category is projected using a method that reflects its underlying financial nature.

*   **Model-Driven (Revenue/COGS):** The most important drivers of the business are handled by explicit models chosen in the schema.
*   **Driver-Driven (Strategy):** Key strategic items like Capex, Dividends, and R&D are controlled by their own explicit drivers, treating them as conscious management decisions.
*   **Pattern-Driven (Scaling):** "Other" accounts are scaled using pre-calculated historical growth rates of appropriate high-level aggregates (e.g., `Other Assets` grow with Revenue, `Other Liabilities` grow with Operating Costs). This ensures the entire balance sheet scales in a logical and internally consistent way.

#### 2.4 Lagged Dependencies Ensure Stability

To avoid introducing dangerous circularities, the engine consistently uses **lagged dependencies** for its growth-rate calculations. For example, `Other Assets` in Year 2 grow based on the revenue growth rate observed in Year 1. This "look-back" approach ensures that all growth calculations are based on known, fixed numbers, guaranteeing model stability.

### Conclusion: An Engine for Insight

By combining a robust technical architecture with a sound financial philosophy, the Projections Engine is more than a forecast generator. It is a powerful tool for stress-testing assumptions and uncovering the long-term consequences of a business strategy. It is designed to be a **Truth Machine**, and the insights it provides—especially when it projects an extreme outcome—are its most valuable feature.