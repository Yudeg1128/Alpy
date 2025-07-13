# **JIT Schema-Driven Financial Projection Pipeline: Detailed Architecture**

## I. Core Philosophy: The "Configurable Machine" Model

The fundamental principle of this architecture is a strict and robust separation of concerns between contextual reasoning and deterministic execution. The system is designed as two distinct components:

1.  **The JIT Selector (LLM): The "Specialist Configurator"**
    The AI's role is not to create logic or formulas. Its sole, highly constrained task is to act as a specialist that analyzes the issuer's financial data and business context to **populate a pre-defined master template**. It is a data-mapper and model-selector, not a model-builder.

2.  **The Projections Engine: The "Configurable Machine"**
    The engine is a complete, self-contained, and deterministic financial projection machine. It is hardcoded with a master chart of accounts, a library of all possible projection methodologies, and an immutable understanding of accounting rules. It requires no financial context on its own; it merely executes the precise configuration provided to it by the populated schema.

This model ensures that all financial logic and accounting integrity are hardcoded and reliable, while the system's ability to adapt to any company is handled by the AI's configuration task. The output is always arithmetically perfect and balanced; its financial relevance is a direct function of the quality of the AI's configuration.

## II. The Master Contract: The Universal Projection Schema (v6.0)

The entire system revolves around a single, authoritative file: the populated `universal_projections_schema.json`. This file is the complete, self-contained instruction set for a single projection run.

### Key Architectural Principles of the Schema:

*   **It is the Master Template:** The schema defines the complete, exhaustive chart of accounts the engine understands. This includes dedicated, segregated slots for all types of revenue, expenses, assets, and liabilities, including the critical distinction between scheduled debt and the unplanned revolver plug.
*   **It is Self-Documenting:** The `__schema_definitions__` section acts as a built-in data dictionary, providing the LLM with unambiguous definitions for all driver structures (e.g., explaining what "short_term" and "terminal" trends mean).
*   **Drivers are Embedded:** Each projectable line item contains its own `projection_model` and `drivers` objects. This collocates the "what" (the line item) with the "how" (the method and assumptions), creating a clear, auditable structure.
*   **The Cash Flow Statement is 100% Derived:** The schema intentionally forbids the assignment of any projection models to Cash Flow Statement items. This enforces the architectural rule that the CFS is the final, balancing output derived from the changes on the Income Statement and Balance Sheet, guaranteeing statement articulation.

## III. The Engine's Core Logic: A Three-Stage Execution Model

The Projections Engine's core logic is designed for robustness and computational efficiency. It replaces a fragile, single-pass iterative solver with a structured, three-stage process powered by a Directed Acyclic Graph (DAG) and Strongly Connected Components (SCC) analysis.

### Phase I: Setup & Configuration

This phase occurs once at the beginning of a projection run.

1.  **Load & Parse:** The engine loads the single, JIT-populated `universal_projections_schema.json`. It uses its hardcoded knowledge of this structure to parse all line items, models, and drivers.
2.  **Hydrate Data:** Using the `historical_account_key` map from the schema, the engine populates its internal master data structure with the historical values. Items not present in the historicals (like the revolver) are correctly initialized at zero.
3.  **Hydrate Drivers:** The engine runs a **"Driver Hydration"** process, converting the structured driver objects (with `baseline`, `justification`, and `trends`) into simple `[Year1, Year2, ...]` time-series arrays that the calculation modules can directly consume.
4.  **Build Execution Graph:** This is the heart of the setup. The engine builds a complete dependency graph for all line items based on their formulas and hierarchical relationships.
    *   **SCC Analysis:** It performs **Strongly Connected Components (SCC)** analysis on the graph. This mathematically identifies any circular loops. In our design, it will precisely isolate the `revolver -> interest expense -> ...` loop as the single SCC.
    *   **Topological Sort:** It then performs a **topological sort on the meta-graph** (where the entire SCC is treated as one node). This produces the unambiguous, three-stage execution order: `[Pre-Circular Items]`, `[Circular Items]`, `[Post-Circular Items]`.

### Phase II: The Three-Stage Projection Loop

The engine iterates through each projection year, executing the three stages in order.

*   **Stage 1: Pre-Circular Calculation**
    *   **Action:** The engine executes the formulas for all items in the `[Pre-Circular Items]` list **once**.
    *   **Scope:** This typically includes revenue, COGS, operating expenses, and assumption-driven working capital and asset projections.
    *   **Result:** A fast, non-iterative calculation of the bulk of the financial model.

*   **Stage 2: Circular Solver (Targeted Iteration)**
    *   **Action:** The engine activates its iterative solver, but its scope is limited exclusively to the items within the `[Circular Items]` list identified by the SCC analysis.
    *   **Scope:** This is the `Revolver-Based Balancing` mechanism. The loop calculates the revolver balance, interest expense, and resulting cash flow impact until the cash balance converges to its target.
    *   **Result:** The circular reference is resolved efficiently and precisely without recalculating the entire model. The final revolver draw/paydown is determined.

*   **Stage 3: Post-Circular Calculation**
    *   **Action:** With the revolver balance now fixed, the engine executes the formulas for the final `[Post-Circular Items]` **once**.
    *   **Scope:** This typically includes final equity roll-forwards, totals that depend on the final revolver balance, and analytical ratios.
    *   **Result:** The period's Balance Sheet and Income Statement are complete.

### Phase III: Final Articulation

After the projection loop for all years is complete, the engine performs the final step: it constructs the entire **Cash Flow Statement** by analyzing the period-over-period changes in the now-complete Balance Sheet and incorporating the net income and other non-cash charges from the Income Statement. This guarantees the three statements are perfectly articulated.

## IV. Key Safeguards & Robustness Features

The engine is engineered to handle real-world data imperfections and configuration gaps gracefully.

*   **The "Null Model Handler":** This is a critical, hardcoded rule. If the engine encounters a projectable item where the JIT-populated schema has a `projection_model: null` (because the LLM lacked sufficient data to make a justified choice), it does not crash. Instead, it:
    1.  Applies a default, conservative projection formula: `value = [self:prior]` (i.e., the balance is held constant).
    2.  Writes a `WARNING` to the execution log, flagging the item and the applied default behavior for user review.
    This transforms a potential failure into a stable, auditable forecast.

*   **"T+1 Plug Neutralization":** The engine automatically identifies any `summation_plugs` in the final historical period. In the first projection year, it posts a one-time, offsetting reversal to a dedicated line item. This "launders" the historical data error, ensuring the projection begins from a clean, arithmetically pure base without contaminating the forecast with historical data quality issues.