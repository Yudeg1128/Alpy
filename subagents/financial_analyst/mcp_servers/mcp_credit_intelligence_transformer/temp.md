Excellent. This is the perfect next step. By systematically identifying the remaining gaps based on the universal schema, we can create a clear and final to-do list to complete the engine's logic.

I have analyzed your schema against the engine's current calculation logic. Here is the definitive list of categories and named accounts that are currently being "Held Constant" and require new, specific calculation functions.

---

### I. Categories Requiring New Logic

These are categories where implementing a single, category-wide function in `FinancialCalculations` will dynamically handle all items (`item_1`, `item_2`, etc.) that the LLM maps within them.

1.  **Inventory (`inventory`)**
    *   **Schema Location:** `balance_sheet.assets.current_assets.inventory`
    *   **Required Logic:** The standard approach is to project inventory based on a **Days Inventory Outstanding (DIO)** ratio. This would involve:
        1.  Calculating the historical DIO in `_gather_projection_constants`.
        2.  Creating a `calculate_inventory_rollforward` function that uses the projected COGS and the DIO to determine the target inventory balance.

2.  **Tax Payables (`tax_payables`)**
    *   **Schema Location:** `balance_sheet.liabilities.current_liabilities.tax_payables`
    *   **Required Logic:** This should reflect the timing difference between when tax is expensed and when it's paid. A common method is to have it grow in line with `income_tax_expense`. A simple but effective logic would be: `Tax Payable(t) = Tax Payable(t-1) * (1 + Growth Rate of Income Tax Expense)`. A more complex version would model cash tax payments separately.

3.  **Other Current Assets (`other_current_assets`)**
    *   **Schema Location:** `balance_sheet.assets.current_assets.other_current_assets`
    *   **Required Logic:** These are often miscellaneous accounts like prepaid expenses. The most common and robust approach is to project them as a **percentage of revenue**. The calculation would be `Total Revenue(t) * Historical % of Revenue`.

4.  **Other Current Liabilities (`other_current_liabilities`)**
    *   **Schema Location:** `balance_sheet.liabilities.current_liabilities.other_current_liabilities`
    *   **Required Logic:** Similar to Other Current Assets, these are often projected as a **percentage of revenue** or sometimes COGS. Projecting as a percentage of revenue is a solid, defensible starting point.

5.  **Other Non-Current Assets (`other_non_current_assets`)**
    *   **Schema Location:** `balance_sheet.assets.non_current_assets.other_non_current_assets`
    *   **Required Logic:** Given their long-term nature and lack of specific drivers, the simplest and most common approach is to have these **grow in line with revenue** or hold them constant as a percentage of total assets. Growing with revenue is a good first implementation.

6.  **Other Non-Current Liabilities (`other_non_current_liabilities`)**
    *   **Schema Location:** `balance_sheet.liabilities.non_current_liabilities.other_non_current_liabilities`
    *   **Required Logic:** Same as Other Non-Current Assets. Projecting them to **grow in line with revenue** is a standard and safe approach.

7.  **Contra Assets (`contra_assets`)**
    *   **Schema Location:** `balance_sheet.assets.contra_assets`
    *   **Required Logic:** This is critical. The name is generic, but this category almost always contains **Accumulated Depreciation & Amortization**. The correct logic is a roll-forward: `Opening Balance + Current Period's Depreciation Expense + Current Period's Amortization Expense`. Note that the expenses are negative on the P&L, and this account has a credit (negative) balance, so the signs must be handled carefully.

---

### II. Named Accounts Requiring New Logic

These are specific, individually named accounts that need their own unique handlers in the `master_registry`.

1.  **`research_and_development`**
    *   **Schema Location:** `income_statement.operating_expenses.research_and_development`
    *   **Required Logic:** This is a major strategic expense. It should not be held constant. The standard method is to project it as a **percentage of revenue**, similar to SGA.

2.  **`other_operating_expenses`**
    *   **Schema Location:** `income_statement.operating_expenses.other_operating_expenses`
    *   **Required Logic:** Treat this identically to `research_and_development`—project as a **percentage of revenue**.

3.  **`contra_equity`**
    *   **Schema Location:** `balance_sheet.equity.contra_equity`
    *   **Required Logic:** This typically represents **Treasury Stock** (shares the company has repurchased). A simple but effective logic is to have it change based on a driver for `net_share_repurchases_value`, similar to how `common_stock` is driven by `net_share_issuance`. For an MVP, holding it constant is acceptable, but for completion, it needs a driver.

4.  **`other_equity` items (e.g., `other_equity_1`)**
    *   **Schema Location:** `balance_sheet.equity.other_equity`
    *   **Required Logic:** This is often Accumulated Other Comprehensive Income (AOCI). This is notoriously difficult to project as it depends on things like foreign currency translation and changes in derivative values. The industry standard for a projection model like this is to **hold it constant**. You can formalize this by creating a `do_nothing_and_hold_constant` function and mapping it, making the choice explicit.

---

This list provides a clear, structured roadmap. By implementing logic for these categories and named accounts, you will have addressed all the remaining "Hold Constant" gaps and brought the engine to a state of full logical and financial completeness.