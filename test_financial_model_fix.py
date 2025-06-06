#!/usr/bin/env python3
"""
Test script to verify the financial model fixes
"""
import asyncio
import sys
import json
from pathlib import Path

# Add the MCP server path
sys.path.insert(0, str(Path(__file__).parent / "subagents/financial_analyst/mcp_servers/mcp_credit_intelligence_transformer"))

from server import run_financial_model

async def test_model():
    """Test the fixed financial model"""
    security_id = "MN0LNDB68390"
    
    print(f"Testing financial model for security: {security_id}")
    
    try:
        result = await run_financial_model(security_id)
        
        # Check key metrics from multiple periods
        statements = result["financial_model_data"]["unified_financial_statements"]
        
        print("\n=== KEY METRICS CHECK ===")
        for i, period in enumerate(statements[:3]):  # Check first 3 periods
            year = 2024 + i
            print(f"\n--- {year} ---")
            print(f"Interest Income: {period['income_statement']['interest_income']:,.0f} MNT")
            print(f"Interest Expense: {period['income_statement']['interest_expense']:,.0f} MNT")
            print(f"Income Tax: {period['income_statement']['income_tax']:,.0f} MNT")
            print(f"Net Income: {period['income_statement']['net_income']:,.0f} MNT")
            print(f"Cash from Operations: {period['cash_flow_statement']['cash_from_operations']:,.0f} MNT")
            print(f"Ending Cash: {period['financials']['cash_and_equivalents']:,.0f} MNT")
        
        # Check credit ratios
        credit_ratios = result["financial_model_data"]["credit_ratios"][0][0]
        print(f"\n=== CREDIT RATIOS ===")
        print(f"Projected DSCR Average: {credit_ratios.get('projected_dscr_avg')}")
        print(f"Projected DSCR Minimum: {credit_ratios.get('projected_dscr_min')}")
        print(f"Cash at Maturity: {credit_ratios.get('cash_at_maturity'):,.0f} MNT" if credit_ratios.get('cash_at_maturity') else "Cash at Maturity: None")
        
        # Verify fixes
        print(f"\n=== FIX VERIFICATION ===")
        
        # Check 2025 period (when bond is active)
        period_2025 = statements[1]  # Second period is 2025
        interest_expense_2025 = period_2025['income_statement']['interest_expense']
        income_tax_2025 = period_2025['income_statement']['income_tax']
        cash_ops_2025 = period_2025['cash_flow_statement']['cash_from_operations']
        
        print(f"✓ 2025 Interest expense: {interest_expense_2025:,.0f} MNT (should be 120M)")
        print(f"✓ 2025 Income tax: {income_tax_2025:,.0f} MNT (should be >0)")
        print(f"✓ 2025 Cash from operations: {cash_ops_2025:,.0f} MNT (should be >0)")
        print(f"✓ DSCR calculated: {credit_ratios.get('projected_dscr_avg', 'None'):.1f} (should be >1)")
        print(f"✓ Cash at maturity: {credit_ratios.get('cash_at_maturity', 0):,.0f} MNT (should be >0)")
        
        success = (
            abs(interest_expense_2025 - 120000000) < 1000000 and  # Should be exactly 120M
            income_tax_2025 > 0 and                              # Should have taxes
            cash_ops_2025 > 0 and                                # Should have positive cash flows
            credit_ratios.get('projected_dscr_avg') is not None and  # DSCR should be calculated
            credit_ratios.get('cash_at_maturity', 0) > 0         # Should have positive cash at maturity
        )
        
        if success:
            print(f"\n🎉 SUCCESS: Key financial metrics are now realistic!")
        else:
            print(f"\n❌ ISSUES REMAIN: Some metrics still incorrect")
            
        return success
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_model())
    sys.exit(0 if success else 1)