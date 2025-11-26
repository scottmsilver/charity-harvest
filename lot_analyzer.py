#!/usr/bin/env python3
"""
Lot Analyzer - Find optimal lots to donate for maximum tax benefit

Usage: python lot_analyzer.py [amount_in_thousands]
  e.g. python lot_analyzer.py 500    # Target $500,000
       python lot_analyzer.py 2000   # Target $2,000,000

Tax rates for high-income California resident:
  Federal ordinary income:     37.0%
  Federal LTCG:                20.0% + 3.8% NIIT = 23.8%
  California (all income):     13.3%

  Combined ordinary income:    50.3%  (deduction value)
  Combined LTCG:               37.1%  (avoided if donating long-term)

IMPORTANT: Long-term donations get FULL MARKET VALUE deduction.
           Short-term donations only get COST BASIS deduction!
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=120)

# Tax rates for high-income California resident
FEDERAL_ORDINARY = 0.37
FEDERAL_LTCG = 0.238  # 20% + 3.8% NIIT
CA_RATE = 0.133

COMBINED_ORDINARY = FEDERAL_ORDINARY + CA_RATE  # 50.3% - deduction value
COMBINED_LTCG = FEDERAL_LTCG + CA_RATE  # 37.1% - avoided on LT donations

# IRS limits for charitable deductions of appreciated stock
# Donations of appreciated stock to public charities: 30% of AGI
# Excess can be carried forward for 5 years
AGI_LIMIT_APPRECIATED_STOCK = 0.30

# Display settings
TOP_N_CANDIDATES = 15


def parse_money(val: Any) -> int:
    """Parse money string to int (no cents)."""
    if pd.isna(val) or val in ("-", "--", ""):
        return 0
    
    val_str = str(val).strip()
    # Handle accounting format: ($1,234.56) -> -1234.56
    if val_str.startswith("(") and val_str.endswith(")"):
        val_str = "-" + val_str[1:-1]
        
    cleaned = val_str.replace("$", "").replace(",", "").strip()
    try:
        return int(round(float(cleaned)))
    except ValueError:
        return 0


def parse_pct(val: Any) -> float:
    """Parse percentage string to float."""
    if pd.isna(val) or val in ("-", "--", ""):
        return 0.0
    cleaned = str(val).replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_float(val: Any) -> float:
    """Parse float from string."""
    if pd.isna(val) or val in ("-", "--", ""):
        return 0.0
    cleaned = str(val).replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def format_money(val: int) -> str:
    """Format integer as money string with commas."""
    return f"${val:,}"


def calc_efficiency(gain_pct: float) -> float:
    """
    Calculate tax efficiency (tax benefit per dollar donated) for a given gain percentage.

    gain_pct is defined as (gain / cost_basis) × 100, e.g.:
      - 100% means the stock doubled (gain equals cost basis)
      - 50% means 50% gain on cost basis

    For efficiency, we need gain as a fraction of MARKET VALUE:
      gain_ratio = gain / market_value = gain / (cost + gain)
                 = (gain_pct/100) / (1 + gain_pct/100)
                 = gain_pct / (100 + gain_pct)

    Efficiency = deduction_rate + (gain_ratio × ltcg_rate)
    """
    if gain_pct <= 0:
        return COMBINED_ORDINARY  # No gain = just the deduction value

    gain_ratio = gain_pct / (100 + gain_pct)  # gain as fraction of market value
    return COMBINED_ORDINARY + (gain_ratio * COMBINED_LTCG)


def is_full_lot(fraction: float) -> bool:
    """Check if fraction represents a full lot (handles float precision)."""
    return math.isclose(fraction, 1.0, rel_tol=1e-9)


def calc_lot_values(row: pd.Series, fraction: float = 1.0) -> tuple[int, int, int, float, float, str, str]:
    """
    Calculate values for a lot, optionally with a fraction applied.
    Returns: (market_val, cost_basis, gain_loss, gain_pct, quantity, open_date, ticker)
    """
    full_market = parse_money(row.get("Market Value", 0))
    full_cost = parse_money(row.get("Cost Basis", 0))
    full_gain = parse_money(row.get("Gain/Loss ($)", 0))
    gain_pct = parse_pct(row.get("Gain/Loss (%)", 0))
    full_qty = parse_float(row.get("Quantity", 0))
    open_date = str(row.get("Open Date", ""))
    ticker = str(row.get("Ticker", ""))

    market_val = int(full_market * fraction)
    cost_basis = int(full_cost * fraction)
    gain_loss = int(full_gain * fraction)
    quantity = full_qty * fraction

    return market_val, cost_basis, gain_loss, gain_pct, quantity, open_date, ticker


def load_csv_file(filepath: str) -> pd.DataFrame:
    """Load a lot details CSV file and return DataFrame."""
    # Use utf-8-sig to handle BOM in Windows CSV exports
    with open(filepath, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()

    # Support tickers with dots or hyphens (e.g. BRK.B)
    ticker_match = re.match(r'^"?([\w.-]+)\s+Lot Details', first_line)
    ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"

    df = pd.read_csv(filepath, skiprows=2)

    if df.empty:
        return pd.DataFrame()

    df = df[df["Open Date"] != "Total"]
    df["Ticker"] = ticker

    return df


def load_all_lots(assets_dir: str) -> pd.DataFrame:
    """Load all lots from all CSV files in assets directory."""
    all_dfs = []
    csv_files = list(Path(assets_dir).glob("*.csv"))

    for filepath in csv_files:
        df = load_csv_file(str(filepath))
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def print_table(df: pd.DataFrame, title: str) -> None:
    """Print lots in a nicely formatted rich table."""
    if df.empty:
        console.print(Panel("[yellow]No lots found[/yellow]", title=title))
        return

    table = Table(title=title, show_header=True, header_style="bold cyan", expand=True)

    table.add_column("Ticker", style="bold", no_wrap=True)
    table.add_column("Date", no_wrap=True)
    table.add_column("Qty", justify="right", no_wrap=True)
    table.add_column("Market Val", justify="right", style="green", no_wrap=True)
    table.add_column("Cost Basis", justify="right", no_wrap=True)
    table.add_column("Gain", justify="right", style="bold green", no_wrap=True)
    table.add_column("Gain%", justify="right", style="bold yellow", no_wrap=True)
    table.add_column("Eff%", justify="right", style="bold cyan", no_wrap=True)

    total_market = 0
    total_cost = 0
    total_gain = 0

    for _, row in df.iterrows():
        market_val, cost_basis, gain_loss, gain_pct, quantity, open_date, ticker = calc_lot_values(row)

        total_market += market_val
        total_cost += cost_basis
        total_gain += gain_loss

        efficiency = calc_efficiency(gain_pct)

        table.add_row(
            ticker,
            open_date,
            f"{quantity:,.0f}",
            format_money(market_val),
            format_money(cost_basis),
            format_money(gain_loss),
            f"{gain_pct:.1f}%",
            f"{efficiency:.0%}",
        )

    # Add totals row
    avg_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
    avg_efficiency = calc_efficiency(avg_gain_pct)

    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        f"[bold]{format_money(total_market)}[/bold]",
        f"[bold]{format_money(total_cost)}[/bold]",
        f"[bold]{format_money(total_gain)}[/bold]",
        f"[bold]{avg_gain_pct:.1f}%[/bold]",
        f"[bold]{avg_efficiency:.0%}[/bold]",
        style="on dark_blue",
    )

    console.print(table)


def print_recommended_lots(df: pd.DataFrame, last_lot_fraction: float, target: int) -> None:
    """Print recommended lots with partial lot support."""
    if df.empty:
        console.print(Panel("[yellow]No lots found[/yellow]", title="Recommended"))
        return

    table = Table(
        title=f"RECOMMENDED LOTS TO DONATE (Target: {format_money(target)})",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )

    table.add_column("Ticker", style="bold", no_wrap=True)
    table.add_column("Date", no_wrap=True)
    table.add_column("Fraction", justify="right", no_wrap=True)
    table.add_column("Shares", justify="right", no_wrap=True)
    table.add_column("Market Val", justify="right", style="green", no_wrap=True)
    table.add_column("Cost Basis", justify="right", no_wrap=True)
    table.add_column("Gain", justify="right", style="bold green", no_wrap=True)
    table.add_column("Gain%", justify="right", style="bold yellow", no_wrap=True)
    table.add_column("Eff%", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Running", justify="right", style="magenta", no_wrap=True)

    running_total = 0
    total_market = 0
    total_cost = 0
    total_gain = 0
    num_lots = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        is_last = i == num_lots - 1
        frac = last_lot_fraction if is_last else 1.0

        market_val, cost_basis, gain_loss, gain_pct, quantity, open_date, ticker = calc_lot_values(row, frac)

        running_total += market_val
        total_market += market_val
        total_cost += cost_basis
        total_gain += gain_loss

        efficiency = calc_efficiency(gain_pct)

        # Show fraction (100% for full lots, percentage for partial)
        frac_str = "100%" if is_full_lot(frac) else f"[bold yellow]{frac:.1%}[/bold yellow]"

        table.add_row(
            ticker,
            open_date,
            frac_str,
            f"{quantity:,.0f}",
            format_money(market_val),
            format_money(cost_basis),
            format_money(gain_loss),
            f"{gain_pct:.1f}%",
            f"{efficiency:.0%}",
            format_money(running_total),
        )

    # Add totals row
    avg_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
    avg_efficiency = calc_efficiency(avg_gain_pct)

    # Count full lots vs partial
    full_lots = num_lots - 1 if not is_full_lot(last_lot_fraction) else num_lots
    lot_desc = f"{full_lots}" if is_full_lot(last_lot_fraction) else f"{full_lots} + partial"

    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        f"[bold]{lot_desc}[/bold]",
        "",
        f"[bold]{format_money(total_market)}[/bold]",
        f"[bold]{format_money(total_cost)}[/bold]",
        f"[bold]{format_money(total_gain)}[/bold]",
        f"[bold]{avg_gain_pct:.1f}%[/bold]",
        f"[bold]{avg_efficiency:.0%}[/bold]",
        "",
        style="on dark_blue",
    )

    console.print(table)


def select_lots_for_target(df: pd.DataFrame, target: int) -> tuple[pd.DataFrame, float]:
    """
    Select lots to reach target donation amount. Highest gain % first.
    Returns (selected_df, fraction_of_last_lot).
    The last lot may be partial to hit target exactly.
    """
    if df.empty:
        return df, 1.0

    df = df.copy()
    df["GainPctNum"] = df["Gain/Loss (%)"].apply(parse_pct)
    sorted_df = df.sort_values("GainPctNum", ascending=False)

    selected_indices = []
    total = 0
    fraction_of_last = 1.0

    for idx, row in sorted_df.iterrows():
        market_val = parse_money(row["Market Value"])

        if total >= target:
            break

        remaining_needed = target - total

        if market_val <= remaining_needed:
            # Take the whole lot
            selected_indices.append(idx)
            total += market_val
            fraction_of_last = 1.0
        else:
            # Take a fraction of this lot to hit target exactly
            selected_indices.append(idx)
            # Guard against division by zero
            if market_val > 0:
                fraction_of_last = remaining_needed / market_val
            else:
                fraction_of_last = 0.0
            total = target
            break

    return sorted_df.loc[selected_indices], fraction_of_last


def calc_selected_totals(selected: pd.DataFrame, last_lot_fraction: float) -> tuple[int, int, int]:
    """Calculate totals for selected lots including partial last lot."""
    selected_market = 0
    selected_cost = 0
    selected_gain = 0
    num_lots = len(selected)

    for i, (_, row) in enumerate(selected.iterrows()):
        is_last = i == num_lots - 1
        frac = last_lot_fraction if is_last else 1.0

        market_val, cost_basis, gain_loss, _, _, _, _ = calc_lot_values(row, frac)
        selected_market += market_val
        selected_cost += cost_basis
        selected_gain += gain_loss

    return selected_market, selected_cost, selected_gain


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal lots to donate for maximum tax benefit")
    parser.add_argument(
        "amount",
        nargs="?",
        type=int,
        default=1000,
        help="Target donation amount in thousands of dollars (default: 1000 = $1M)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, "assets")
    target_amount = args.amount * 1000

    console.print()
    console.print(
        Panel.fit(
            "[bold]LOT ANALYZER[/bold]\n" "Find optimal lots to donate for maximum tax benefit",
            border_style="blue",
        )
    )

    all_lots = load_all_lots(assets_dir)

    if all_lots.empty:
        console.print("[red]No lots found! Check the assets directory.[/red]")
        return

    long_term = all_lots[all_lots["Holding Period"] == "Long Term"].copy()
    short_term = all_lots[all_lots["Holding Period"] == "Short Term"].copy()

    total_lt_value = sum(parse_money(v) for v in long_term["Market Value"])
    total_lt_gain = sum(parse_money(v) for v in long_term["Gain/Loss ($)"])

    # Summary stats
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Label", style="dim")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Total lots loaded", f"{len(all_lots)}")
    stats_table.add_row("Long-term lots", f"{len(long_term)}")
    stats_table.add_row("Short-term lots", f"{len(short_term)}")
    stats_table.add_row("Long-term market value", format_money(total_lt_value))
    stats_table.add_row("Long-term unrealized gain", format_money(total_lt_gain))
    console.print(stats_table)
    console.print()

    # Warning if target exceeds available
    if target_amount > total_lt_value:
        console.print(
            f"[bold yellow]Warning: Target {format_money(target_amount)} exceeds "
            f"available long-term value {format_money(total_lt_value)}[/bold yellow]\n"
        )

    # Tax logic explanation
    console.print(
        Panel(
            "[bold yellow]WHY DONATE HIGHEST GAIN % LONG-TERM LOTS?[/bold yellow]\n\n"
            "[bold]CA + Federal Tax Rates (High Income):[/bold]\n"
            "  Ordinary income (deduction):  50.3%  (37% fed + 13.3% CA)\n"
            "  Long-term cap gains:          37.1%  (23.8% fed + 13.3% CA)\n\n"
            "[bold red]CRITICAL:[/bold red] Long-term → deduct FULL MARKET VALUE\n"
            "          Short-term → only deduct COST BASIS!\n\n"
            "[bold cyan]EFFICIENCY[/bold cyan] = Tax benefit per $1 donated (long-term only)\n"
            "  Formula: 50.3% + (gain/market_value) × 37.1%\n"
            "  [dim]200% gain (tripled) →  75% efficiency (75¢ per $1 donated)[/dim]\n"
            "  [dim]100% gain (doubled) →  69% efficiency (69¢ per $1 donated)[/dim]\n"
            "  [dim] 50% gain           →  63% efficiency[/dim]\n"
            "  [dim]  0% gain           →  50% efficiency (just the deduction)[/dim]\n\n"
            "[bold yellow]AGI LIMITATION:[/bold yellow] Appreciated stock deductions limited to 30% of AGI.\n"
            "  [dim]$1M donation requires $3.33M+ AGI. Excess carries forward 5 years.[/dim]",
            title="Tax Strategy (CA + Federal)",
            border_style="yellow",
        )
    )
    console.print()

    # Select optimal lots (may include partial last lot)
    selected, last_lot_fraction = select_lots_for_target(long_term, target_amount)

    # Show top lots by gain %
    long_term["GainPctNum"] = long_term["Gain/Loss (%)"].apply(parse_pct)
    top_by_gain = long_term.sort_values("GainPctNum", ascending=False).head(TOP_N_CANDIDATES)
    print_table(top_by_gain, f"TOP {TOP_N_CANDIDATES} LONG-TERM LOTS BY GAIN % (Best candidates)")

    console.print()

    # Show selected lots with partial lot support
    print_recommended_lots(selected, last_lot_fraction, target_amount)

    # Summary - calculate with partial lot applied to last lot
    selected_market, selected_cost, selected_gain = calc_selected_totals(selected, last_lot_fraction)
    num_lots = len(selected)

    avg_pct = (selected_gain / selected_cost * 100) if selected_cost > 0 else 0
    avg_eff = calc_efficiency(avg_pct)

    # Tax benefit calculations (CA + Federal)
    deduction_value = int(selected_market * COMBINED_ORDINARY)
    tax_saved_on_gains = int(selected_gain * COMBINED_LTCG)
    total_benefit = deduction_value + tax_saved_on_gains

    console.print()
    summary_table = Table(title="SUMMARY", show_header=False, border_style="green")
    summary_table.add_column("", style="dim")
    summary_table.add_column("", justify="right")
    summary_table.add_row("Target donation amount", format_money(target_amount))
    summary_table.add_row("Selected lots market value", format_money(selected_market))
    summary_table.add_row("Selected lots cost basis", format_money(selected_cost))
    summary_table.add_row("Total unrealized gain", f"[bold green]{format_money(selected_gain)}[/bold green]")
    summary_table.add_row("Average gain percentage", f"[bold yellow]{avg_pct:.1f}%[/bold yellow]")
    summary_table.add_row("Efficiency (tax benefit/$)", f"[bold cyan]{avg_eff:.0%}[/bold cyan]")
    full_lots = num_lots - 1 if not is_full_lot(last_lot_fraction) else num_lots
    lot_desc = (
        f"{num_lots}" if is_full_lot(last_lot_fraction) else f"{full_lots} full + 1 partial ({last_lot_fraction:.1%})"
    )
    summary_table.add_row("Lots to donate", lot_desc)
    console.print(summary_table)

    console.print()
    tax_table = Table(title="ESTIMATED TAX BENEFIT (CA + Federal)", show_header=False, border_style="cyan")
    tax_table.add_column("", style="dim")
    tax_table.add_column("", justify="right")
    tax_table.add_row(f"Deduction value ({COMBINED_ORDINARY:.1%} combined)", format_money(deduction_value))
    tax_table.add_row(f"LTCG tax avoided ({COMBINED_LTCG:.1%} combined)", format_money(tax_saved_on_gains))
    tax_table.add_row(
        "[bold]TOTAL ESTIMATED TAX BENEFIT[/bold]",
        f"[bold green]{format_money(total_benefit)}[/bold green]",
    )
    console.print(tax_table)

    # AGI requirement warning
    min_agi_required = int(selected_market / AGI_LIMIT_APPRECIATED_STOCK)
    console.print()
    console.print(
        f"[yellow]Note: To fully deduct {format_money(selected_market)} of appreciated stock, "
        f"you need AGI of at least {format_money(min_agi_required)} (30% limit).[/yellow]"
    )
    console.print("[dim]Excess deduction can be carried forward for up to 5 years.[/dim]")
    console.print()


if __name__ == "__main__":
    main()
