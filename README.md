# Lot Analyzer

A Python tool to find optimal stock lots to donate for maximum tax benefit, specifically tailored for high-income California residents.

## Features

- **Format Support**: Handles standard "Lot Details" CSV exports and the "Schwab Aligned" format.
- **Input Flexibility**: Load a single CSV file or an entire directory of CSVs using the `--input` switch.
- **Real-time Data**: Optional `--live` flag to fetch current market prices via `yfinance`.
- **Whole Shares Only**: Automatically handles the constraint that fractional shares cannot be donated by flooring available lots and rounding up recommendations to the next whole share.
- **Ineligibility Filtering**: Automatically excludes lots with less than 1 whole share.
- **Lot Disambiguation**: Shows "Lot Size" alongside purchase dates to help identify specific lots.
- **Tax Optimization**: Prioritizes highest unrealized gain percentage lots to maximize tax efficiency (deduction value + avoided capital gains tax).

## Installation

1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python lot_analyzer.py [amount_in_thousands] [--input PATH] [--live]
```

### Examples

- **Target $1,000,000 using default `assets/` directory**:
  ```bash
  python lot_analyzer.py 1000
  ```

- **Target $500,000 using a specific Schwab export**:
  ```bash
  python lot_analyzer.py 500 --input Schwab_Lots.csv
  ```

- **Use live prices for analysis**:
  ```bash
  python lot_analyzer.py 1000 --live
  ```

## Tax Logic (CA + Federal)

The tool calculates efficiency based on:
- **Combined Ordinary Rate (50.3%)**: Value of the charitable deduction.
- **Combined LTCG Rate (37.1%)**: Tax avoided by donating appreciated long-term assets.

**Note**: Short-term lots only receive a deduction for the cost basis, making them significantly less efficient for donation than long-term lots with high appreciation.
