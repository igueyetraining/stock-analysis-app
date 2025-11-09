# ==============================================================================
# --- MERGED STOCK ANALYSIS PIPELINE ---
# This script combines the functionality of three separate scripts:
# 1. Finviz scanner for high-potential stocks.
# 2. YFinance data fetcher for historical OHLCV and indicators.
# 3. Pattern scanner (Cup-with-Handle, Stair) and PDF report generator.
#
# The script now runs as a single pipeline, passing data in-memory via
# pandas DataFrames, and produces a single PDF report as its final output.
#
# --- USAGE EXAMPLES ---
#
# 1. Run for the current date, scanning all stocks:
#    python merged_stock_analysis_pipeline.py
#
# 2. Run for a specific historical date:
#    python merged_stock_analysis_pipeline.py --endDate 2023-11-15
#
# 3. Analyze specific tickers for the current date (bypasses Finviz scan):
#    python merged_stock_analysis_pipeline.py --ticker NVDA AMD SMCI
#
# 4. Analyze a specific ticker for a specific historical date:
#    python merged_stock_analysis_pipeline.py --ticker AXTI --endDate 2023-10-22
# ==============================================================================

# --- COMMON IMPORTS ---
import pandas as pd
import yfinance as yf
import numpy as np
import time
import argparse
import sys
import os
import random
from datetime import datetime, timedelta
from io import BytesIO
from tqdm import tqdm

# --- Finviz Imports ---
from finvizfinance.screener.performance import Performance
from finvizfinance.quote import finvizfinance

# --- PDF Generation & Charting Imports ---
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    import mplfinance as mpf
    import matplotlib
    matplotlib.use('Agg') # Use a non-interactive backend for server-side execution
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("Error: Required libraries for PDF generation and charting not found.")
    print("Please run: pip install pandas fpdf2 mplfinance matplotlib yfinance finvizfinance tqdm")
    sys.exit(1)

# Suppress specific pandas warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==============================================================================
# --- PART 1: FINVIZ SCANNER & SCORER (from 1_Weekly_scan_...) ---
# ==============================================================================

# --- Configuration variables (Script 1) ---
HIGH_VOLUME_THRESHOLD = 20_000_000
MAX_PRICE_THRESHOLD = 70
MIN_PERF_UPTREND_THRESHOLD = 50.0
MIN_INSTITUTION_DAYS_THRESHOLD = 4
INSTITUTION_DAYS_LOOKBACK = 40

# --- Helper Functions (Script 1) ---
def convert_value_to_number(value):
    if isinstance(value, (int, float)): return int(value)
    if not isinstance(value, str): return 0
    try:
        value_str = value.strip().upper()
        if value_str == '-' or value_str == '': return 0
        elif value_str.endswith('M'): return int(float(value_str[:-1]) * 1_000_000)
        elif value_str.endswith('B'): return int(float(value_str[:-1]) * 1_000_000_000)
        elif value_str.endswith('K'): return int(float(value_str[:-1]) * 1_000)
        else: return int(float(value_str))
    except (ValueError, TypeError): return 0

def format_as_percentage(value):
    try: return f"{(float(value) * 100):.2f}%"
    except (ValueError, TypeError): return 'N/A'

def convert_percentage_to_float(value):
    try:
        if isinstance(value, (int, float)): return value * 100
        if isinstance(value, str) and '%' in value: return float(value.replace('%', ''))
        return 0.0
    except (ValueError, TypeError): return 0.0

def convert_sales_to_millions(value_str):
    try:
        if not isinstance(value_str, str): return 0.0
        value_str = value_str.strip().upper()
        if value_str.endswith('M'): return round(float(value_str[:-1]), 2)
        elif value_str.endswith('B'): return round(float(value_str[:-1]) * 1000, 2)
        elif value_str.endswith('K'): return round(float(value_str[:-1]) / 1000, 2)
        elif value_str == '-' or value_str == '': return 0.0
        else: return round(float(value_str), 2)
    except (ValueError, TypeError): return 0.0

def calculate_score(row, debug=False):
    if debug:
        print(f"\n--- Score Calculation for [{row.get('Ticker')}] ---")

    # --- 1. Sales Points ---
    sales_points = 0
    sales_qq_numeric = row.get('Sales Q/Q')
    if pd.isna(sales_qq_numeric): sales_qq_numeric = 0 # Handle potential NaN

    if sales_qq_numeric > 150: sales_points = 4
    elif 80 <= sales_qq_numeric < 150: sales_points = 3
    elif 20 <= sales_qq_numeric < 80: sales_points = 2
    if debug:
        print(f"1. Sales Q/Q Growth = {sales_qq_numeric:.2f}% -> Awarded {sales_points} points.")

    # --- 2. Price Points ---
    quarter_points, month_points = 0, 0
    perf_quart_numeric = row.get('Perf Quart')
    if pd.isna(perf_quart_numeric): perf_quart_numeric = 0

    if perf_quart_numeric > 200: quarter_points = 6
    elif 100 <= perf_quart_numeric < 200: quarter_points = 4
    elif 70 <= perf_quart_numeric < 100: quarter_points = 2

    perf_month_numeric = row.get('Perf Month')
    if pd.isna(perf_month_numeric): perf_month_numeric = 0

    if perf_month_numeric > 100: month_points = 6
    elif 70 <= perf_month_numeric < 100: month_points = 4
    elif 50 <= perf_month_numeric < 70: month_points = 2

    price_points = max(quarter_points, month_points)
    if debug:
        print(f"2. Price Momentum:")
        print(f"   - Perf Quarter = {perf_quart_numeric:.2f}% -> {quarter_points} points")
        print(f"   - Perf Month   = {perf_month_numeric:.2f}% -> {month_points} points")
        print(f"   -> Max of Price points is {price_points}.")

    # --- 3. Institution Points ---
    inst_points = 0
    inst_days = row.get('Days_Institution', 0)
    if inst_days > 15: inst_points = 6
    elif 10 <= inst_days <= 14: inst_points = 4
    elif 7 <= inst_days <= 9: inst_points = 2
    elif 4 <= inst_days <= 6: inst_points = 1
    if debug:
        print(f"3. Institution Days = {inst_days} -> Awarded {inst_points} points.")

    # --- 4. Volume Points ---
    avg_vol_points = 0
    avg_vol_numeric = convert_value_to_number(row.get('Avg Volume'))
    if avg_vol_numeric > 50_000_000: avg_vol_points = 6
    elif avg_vol_numeric > 20_000_000: avg_vol_points = 4
    elif avg_vol_numeric > 10_000_000: avg_vol_points = 2
    if debug:
        print(f"4. Average Volume = {avg_vol_numeric:,} -> Awarded {avg_vol_points} points.")

    # --- Total Score ---
    total_score = sales_points + price_points + inst_points + avg_vol_points
    if debug:
        print("----------------------------------------")
        print(f"TOTAL SCORE = {sales_points} + {price_points} + {inst_points} + {avg_vol_points} = {total_score}")
        print("="*40)

    return total_score

# --- Core Logic Functions (Script 1) ---
def perform_finviz_scans():
    print("="*80); print("--- [1A] FINVIZ SCANS & INITIAL PRICE FILTER ---"); print("="*80)
    FUNDAMENTAL_QUARTERLY_PERF, FUNDAMENTAL_MONTHLY_PERF = 50.0, 30.0
    MOMENTUM_QUARTERLY_PERF, MOMENTUM_MONTHLY_PERF = 70.0, 50.0
    base_filters = {'Average Volume': 'Over 1M', '20-Day Simple Moving Average': 'Price above SMA20','50-Day Simple Moving Average': 'SMA50 below SMA20', 'Price': 'Over $1', 'Industry': 'Stocks only (ex-Funds)'}
    f_q_filters = base_filters.copy(); f_q_filters.update({'Performance': 'Quarter +50%', 'Sales growthqtr over qtr': 'Over 20%'})
    f_m_filters = base_filters.copy(); f_m_filters.update({'Performance': 'Month +30%', 'Sales growthqtr over qtr': 'Over 20%'})
    m_q_filters = base_filters.copy(); m_q_filters.update({'Performance': 'Quarter +50%'})
    m_m_filters = base_filters.copy(); m_m_filters.update({'Performance': 'Month +50%'})
    h_v_filters = base_filters.copy(); h_v_filters.update({'Average Volume': 'Over 2M', 'Performance': 'Quarter +10%'})
    all_scans_config = [
        {'name': 'Fundamental Quarterly', 'type': 'Fundamental', 'filters': f_q_filters, 'threshold': FUNDAMENTAL_QUARTERLY_PERF, 'metric': 'Perf Quart'},
        {'name': 'Fundamental Monthly',   'type': 'Fundamental', 'filters': f_m_filters, 'threshold': FUNDAMENTAL_MONTHLY_PERF,   'metric': 'Perf Month'},
        {'name': 'Momentum Quarterly',    'type': 'Momentum',    'filters': m_q_filters, 'threshold': MOMENTUM_QUARTERLY_PERF,    'metric': 'Perf Quart'},
        {'name': 'Momentum Monthly',      'type': 'Momentum',    'filters': m_m_filters, 'threshold': MOMENTUM_MONTHLY_PERF,      'metric': 'Perf Month'},
        {'name': 'High Volume Scan',      'type': 'Momentum',    'filters': h_v_filters, 'threshold': -100.0, 'metric': 'Perf Month'},
    ]
    all_qualified_dfs, fundamental_tickers, momentum_tickers = [], set(), set()
    fperf = Performance()
    for config in all_scans_config:
        print(f"Running Scan: {config['name']}...")
        fperf.set_filter(filters_dict=config['filters'])
        try: df_perf = fperf.screener_view()
        except Exception as e: print(f"Error running finviz scan: {e}"); continue
        if df_perf is None or df_perf.empty: print("-> Found 0 tickers."); continue
        df_perf_filtered = pd.DataFrame()
        if config['name'] == 'High Volume Scan':
            print(f"-> Found {len(df_perf)} tickers from Finviz (using 'Over 2M' filter).")
            df_perf['Avg Volume_numeric'] = df_perf['Avg Volume'].apply(convert_value_to_number)
            df_perf_filtered = df_perf[df_perf['Avg Volume_numeric'] >= HIGH_VOLUME_THRESHOLD].copy()
            print(f"--> Kept {len(df_perf_filtered)} tickers after manually filtering for volume > {HIGH_VOLUME_THRESHOLD:,}.")
        else:
            metric_col, threshold = config['metric'], 'threshold'
            df_perf[f'{metric_col}_numeric'] = df_perf[metric_col].apply(convert_percentage_to_float)
            df_perf_filtered = df_perf[df_perf[f'{metric_col}_numeric'] >= config[threshold]].copy()
            print(f"--> Kept {len(df_perf_filtered)} tickers (Threshold: {config[threshold]}%).")
        if not df_perf_filtered.empty:
            all_qualified_dfs.append(df_perf_filtered)
            tickers = df_perf_filtered['Ticker'].tolist()
            if config['type'] == 'Fundamental': fundamental_tickers.update(tickers)
            elif config['type'] == 'Momentum': momentum_tickers.update(tickers)
    if not all_qualified_dfs: return None, None, None
    combined_perf_df = pd.concat(all_qualified_dfs, ignore_index=True)
    unique_perf_df = combined_perf_df.drop_duplicates(subset=['Ticker'], keep='first').copy()
    print(f"\nApplying manual price filter: Price <= ${MAX_PRICE_THRESHOLD}")
    unique_perf_df['Price'] = pd.to_numeric(unique_perf_df['Price'], errors='coerce')
    original_count = len(unique_perf_df)
    unique_perf_df.dropna(subset=['Price'], inplace=True)
    unique_perf_df = unique_perf_df[unique_perf_df['Price'] <= MAX_PRICE_THRESHOLD].copy()
    print(f"-> Filtered from {original_count} to {len(unique_perf_df)} unique tickers based on price.")
    if unique_perf_df.empty:
        print("No tickers remained after the manual price filter.")
        return None, None, None
    return unique_perf_df, fundamental_tickers, momentum_tickers

def fetch_finviz_performance_data(tickers_list):
    print("\n--- Fetching individual performance data from Finviz ---")
    perf_data_list = []
    for ticker in tqdm(tickers_list, desc="Fetching Performance"):
        try:
            stock = finvizfinance(ticker)
            all_info = stock.ticker_full_info()
            perf_data_list.append({
                'Ticker': ticker,
                'Perf Quarter': all_info.get('Perf Quarter', '0%'),
                'Perf Month': all_info.get('Perf Month', '0%')
            })
            time.sleep(0.2)
        except Exception as e:
            print(f"Could not fetch performance data for {ticker}: {e}")
            perf_data_list.append({'Ticker': ticker, 'Perf Quarter': '0%', 'Perf Month': '0%'})
    df = pd.DataFrame(perf_data_list)
    df.rename(columns={'Perf Quarter': 'Perf Quart'}, inplace=True)
    return df

def fetch_and_process_history_for_scanner(tickers, end_date):
    print("\n" + "="*80); print("--- [1B] YFINANCE HISTORY FOR SCANNER METRICS ---"); print("="*80)
    start_date = end_date - timedelta(days=190)
    yf_tickers = [str(t).replace('.', '-') for t in tickers]
    print(f"Downloading daily data for {len(yf_tickers)} tickers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    try:
        data = yf.download(yf_tickers, start=start_date, end=end_date + timedelta(days=1),
                           interval="1d", group_by='ticker', progress=False, threads=True)
    except Exception as e:
        print(f"Error downloading yfinance data: {e}")
        return None
    if data.empty:
        print("Warning: yfinance returned an empty DataFrame. No data for the given tickers/date range.")
        return None
    print("Calculating indicators (Volume MA50)...")
    stacked_df = data.stack(level=0).reset_index()
    stacked_df.rename(columns={'level_1': 'Ticker'}, inplace=True)
    if stacked_df.empty:
        print("Warning: No valid historical data was processed. The ticker might be invalid or have no data for the period.")
        return None
    for col in ['Close', 'Volume', 'High', 'Low']:
        stacked_df[col] = pd.to_numeric(stacked_df[col], errors='coerce')
    stacked_df.dropna(subset=['Close', 'Volume', 'High', 'Low'], inplace=True)
    stacked_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    if stacked_df.empty:
        print("Warning: All downloaded data was invalid after cleaning. No usable rows remain.")
        return None
    grouped = stacked_df.groupby('Ticker')
    stacked_df['Volume_MA50'] = grouped['Volume'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
    last_date_in_data = stacked_df['Date'].max(); start_date_4mo = last_date_in_data - pd.DateOffset(months=4)
    return stacked_df[stacked_df['Date'] >= start_date_4mo].copy()

def calculate_institution_days(hist_df):
    print("\n" + "="*80); print(f"--- [1C] CALCULATING INSTITUTION DAYS (LAST {INSTITUTION_DAYS_LOOKBACK} TRADING DAYS) ---"); print("="*80)
    results = {}
    if not pd.api.types.is_datetime64_any_dtype(hist_df['Date']): hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    for ticker, group in tqdm(hist_df.groupby('Ticker'), desc="Analyzing Volume"):
        if group.empty: continue
        recent_data = group.tail(INSTITUTION_DAYS_LOOKBACK).copy()
        mask = (recent_data['Volume'] > recent_data['Volume_MA50']) & (recent_data['Volume_MA50'].notna())
        orig_ticker = ticker.replace('-', '.')
        results[orig_ticker] = mask.sum()
    return pd.Series(results, name='Days_Institution')

def calculate_perf_uptrend(hist_df):
    print("\n" + "="*80); print("--- [1D] CALCULATING 'Perf UpTrend' METRIC ---"); print("="*80)
    results = {}
    if not pd.api.types.is_datetime64_any_dtype(hist_df['Date']): hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    for ticker, group in tqdm(hist_df.groupby('Ticker'), desc="Calculating UpTrend"):
        if group.empty: continue
        window_40d = group.tail(40).copy()
        window_20d = group.tail(20).copy()
        if window_40d.empty or window_20d.empty: continue
        lowest_low_40d = window_40d['Low'].min()
        highest_high_20d = window_20d['High'].max()
        orig_ticker = ticker.replace('-', '.')
        if pd.notna(lowest_low_40d) and pd.notna(highest_high_20d) and lowest_low_40d > 0:
            perf = (highest_high_20d - lowest_low_40d) / lowest_low_40d
            results[orig_ticker] = perf
        else:
            results[orig_ticker] = 0.0
    return pd.Series(results, name='Perf UpTrend')

def fetch_finviz_fundamentals(tickers_list):
    print("\n" + "="*80); print(f"--- [1E] FETCHING FINVIZ FUNDAMENTALS FOR {len(tickers_list)} STOCKS ---"); print("="*80)
    if not tickers_list:
        print("No tickers provided to fetch fundamentals.")
        return pd.DataFrame()
    overview_data_list = []
    for ticker in tqdm(sorted(tickers_list), desc="Fetching Fundamentals"):
        try:
            stock = finvizfinance(ticker); fund_data = stock.ticker_fundament(); fund_data['Ticker'] = ticker
            overview_data_list.append(fund_data); time.sleep(0.2)
        except Exception as e:
            print(f"Could not fetch fundamentals for {ticker}: {e}")
            pass
    if not overview_data_list:
        print("Could not retrieve any fundamental data.")
        return pd.DataFrame()
    return pd.DataFrame(overview_data_list)

# --- Orchestration function (replaces `main` from Script 1) ---
def run_scanner_and_scorer(args):
    """
    Orchestrates the Finviz scanning, data fetching, filtering, and scoring.
    Returns a DataFrame of qualified stocks.
    """
    if args.endDate:
        end_date = datetime.strptime(args.endDate, '%Y-%m-%d')
        print(f"Running analysis for historical date: {args.endDate}")
    else:
        end_date = datetime.now()
        print("Running analysis for current date.")

    if args.ticker:
        print("\n" + "="*80)
        print(f"--- TICKER MODE: Bypassing Finviz scan, analyzing: {', '.join(args.ticker)} ---")
        print("="*80)
        initial_tickers = [t.upper() for t in args.ticker]
        screener_df = pd.DataFrame({'Ticker': initial_tickers})
        perf_df = fetch_finviz_performance_data(initial_tickers)
        screener_df = pd.merge(screener_df, perf_df, on='Ticker', how='left')
        fundamental_tickers, momentum_tickers = set(), set()
    else:
        screener_df, fundamental_tickers, momentum_tickers = perform_finviz_scans()
        if screener_df is None or screener_df.empty:
            print("Process aborted: No stocks found after initial scans and price filter.")
            return None
        initial_tickers = screener_df['Ticker'].unique().tolist()

    hist_df = fetch_and_process_history_for_scanner(initial_tickers, end_date)
    if hist_df is None or hist_df.empty:
        print("Process aborted: No historical data could be fetched for scanner metrics.")
        return None

    inst_days_series = calculate_institution_days(hist_df)
    screener_df['Days_Institution'] = screener_df['Ticker'].map(inst_days_series).fillna(0).astype(int)

    perf_uptrend_series = calculate_perf_uptrend(hist_df)
    screener_df['Perf UpTrend'] = screener_df['Ticker'].map(perf_uptrend_series).fillna(0.0) * 100.0

    print("\n" + "="*80); print(f"--- [1F] FILTERING BY 'Perf UpTrend' (>= {MIN_PERF_UPTREND_THRESHOLD}%) ---"); print("="*80)
    original_count = len(screener_df)
    screener_df = screener_df[screener_df['Perf UpTrend'] >= MIN_PERF_UPTREND_THRESHOLD].copy()
    print(f"-> Filtered from {original_count} to {len(screener_df)} tickers based on Perf UpTrend.")
    if screener_df.empty:
        print(f"Process aborted: No stocks met the Perf UpTrend threshold of {MIN_PERF_UPTREND_THRESHOLD}%.")
        return None

    print("\n" + "="*80); print(f"--- [1G] FILTERING BY INSTITUTION DAYS (>= {MIN_INSTITUTION_DAYS_THRESHOLD}) ---"); print("="*80)
    original_count = len(screener_df)
    screener_df = screener_df[screener_df['Days_Institution'] >= MIN_INSTITUTION_DAYS_THRESHOLD].copy()
    print(f"-> Filtered from {original_count} to {len(screener_df)} tickers based on institution days.")
    if screener_df.empty:
        print(f"Process aborted: No stocks remained after filtering for at least {MIN_INSTITUTION_DAYS_THRESHOLD} institution days.")
        return None

    final_tickers = screener_df['Ticker'].unique().tolist()
    fundamentals_df = fetch_finviz_fundamentals(final_tickers)
    if fundamentals_df.empty:
        print("Process aborted: Could not fetch fundamental data for the final list of tickers.")
        return None

    print("\n" + "="*80); print("--- [1H] SCORING, MERGING & FINALIZING DATA ---"); print("="*80)
    base_df = pd.merge(fundamentals_df, screener_df, on='Ticker', how='left', suffixes=('_ov', '_perf'))
    base_df['Fundamental'] = base_df['Ticker'].isin(fundamental_tickers).apply(lambda x: 'x' if x else '')
    base_df['Momentum'] = base_df['Ticker'].isin(momentum_tickers).apply(lambda x: 'x' if x else '')

    if 'Avg Volume_ov' in base_df.columns:
        base_df['Avg Volume'] = base_df['Avg Volume_ov'].fillna(base_df.get('Avg Volume_perf'))
    if 'Price_perf' in base_df.columns:
        base_df['Price'] = base_df['Price_perf'].fillna(base_df.get('Price_ov'))

    rename_dict = {'Perf Month_perf': 'Perf Month', 'Perf Quart_perf': 'Perf Quart'}
    cols_to_rename = {k: v for k, v in rename_dict.items() if k in base_df.columns}
    if cols_to_rename:
        base_df.rename(columns=cols_to_rename, inplace=True)

    for col in ['Perf Quart', 'Perf Month', 'Sales Q/Q', 'EPS Q/Q']:
         if col in base_df.columns:
            base_df[col] = base_df[col].apply(convert_percentage_to_float)

    print("Calculating custom score for each stock...")
    # --- START OF FIX: The trigger for debug mode is now ONLY the --debug flag ---
    is_debug_mode = args.debug
    # --- END OF FIX ---
    
    if is_debug_mode:
        print("\n" + "="*25 + " DEBUG: Score Calculation Breakdown " + "="*25)

    base_df['Score'] = base_df.apply(lambda row: calculate_score(row, debug=is_debug_mode), axis=1)

    print("\nFormatting data for final score DataFrame...")
    if 'Sales' in base_df.columns:
        base_df['Sales'] = base_df['Sales'].apply(convert_sales_to_millions)
        base_df.rename(columns={'Sales': 'Sales (M)'}, inplace=True)
    for col in ['Market Cap', 'Avg Volume']:
        if col in base_df.columns: base_df[col] = base_df[col].apply(convert_value_to_number)

    final_cols = [
        'Ticker', 'Company', 'Sector', 'Industry', 'Score', 'Days_Institution',
        'Fundamental', 'Momentum',
        'Sales Q/Q', 'Sales (M)', 'EPS Q/Q', 'Perf UpTrend', 'Perf Quart', 'Perf Month',
        'Avg Volume', 'Price', 'Country', 'Market Cap'
    ]

    final_df = base_df[[c for c in final_cols if c in base_df.columns]].fillna(0)

    print("Applying final multi-level sort...")
    final_df = final_df.sort_values(
        by=['Score', 'Days_Institution', 'Perf Quart'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    print(f"\nSUCCESS! Part 1 (Scanner & Scorer) completed. Found {len(final_df)} qualified stocks.")

    print("\n" + "="*25 + " DEBUG: Final Scored Data (score_df) " + "="*25)
    print(f"DataFrame contains {len(final_df)} stocks.")
    print("Head (first 10 rows):")
    print(final_df.head(10).to_string())
    print("\nInfo (Data types and non-null counts):")
    final_df.info()
    print("="*80 + "\n")

    return final_df
    
# ==============================================================================
# --- PART 2: HISTORICAL DATA FETCHER (from 2_csv_stocks_data_fetcher_...) ---
# ==============================================================================
def fetch_historical_data_and_indicators(score_df, end_date_str=None):
    """
    Reads tickers from the score_df, adds static tickers, fetches historical data,
    calculates indicators, and returns the consolidated data as a DataFrame.
    """
    print("\n" + "="*80); print("--- [2A] FETCHING HISTORICAL OHLCV & CALCULATING INDICATORS ---"); print("="*80)
    # --- CONFIGURATION (Script 2) ---
    DATA_OUTPUT_PERIOD_MONTHS = 5
    DATA_DOWNLOAD_MONTHS = DATA_OUTPUT_PERIOD_MONTHS + 2
    STATIC_TICKERS_TO_ADD = ['QQQ', 'SPY', 'SQQQ', 'UVIX']

    # --- LOAD TICKERS FROM DATAFRAME ---
    if score_df is None or score_df.empty:
        print("Error: The provided score DataFrame is empty. Cannot fetch historical data.")
        return None
    tickers_from_df = score_df['Ticker'].dropna().unique().tolist()
    print(f"Loaded {len(tickers_from_df)} unique tickers from the scanner results.")

    print(f"Adding static tickers for market context: {STATIC_TICKERS_TO_ADD}")
    combined_tickers = tickers_from_df + STATIC_TICKERS_TO_ADD
    final_tickers_list = sorted(list(set(combined_tickers)))
    final_tickers_list_yf = [str(t).replace('.', '-') for t in final_tickers_list]
    print(f"Total unique tickers to download: {len(final_tickers_list_yf)}")

    # --- DETERMINE DATE RANGE ---
    try:
        if end_date_str:
            end_date = pd.to_datetime(end_date_str).date()
            print(f"Using specified end date: {end_date}")
        else:
            end_date = datetime.now().date()
            print(f"No end date specified. Using today: {end_date}")
        start_date_for_download = end_date - pd.DateOffset(months=DATA_DOWNLOAD_MONTHS)
        end_date_for_download = end_date + timedelta(days=1)
    except ValueError:
        print(f"Error: Invalid date format for --endDate. Please use YYYY-MM-DD.")
        return None

    # --- DOWNLOAD DATA ---
    print(f"Downloading data from {start_date_for_download.date()} to {end_date} to calculate indicators...")
    try:
        data = yf.download(final_tickers_list_yf, start=start_date_for_download, end=end_date_for_download,
                           interval="1d", group_by='ticker', progress=True)
        if data.empty:
            print("\nError: No data was downloaded. Check tickers or date range.")
            return None
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        return None

    # --- RESHAPE DATA AND CALCULATE INDICATORS ---
    print("Reshaping data and calculating indicators...")
    if len(final_tickers_list_yf) == 1:
        master_df = data.copy()
        master_df['Ticker'] = final_tickers_list_yf[0]
        master_df.reset_index(inplace=True)
    else:
        master_df = data.stack(level=0).reset_index()
        master_df.rename(columns={'level_1': 'Ticker'}, inplace=True)

    master_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    grouped = master_df.groupby('Ticker')
    master_df['Volume_MA50'] = grouped['Volume'].transform(lambda x: x.rolling(window=50, min_periods=50).mean())
    delta = grouped['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.transform(lambda x: x.ewm(com=13, adjust=False).mean())
    avg_loss = loss.transform(lambda x: x.ewm(com=13, adjust=False).mean())
    rs = avg_gain / avg_loss
    master_df['RSI14'] = 100 - (100 / (1 + rs))
    master_df['8DEMA'] = grouped['Close'].transform(lambda x: x.ewm(span=8, adjust=False).mean())
    master_df['21DEMA'] = grouped['Close'].transform(lambda x: x.ewm(span=21, adjust=False).mean())

    # --- TRIM DATAFRAME & FORMAT ---
    last_date_in_data = master_df['Date'].max()
    start_date_for_output = last_date_in_data - pd.DateOffset(months=DATA_OUTPUT_PERIOD_MONTHS)
    final_df = master_df[master_df['Date'] >= start_date_for_output].copy()
    final_df.columns = [c.lower() for c in final_df.columns]
    final_df['ticker'] = final_df['ticker'].str.replace('-', '.', regex=False)

    # --- START OF FIX: Standardize ticker data to lowercase ---
    final_df['ticker'] = final_df['ticker'].str.lower()
    # --- END OF FIX ---


    print("\nSUCCESS! Part 2 (Historical Data) completed.")

    print("\n" + "="*25 + " DEBUG: Full Historical Data (historical_df) " + "="*25)
    print(f"DataFrame shape: {final_df.shape}")
    print("Summary of data downloaded per ticker:")
    with pd.option_context('display.max_rows', None):
        print(final_df.groupby('ticker')['date'].agg(['min', 'max', 'count']))
    print("\nTail of data for SPY (to check latest date):")
    if 'spy' in final_df['ticker'].unique():
        print(final_df[final_df['ticker'] == 'spy'].tail(5).to_string())
    else:
        print("SPY data not found.")
    print("="*80 + "\n")

    return final_df
    
# ==============================================================================
# --- PART 3: PATTERN SCANNER & PDF REPORTER (from 4_cah_cup_in_handle_...) ---
# ==============================================================================

# --- CONFIGURABLE PARAMETERS (Script 3) ---
CONFIG = {
    'lrc_new_high_lookback': 20, 'lrc_uptrend_lookback_days': 60, 'min_prior_uptrend_pct': 60.0,
    'lrc_confirmation_dip_days': 3, 'min_days_to_rrc': 4, 'max_days_to_rrc': 21, 'rrc_high_lookback': 4,
    'max_rrc_height_pct_vs_lrc': 10.0, 'min_bc_rsi': 40.0, 'handle_min_duration': 1, 'handle_max_duration': 6,
    'stair_lth_lookback': 13, 'stair_lth_prev_high_lookback': 10, 'stair_lth_confirmation_days': 2,
    'stair_ltl_lookback': 20, 'stair_min_rise_pct': 30.0, 'stair_rsth_lookforward': 13,
    'stair_rsth_recency_days': 3,
    'cah_hb_recency_days': 3,
    'min_score_threshold': 6
}

# --- PATTERN SCANNING FUNCTIONS (Script 3) ---
def calculate_in_handle_metrics(df, lrc_idx, rrc_idx, handle_date_idx, bc_idx, bh_idx, breakout_entry_point):
    lrc_data = df.iloc[lrc_idx]; rrc_data = df.iloc[rrc_idx]; bc_data = df.iloc[bc_idx]; bh_data = df.iloc[bh_idx]; scan_data = df.iloc[handle_date_idx]
    uptrend_lookback_start_idx = lrc_idx - CONFIG['lrc_uptrend_lookback_days']
    if uptrend_lookback_start_idx < 0: return None
    lowest_low_prior = df.iloc[uptrend_lookback_start_idx:lrc_idx]['low'].min()
    prior_uptrend_pct = ((lrc_data['high'] / lowest_low_prior) - 1) * 100
    cup_depth_pct = ((bc_data['low'] / lrc_data['high']) - 1) * 100
    cup_duration = rrc_idx - lrc_idx
    ratio_cup_depth_vs_uptrend = (cup_depth_pct / prior_uptrend_pct) * 100 if prior_uptrend_pct > 0 else 0
    handle_depth_pct = ((bh_data['low'] / rrc_data['high']) - 1) * 100
    handle_duration = handle_date_idx - rrc_idx
    low_pt_range = lrc_data['high'] - bc_data['low']
    low_price_target = round(bc_data['low'] + (low_pt_range * 3.618), 2)
    high_pt_range = lrc_data['high'] - lowest_low_prior
    high_price_target = round(lowest_low_prior + (high_pt_range * 2.618), 2)
    conservative_sl_pct = ((bh_data['low'] / breakout_entry_point) - 1) * 100 if breakout_entry_point > 0 else 0
    dsma_vol_50 = scan_data['volume_ma50']
    right_cup_depth_pct = ((rrc_data['high'] / bc_data['low']) - 1) * 100 if bc_data['low'] > 0 else 0
    return {'prior_uptrend_pct': prior_uptrend_pct, 'cup_depth_pct': cup_depth_pct, 'cup_duration': cup_duration,
            'ratio_cup_depth_vs_uptrend': ratio_cup_depth_vs_uptrend, 'handle_depth_pct': handle_depth_pct,
            'handle_duration': handle_duration, 'conservative_sl_pct': conservative_sl_pct,
            'high_price_target': high_price_target, 'right_cup_depth_pct': right_cup_depth_pct,
            'low_price_target': low_price_target, 'dsma_vol_50': dsma_vol_50}

def calculate_stair_metrics(df, lstl_idx, lsth_idx, rstl_idx, scan_idx):
    lstl_data = df.iloc[lstl_idx]; lsth_data = df.iloc[lsth_idx]; rstl_data = df.iloc[rstl_idx]; scan_data = df.iloc[scan_idx]
    breakout_entry_point = lsth_data['high']
    prior_uptrend_pct = ((lsth_data['high'] / lstl_data['low']) - 1) * 100
    conservative_sl_pct = ((rstl_data['low'] / breakout_entry_point) - 1) * 100
    low_pt_range = lsth_data['high'] - rstl_data['low']
    low_price_target = round(rstl_data['low'] + (low_pt_range * 3.618), 2)
    high_pt_range = lsth_data['high'] - lstl_data['low']
    high_price_target = round(lstl_data['low'] + (high_pt_range * 2.618), 2)
    return {'prior_uptrend_pct': prior_uptrend_pct, 'breakout_entry_point': breakout_entry_point,
            'conservative_sl_pct': conservative_sl_pct, 'high_price_target': high_price_target,
            'low_price_target': low_price_target, 'dsma_vol_50': scan_data['volume_ma50']}

def find_in_handle_patterns(ticker, df, scan_date):
    scan_indices = df.index[df['date'] == scan_date].tolist()
    if not scan_indices: return []
    scan_idx = scan_indices[0]
    found_patterns = []
    for duration in range(CONFIG['handle_min_duration'], CONFIG['handle_max_duration'] + 1):
        rrc_idx = scan_idx - duration
        if rrc_idx <= 0: continue
        rrc_data = df.iloc[rrc_idx]
        if not (rrc_data['high'] > df.iloc[max(0, rrc_idx - CONFIG['rrc_high_lookback']) : rrc_idx]['high'].max()): continue
        if not (rrc_idx + 1 < len(df) and df.iloc[rrc_idx + 1]['high'] < rrc_data['high']): continue
        breakout_entry_point = round(rrc_data['high'], 2)
        for lrc_idx in range(rrc_idx - CONFIG['min_days_to_rrc'] - 1, rrc_idx - CONFIG['max_days_to_rrc'] - 1, -1):
            if lrc_idx < CONFIG['lrc_uptrend_lookback_days']: continue
            lrc_data = df.iloc[lrc_idx]
            if not (rrc_data['high'] <= (lrc_data['high'] * (1 + CONFIG['max_rrc_height_pct_vs_lrc'] / 100))): continue
            if not (lrc_data['high'] > df.iloc[max(0, lrc_idx - CONFIG['lrc_new_high_lookback']) : lrc_idx]['high'].max()): continue
            lookback_period = df.iloc[max(0, lrc_idx - CONFIG['lrc_uptrend_lookback_days']) : lrc_idx]
            if lookback_period.empty: continue
            prior_uptrend = ((lrc_data['high'] / lookback_period['low'].min()) - 1) * 100
            if not (prior_uptrend >= CONFIG['min_prior_uptrend_pct']): continue
            dip_period = df.iloc[lrc_idx + 1 : lrc_idx + 1 + CONFIG['lrc_confirmation_dip_days']]
            if not (len(dip_period) == CONFIG['lrc_confirmation_dip_days'] and dip_period['high'].max() <= lrc_data['high']): continue
            vol_ok = (df.iloc[lrc_idx-1]['volume'] > df.iloc[lrc_idx-1]['volume_ma50'] and lrc_data['volume'] > lrc_data['volume_ma50']) or \
                     (lrc_data['volume'] > lrc_data['volume_ma50'] and df.iloc[lrc_idx+1]['volume'] > df.iloc[lrc_idx+1]['volume_ma50'])
            if not vol_ok: continue
            cup_period = df.iloc[lrc_idx + 1 : rrc_idx]
            if cup_period.empty: continue
            bc_idx = cup_period['low'].idxmin()
            if not (df.loc[bc_idx]['rsi14'] > CONFIG['min_bc_rsi']): continue
            handle_period = df.iloc[rrc_idx + 1 : scan_idx + 1]
            if handle_period.empty: continue
            bh_idx = handle_period['low'].idxmin()
            if not (df.loc[bh_idx]['low'] > df.loc[bc_idx]['low']): continue
            if '21dema' not in handle_period.columns: continue
            if not (handle_period['close'] >= handle_period['21dema']).all(): continue
            metrics = calculate_in_handle_metrics(df, lrc_idx, rrc_idx, scan_idx, bc_idx, bh_idx, breakout_entry_point)
            if metrics is None: continue
            hb_idx, days_since_hb, breakout_condition_type = None, -1, None
            for current_day_idx in range(rrc_idx + 2, scan_idx + 1):
                current_day_data = df.iloc[current_day_idx]; bh_data = df.iloc[bh_idx]
                is_green_candle = current_day_data['close'] > current_day_data['open']
                midpoint_handle = (rrc_data['high'] + bh_data['low']) / 2
                is_quality_breakout = (is_green_candle and current_day_data['close'] > midpoint_handle and current_day_data['close'] > current_day_data['8dema'] and current_day_data['close'] > current_day_data['21dema'] and current_day_data['volume'] > current_day_data['volume_ma50'])
                is_price_level_breakout = (is_green_candle and current_day_data['close'] > rrc_data['high'])
                if is_quality_breakout: hb_idx, days_since_hb, breakout_condition_type = current_day_idx, scan_idx - current_day_idx, 1; break
                elif is_price_level_breakout: hb_idx, days_since_hb, breakout_condition_type = current_day_idx, scan_idx - current_day_idx, 2; break
            pattern_info = {'Ticker': ticker.upper(), **metrics, 'breakout_entry_point': breakout_entry_point, 'scan_date': scan_date, 'LRC_Date': lrc_data['date'], 'LRC_High': lrc_data['high'], 'BC_Date': df.loc[bc_idx]['date'], 'BC_Low': df.loc[bc_idx]['low'], 'BC_RSI14': df.loc[bc_idx]['rsi14'], 'RRC_Date': rrc_data['date'], 'RRC_High': rrc_data['high'], 'RRC_Close': rrc_data['close'], 'BH_Date': df.loc[bh_idx]['date'], 'BH_Low': df.loc[bh_idx]['low'], 'HB_Date': None, 'days_since_hb': -1, 'breakout_condition_type': None}
            if hb_idx is not None and days_since_hb >= 0:
                pattern_info.update({'Pattern_Type': 'Completed C/H', 'HB_Date': df.iloc[hb_idx]['date'], 'days_since_hb': days_since_hb, 'breakout_condition_type': breakout_condition_type})
                if days_since_hb < CONFIG['cah_hb_recency_days']: found_patterns.append(pattern_info)
            elif df.iloc[scan_idx]['close'] <= rrc_data['high']:
                pattern_info['Pattern_Type'] = 'On Going C/H'
                found_patterns.append(pattern_info)
    if found_patterns:
        completed = sorted([p for p in found_patterns if p['Pattern_Type'] == 'Completed C/H'], key=lambda p: p['HB_Date'], reverse=True)
        ongoing = sorted([p for p in found_patterns if p['Pattern_Type'] == 'On Going C/H'], key=lambda p: p['LRC_Date'], reverse=True)
        if completed: return [completed[0]]
        if ongoing: return [ongoing[0]]
    return []

def find_stair_patterns(ticker, df, scan_date):
    scan_indices = df.index[df['date'] == scan_date].tolist()
    if not scan_indices: return []
    scan_idx = scan_indices[0]; scan_data = df.iloc[scan_idx]
    if scan_data['close'] <= scan_data['21dema']: return []
    for lsth_idx in range(scan_idx, scan_idx - CONFIG['stair_lth_lookback'], -1):
        if lsth_idx < CONFIG['stair_lth_prev_high_lookback'] + CONFIG['stair_lth_confirmation_days']: continue
        lsth_data = df.iloc[lsth_idx]
        prev_10_days = df.iloc[max(0, lsth_idx - CONFIG['stair_lth_prev_high_lookback']) : lsth_idx]
        if lsth_data['high'] <= prev_10_days['high'].max(): continue
        next_2_days = df.iloc[lsth_idx + 1 : lsth_idx + 1 + CONFIG['stair_lth_confirmation_days']]
        if len(next_2_days) < CONFIG['stair_lth_confirmation_days'] or lsth_data['high'] <= next_2_days['high'].max(): continue
        lstl_lookback_period = df.iloc[max(0, lsth_idx - CONFIG['stair_ltl_lookback']) : lsth_idx]
        if lstl_lookback_period.empty: continue
        lstl_idx = lstl_lookback_period['low'].idxmin(); lstl_data = df.iloc[lstl_idx]
        if ((lsth_data['high'] / lstl_data['low']) - 1) * 100 < CONFIG['stair_min_rise_pct']: continue
        rstl_search_period = df.iloc[lsth_idx + 1 : scan_idx + 1]
        if rstl_search_period.empty: continue
        rstl_idx = rstl_search_period['low'].idxmin(); rstl_data = df.iloc[rstl_idx]
        if rstl_data['low'] <= lstl_data['low']: continue
        base_pattern_data = {'Ticker': ticker.upper(), 'LSTH_Date': lsth_data['date'], 'LSTH_High': lsth_data['high'], 'LSTL_Date': lstl_data['date'], 'LSTL_Low': lstl_data['low'], 'RSTL_Date': rstl_data['date'], 'RSTL_Low': rstl_data['low']}
        metrics = calculate_stair_metrics(df, lstl_idx, lsth_idx, rstl_idx, scan_idx)
        first_rsth, breakout_condition_type = None, None
        for rsth_iter_idx in range(rstl_idx + 1, min(scan_idx + 1, lsth_idx + CONFIG['stair_rsth_lookforward'] + 1)):
            if rsth_iter_idx < 2 or rsth_iter_idx >= len(df): continue
            rsth_data = df.iloc[rsth_iter_idx]
            is_green = rsth_data['close'] > rsth_data['open']
            is_quality = (rsth_data['volume'] > rsth_data['volume_ma50'] and rsth_data['close'] > df.iloc[rsth_iter_idx-1]['close'] and rsth_data['close'] > df.iloc[rsth_iter_idx-2]['close'] and rsth_data['close'] > (lsth_data['high'] + rstl_data['low']) / 2 and rsth_data['close'] > rsth_data['8dema'] and rsth_data['close'] > rsth_data['21dema'] and is_green)
            is_price_level = (rsth_data['close'] > lsth_data['high']) and is_green
            if is_quality: first_rsth, breakout_condition_type = {'idx': rsth_iter_idx, 'data': rsth_data}, 1; break
            elif is_price_level: first_rsth, breakout_condition_type = {'idx': rsth_iter_idx, 'data': rsth_data}, 2; break
        if first_rsth and first_rsth['idx'] >= (scan_idx - (CONFIG['stair_rsth_recency_days'] - 1)):
            return [{'Pattern_Type': 'Completed Stair', 'RSTH_Date': first_rsth['data']['date'], 'RSTH_High': first_rsth['data']['high'], 'RSTH_Open': first_rsth['data']['open'], 'RSTH_Close': first_rsth['data']['close'], 'days_since_completion': scan_idx - first_rsth['idx'], 'breakout_condition_type': breakout_condition_type, **base_pattern_data, **metrics}]
        if scan_data['close'] > lsth_data['high']: continue
        return [{'Pattern_Type': 'On Going Stair', **base_pattern_data, **metrics}]
    return []

def find_high_rvol(ticker, df, scan_date):
    scan_indices = df.index[df['date'] == scan_date].tolist()
    if not scan_indices: return []
    for lookback_days in range(3):
        current_idx = scan_indices[0] - lookback_days
        if current_idx < 2: continue
        day_data = df.iloc[current_idx]
        if (day_data['volume'] > day_data['volume_ma50'] and day_data['close'] > df.iloc[current_idx - 1]['close'] and day_data['close'] > df.iloc[current_idx - 2]['close'] and day_data['close'] > day_data['8dema'] and day_data['close'] > day_data['21dema'] and day_data['close'] > day_data['open']):
            return [{'Pattern_Type': 'High RVOL', 'Ticker': ticker.upper(), 'days_since_rvol': lookback_days, 'rvol_date': day_data['date'], 'rvol_high': day_data['high']}]
    return []

# --- SUPPORTING FUNCTIONS (Script 3) ---
def calculate_generic_uptrend(ticker_df, scan_date):
    if ticker_df.empty: return None
    relevant_df = ticker_df[ticker_df['date'] <= scan_date]
    if len(relevant_df) < 30: return None
    last_30d, last_10d = relevant_df.tail(30), relevant_df.tail(10)
    lowest_low, highest_high = last_30d['low'].min(), last_10d['high'].max()
    if lowest_low > 0: return ((highest_high / lowest_low) - 1) * 100
    return None

def get_dynamic_breakout_entry(p, scan_date, ticker_df):
    breakout_price = 0; df_copy = ticker_df.copy().set_index('date')
    if p.get('Pattern_Type') in ['On Going C/H', 'Completed C/H']:
        midpoint_handle = (p['RRC_High'] + p['BH_Low']) / 2
        if p.get('Pattern_Type') == 'On Going C/H':
            breakout_price = midpoint_handle if df_copy.loc[scan_date]['close'] <= midpoint_handle else p['RRC_High']
        elif p.get('Pattern_Type') == 'Completed C/H' and 'breakout_condition_type' in p:
            if p['breakout_condition_type'] == 1: breakout_price = midpoint_handle if df_copy.loc[p['HB_Date']]['open'] < midpoint_handle else p['RRC_High']
            else: breakout_price = p['RRC_High']
        else: breakout_price = p['RRC_High']
    elif p.get('Pattern_Type') == 'Completed Stair':
        midpoint = (p['LSTH_High'] + p['RSTL_Low']) / 2; rsth_open = p.get('RSTH_Open', 0); rsth_close = p.get('RSTH_Close', 0); lsth_high = p.get('LSTH_High', 0)
        if rsth_open < midpoint or (rsth_open > midpoint and rsth_close < lsth_high): breakout_price = midpoint
        else: breakout_price = lsth_high
    elif p.get('Pattern_Type') == 'On Going Stair':
        midpoint_stair = (p['LSTH_High'] + p['RSTL_Low']) / 2
        breakout_price = p['LSTH_High'] if df_copy.loc[scan_date]['close'] > midpoint_stair else midpoint_stair
    elif 'breakout_entry_point' in p: breakout_price = p.get('display_be_price', p['breakout_entry_point'])
    return breakout_price

def create_stock_chart(ticker, ticker_df, scan_date, fundamental_data, pattern_details=None, market_condition=None):
    try:
        df_copy = ticker_df.copy(); df_copy.set_index('date', inplace=True); df_copy['sma50'] = df_copy['close'].rolling(window=50).mean()
        chart_df = df_copy.loc[scan_date - pd.Timedelta(days=90):scan_date].copy()

        if chart_df.empty:
            print(f"  - Warning: No chart data found for {ticker} in the 90-day window ending {scan_date.strftime('%Y-%m-%d')}.")
            return None

        chart_df['volume_spike_signal'] = np.where(chart_df['volume'] > chart_df['volume_ma50'], chart_df['volume'] * 1.05, np.nan)
        last_day_volume_ma50 = chart_df.dropna(subset=['volume_ma50']).iloc[-1]['volume_ma50'] if not chart_df.dropna(subset=['volume_ma50']).empty else None
        future_dates = pd.date_range(start=chart_df.index[-1], periods=6, freq='B'); padding_df = pd.DataFrame(index=future_dates[1:]); chart_df = pd.concat([chart_df, padding_df])
        plots = [mpf.make_addplot(chart_df[col], **kwargs) for col, kwargs in [('8dema', {'color':'blue', 'width':0.7}), ('21dema', {'color':'green', 'width':0.7}), ('sma50', {'color':'red', 'width':0.7}), ('volume_ma50', {'color':'orange', 'width':0.9, 'panel':1}), ('volume_spike_signal', {'type':'scatter', 'marker':'v', 'color':'blue', 'panel':1})]]
        if pd.notna(last_day_volume_ma50):
            label = f"{last_day_volume_ma50/1e6:.1f}M" if last_day_volume_ma50 >= 1e6 else f"{last_day_volume_ma50/1e3:.0f}K"; chart_df.loc[future_dates[4], 'dsma_vol_label'] = last_day_volume_ma50
            plots.append(mpf.make_addplot(chart_df['dsma_vol_label'], panel=1, type='scatter', marker=rf"${label}$", color='orange', markersize=800))
        if pattern_details:
            p = pattern_details
            if p.get('Pattern_Type') in ['On Going C/H', 'Completed C/H']:
                points_to_plot = {
                    'lrc': {'date': p['LRC_Date'], 'price': p['LRC_High'], 'pos': 1.03, 'label': f"$\\${p['LRC_High']:.2f}$"},
                    'rrc': {'date': p['RRC_Date'], 'price': p['RRC_High'], 'pos': 1.03, 'label': f"$\\${p['RRC_High']:.2f}$"},
                    'bc': {'date': p['BC_Date'], 'price': p['BC_Low'], 'pos': 0.96, 'label': f"$\\${p['BC_Low']:.2f}$"}
                }
                for name, data in points_to_plot.items():
                    if data['date'] in chart_df.index:
                        chart_df.loc[data['date'], f'{name}_label'] = data['price'] * data['pos']
                        plots.append(mpf.make_addplot(chart_df[f'{name}_label'], type='scatter', marker=data['label'], color='black', markersize=600))
                if p.get('Pattern_Type') == 'Completed C/H' and p.get('HB_Date') and p['HB_Date'] in chart_df.index: chart_df.loc[p['HB_Date'], 'hb_arrow'] = chart_df.loc[p['HB_Date']]['high'] * 1.05; plots.append(mpf.make_addplot(chart_df['hb_arrow'], type='scatter', marker='v', color='black', markersize=70))
            elif p.get('Pattern_Type') in ['Completed Stair', 'On Going Stair']:
                points_to_plot = {'LSTH': {'date': p['LSTH_Date'], 'price': p['LSTH_High'], 'pos': 'above', 'color': 'black'}, 'LSTL': {'date': p['LSTL_Date'], 'price': p['LSTL_Low'], 'pos': 'below', 'color': 'black'}, 'RSTL': {'date': p['RSTL_Date'], 'price': p['RSTL_Low'], 'pos': 'below', 'color': 'black'}};
                if p.get('Pattern_Type') == 'Completed Stair': points_to_plot['RSTH'] = {'date': p['RSTH_Date'], 'price': p['RSTH_High'], 'pos': 'above', 'color': 'black'}
                for name, data in points_to_plot.items():
                    if data['date'] in chart_df.index: y, m = (data['price'] * 1.04, 'v') if data['pos'] == 'above' else (data['price'] * 0.96, '^'); chart_df.loc[data['date'], f'{name}_arrow'] = y; plots.append(mpf.make_addplot(chart_df[f'{name}_arrow'], type='scatter', marker=m, color=data['color'], markersize=70))
            elif p.get('Pattern_Type') == 'High RVOL' and p['rvol_date'] in chart_df.index: chart_df.loc[p['rvol_date'], 'hrvol_arrow'] = p['rvol_high'] * 1.04; plots.append(mpf.make_addplot(chart_df['hrvol_arrow'], type='scatter', marker='v', color='purple', markersize=100))
            if 'breakout_entry_point' in p:
                sl_price = p.get('BH_Low') if p.get('Pattern_Type') in ['On Going C/H', 'Completed C/H'] else p.get('RSTL_Low', 0)
                be_price = get_dynamic_breakout_entry(p, scan_date, ticker_df)
                sl_pct = ((sl_price / be_price) - 1) * 100 if be_price > 0 else 0; pct_low_pt = ((p['low_price_target'] / be_price) - 1) * 100 if be_price > 0 else 0; pct_high_pt = ((p['high_price_target'] / be_price) - 1) * 100 if be_price > 0 else 0; rrr = (pct_low_pt / abs(sl_pct)) if sl_pct != 0 else 0
                start_date = p.get('HB_Date') or p.get('RSTH_Date') or scan_date; sl_start_date = p.get('BH_Date') or p.get('RSTL_Date') or scan_date
                if start_date in chart_df.index: chart_df.loc[start_date:, 'breakout_line'] = be_price; plots.append(mpf.make_addplot(chart_df['breakout_line'], color='green', linestyle='--', width=1.0))
                if sl_start_date in chart_df.index: chart_df.loc[sl_start_date:, 'sl_line'] = sl_price; plots.append(mpf.make_addplot(chart_df['sl_line'], color='darkred', linestyle='--', width=1.0))
        fig, axlist = mpf.plot(chart_df, type='candle', style='yahoo', addplot=plots, figsize=(12, 7), volume=True, returnfig=True)
        axlist[0].set_title(f"{ticker.upper()} (Daily) {scan_date.strftime('%Y-%m-%d')}", fontsize=20, fontweight='bold')
        if market_condition: axlist[0].text(0.5, 0.90, market_condition['text'], transform=axlist[0].transAxes, fontsize=20, color=market_condition['color'], fontweight='bold', ha='center', va='top')
        if pattern_details:
            p = pattern_details
            if p.get('Pattern_Type') in ['Completed Stair', 'On Going Stair']:
                for name, data in points_to_plot.items():
                    if data['date'] in chart_df.index:
                        x_coord = chart_df.index.get_loc(data['date'])
                        if data['pos'] == 'above': text_y, va = data['price'] * 1.09, 'bottom'
                        else: text_y, va = data['price'] * 0.91, 'top'
                        if p.get('Pattern_Type') == 'Completed Stair' and name == 'RSTH':
                            text_label = "RSTH"
                        else:
                            text_label = f"{name}: {data['price']:.2f}"
                        axlist[0].text(x_coord, text_y, text_label, color=data['color'], fontsize=8, fontweight='bold', ha='center', va=va)
            elif p.get('Pattern_Type') == 'High RVOL':
                if p['rvol_date'] in chart_df.index:
                    x_coord = chart_df.index.get_loc(p['rvol_date'])
                    text_y = p['rvol_high'] * 1.09
                    axlist[0].text(x_coord, text_y, "HRVOL", color='purple', fontsize=10, fontweight='bold', ha='center', va='bottom')
            elif p.get('Pattern_Type') == 'Completed C/H' and p.get('HB_Date') and p['HB_Date'] in chart_df.index:
                x_coord = chart_df.index.get_loc(p['HB_Date'])
                hb_high = chart_df.loc[p['HB_Date']]['high']
                text_y = hb_high * 1.09
                axlist[0].text(x_coord, text_y, "HB", color='black', fontsize=8, fontweight='bold', ha='center', va='bottom')
            if 'breakout_entry_point' in p:
                label_placement_date = future_dates[4]
                x_coord = chart_df.index.get_loc(label_placement_date)
                axlist[0].text(x_coord, be_price * 1.01, f"BE: ${be_price:.2f}", fontsize=9, color='green', fontweight='bold', ha='center', va='bottom')
                axlist[0].text(x_coord, sl_price * 0.95, f"SL: ${sl_price:.2f} ({sl_pct:.0f}%)", fontsize=9, color='darkred', fontweight='bold', ha='center', va='bottom')
        if fundamental_data:
            y_pos, x_pos, ls = 0.98, 0.02, 0.045
            fd = fundamental_data; sd = df_copy.loc[scan_date] if scan_date in df_copy.index else None
            def fmt_vol(v): return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"
            
            # --- START OF FIX: Use Perf UpTrend from fundamental_data directly ---
            up_val = fd.get('Perf UpTrend', 0)
            # --- END OF FIX ---
            
            lines = [f"Score: {fd.get('Score')} (Sector: {fd.get('Sector')}, Industry: {fd.get('Industry')})", f"Number of Institution days : {fd.get('Days_Institution')} days"]
            if sd is not None and pd.notna(sd.get('volume')) and pd.notna(sd.get('volume_ma50')): lines.append(f"Volume: {fmt_vol(sd['volume'])}, AVG Vol: {fmt_vol(sd['volume_ma50'])}, RVOL: {sd['volume']/sd['volume_ma50'] if sd['volume_ma50']>0 else 0:.1f}")
            if sd is not None:
                prev_close = df_copy.iloc[df_copy.index.get_loc(scan_date)-1]['close'] if df_copy.index.get_loc(scan_date)>0 else 0
                chg_str = f"{((sd['close']/prev_close)-1)*100:+.2f}%" if prev_close>0 else "N/A"
                lines.append(f"Price: ${sd['close']:.2f}, Change: {chg_str}")
            
            lines.extend([f"Prior UpTrend: +{up_val:.0f}%", 
                          f"Perf Quart: {fd.get('Perf Quart', 0):.2f}%, Perf Month: {fd.get('Perf Month', 0):.2f}%", 
                          f"Sales Q/Q: {fd.get('Sales Q/Q', 0):.2f}%, Sales (M): ${fd.get('Sales (M)')}"])

            for line in lines: axlist[0].text(x_pos, y_pos, line, transform=axlist[0].transAxes, fontsize=9, ha='left', va='top'); y_pos -= ls
            if pattern_details and 'low_price_target' in p:
                axlist[0].text(x_pos, y_pos-ls*0.5, f"Low PT: ${p['low_price_target']:.2f} (+{pct_low_pt:.0f}%), RRR: {rrr:.2f}", transform=axlist[0].transAxes, fontsize=9, ha='left', va='top')
                axlist[0].text(x_pos, y_pos-ls*1.5, f"High PT: ${p['high_price_target']:.2f} (+{pct_high_pt:.0f}%)", transform=axlist[0].transAxes, fontsize=9, ha='left', va='top')
                if p.get('Pattern_Type') in ['On Going C/H', 'Completed C/H']:
                    axlist[0].text(x_pos, y_pos-ls*2.5, f"Duration C/H: {p['cup_duration']}/{p['handle_duration']}d", transform=axlist[0].transAxes, fontsize=9, ha='left', va='top')
                    axlist[0].text(x_pos, y_pos-ls*3.5, f"Cup Depth | Handle Depth: {p['cup_depth_pct']:.0f}% | {p['handle_depth_pct']:.0f}%", transform=axlist[0].transAxes, fontsize=9, ha='left', va='top')
        buf = BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig); return buf
    except Exception as e:
        print(f"  - Error generating chart for {ticker}: {e}")
        plt.close('all')
        return None
        
def generate_pdf_report(categories, patterns_dict, full_df, scan_date):
    if not any(not cat['df'].empty for cat in categories):
        print("\nNo stocks to report in any category. Skipping PDF generation.")
        return
    print("\n" + "="*80); print("--- [3C] GENERATING FINAL PDF REPORT ---"); print("="*80)
    pdf_filename = f"stock_analysis_report_{scan_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}_{random.randint(100000, 999999)}.pdf"
    pdf = FPDF()
    try:
        pdf.add_font("DejaVu", "", fm.findfont(fm.FontProperties(family="DejaVu Sans")))
        pdf.add_font("DejaVu", "B", fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")))
    except Exception as e:
        print(f"FATAL: Could not find DejaVu Sans font. Please ensure Matplotlib is installed correctly. Error: {e}"); sys.exit(1)

    for i, ticker in enumerate(['qqq', 'spy', 'sqqq', 'uvix']):
        if i % 2 == 0: pdf.add_page()
        print(f"  Generating market context chart for {ticker.upper()}...")
        if ticker in full_df['ticker'].values:
            ticker_df = full_df[full_df['ticker'] == ticker].copy()
            ticker_df['sma50'] = ticker_df['close'].rolling(window=50).mean()
            last = ticker_df[ticker_df['date'] <= scan_date].iloc[-1]
            uptrend = (last['8dema'] >= last['21dema'] and last['close'] > last['21dema']) or (last['8dema'] > last['sma50'] and last['21dema'] > last['sma50'] and last['close'] > last['sma50'])
            chart_buffer = create_stock_chart(ticker, ticker_df, scan_date, market_condition={'text': 'Market Uptrend', 'color': 'green'} if uptrend else {'text': 'Market Downtrend', 'color': 'red'}, fundamental_data={})
            if chart_buffer: pdf.image(chart_buffer, x=(pdf.w - 190) / 2, y=(10 if i % 2 == 0 else 145), w=190); chart_buffer.close()

    headers = ['Date', 'Ticker', 'Sector', 'Industry', 'Score', 'RVOL', 'AVG Vol', 'Inst Days', 'Sales Q/Q', 'Sales (M)', 'Perf Q', 'Perf M', 'Prior Up', 'Price', 'Low PT']
    col_widths = {'Date': 14, 'Ticker': 58, 'Sector': 27, 'Industry': 27, 'Score': 9, 'RVOL': 9, 'AVG Vol': 12, 'Inst Days': 14, 'Sales Q/Q': 15, 'Sales (M)': 15, 'Perf Q': 12, 'Perf M': 12, 'Prior Up': 13, 'Price': 12, 'Low PT': 22}
    def fmt_disp_str(t, r, total, short, p):
        base = f"{t} - {r}/{total} - {short}"; suffix = ""
        if p:
            pt = p.get('Pattern_Type'); days = p.get('days_since_completion', p.get('days_since_hb', p.get('days_since_rvol', 0)))
            if 'Completed' in pt or 'High RVOL' in pt: suffix = f" ({'D' if days==0 else f'D-{days}'})"
            base += f" - {pt}{suffix}"
        return base
    for category in categories:
        if category['df'].empty: continue
        print(f"  Processing category for PDF: {category['title']}...")
        pdf.add_page(orientation='L'); pdf.set_font('DejaVu', 'B', 12); pdf.cell(0, 10, category['title'], new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C'); pdf.ln(5)
        pdf.set_font('DejaVu', 'B', 7)
        for h in headers: pdf.cell(col_widths[h], 7, h, border=1, align='C')
        pdf.ln(); pdf.set_font('DejaVu', '', 6)
        for index, row in category['df'].iterrows():
            if pdf.get_y() + 6 > pdf.h - pdf.b_margin:
                pdf.add_page(orientation='L'); pdf.set_font('DejaVu', 'B', 7); [pdf.cell(col_widths[h], 7, h, border=1, align='C') for h in headers]; pdf.ln(); pdf.set_font('DejaVu', '', 6)
            ticker = row['Ticker']; pattern = patterns_dict.get(ticker)
            display_ticker = fmt_disp_str(ticker, index + 1, len(category['df']), category['short_name'], pattern)
            
            # --- START OF FIX: Use Perf UpTrend from the row data directly ---
            perf_uptrend_val = row.get('Perf UpTrend')
            prior_up_str = f"+{perf_uptrend_val:.0f}%" if pd.notna(perf_uptrend_val) else 'N/A'
            # --- END OF FIX ---

            avg_vol_val = row.get('AVG_Vol')
            avg_vol_str = f"{avg_vol_val / 1_000_000:.1f}M" if pd.notna(avg_vol_val) and avg_vol_val >= 1_000_000 else 'N/A'
            rvol_val = row.get('RVOL')
            rvol_str = f"{rvol_val:.1f}" if pd.notna(rvol_val) else 'N/A'
            sales_qq_val = row.get('Sales Q/Q')
            sales_qq_str = f"{sales_qq_val:.2f}%" if pd.notna(sales_qq_val) else "N/A"
            perf_q_val = row.get('Perf Quart')
            perf_q_str = f"{perf_q_val:.2f}%" if pd.notna(perf_q_val) else "N/A"
            perf_m_val = row.get('Perf Month')
            perf_m_str = f"{perf_m_val:.2f}%" if pd.notna(perf_m_val) else "N/A"
            sector_str = str(row.get('Sector', ''))[:20]
            industry_str = str(row.get('Industry', ''))[:20]
            low_pt_str = 'N/A'
            if pattern and 'low_price_target' in pattern:
                be = get_dynamic_breakout_entry(pattern, scan_date, full_df[full_df['ticker'] == ticker.lower()])
                pct = ((pattern['low_price_target'] / be) - 1) * 100 if be > 0 else 0
                low_pt_str = f"${pattern['low_price_target']:.2f} (+{pct:.0f}%)"
            data = [scan_date.strftime('%Y-%m-%d'), display_ticker, sector_str, industry_str, str(row.get('Score','')), rvol_str, avg_vol_str, f"{row.get('Days_Institution', '')} days", sales_qq_str, f"${row.get('Sales (M)','')}", perf_q_str, perf_m_str, prior_up_str, f"${row.get('Price','')}", low_pt_str]
            for j, d in enumerate(data): pdf.cell(col_widths[headers[j]], 6, str(d), border=1)
            pdf.ln()
        print(f"  Generating charts for {category['title']}...")
        for index, row in category['df'].iterrows():
            ticker = row['Ticker']
            print(f"    Generating chart {index + 1}/{len(category['df'])} for {ticker}...")
            if index % 2 == 0: pdf.add_page(orientation='P'); y = 10
            else: y = 145
            pattern = patterns_dict.get(ticker)
            try:
                title = fmt_disp_str(ticker, index+1, len(category['df']), category['short_name'], pattern)
                pdf.set_font("DejaVu", 'B', 12); pdf.set_y(y); pdf.cell(0, 10, title, align='C')
                chart_buffer = create_stock_chart(ticker, full_df[full_df['ticker'] == ticker.lower()], scan_date, fundamental_data=row.to_dict(), pattern_details=pattern)
                if not chart_buffer: raise ValueError("Chart buffer generation failed.")
                pdf.image(chart_buffer, x=(pdf.w - 190) / 2, y=y + 8, w=190); chart_buffer.close()
            except Exception as e: print(f"  - Error adding chart for {ticker} to PDF: {e}"); pdf.set_font("DejaVu", size=10); pdf.set_y(y + 60); pdf.cell(0, 10, f"Chart failed for {ticker}", align='C')
    pdf.output(pdf_filename); print(f"\nSUCCESS! PDF report generated: {os.path.abspath(pdf_filename)}")
    
# --- Orchestration function (replaces `main` from Script 3) ---
def run_pattern_analysis_and_generate_report(historical_df, score_df, scan_date, debug=False):
    print("\n" + "="*80); print("--- [3A] PREPARING DATA FOR PATTERN ANALYSIS ---"); print("="*80)
    df = historical_df.copy()
    print(f"Using consistent scan date for pattern analysis: {scan_date.strftime('%Y-%m-%d')}")

    scan_day_data = df[df['date'] == scan_date].copy()
    if scan_day_data.empty:
        print(f"\nFATAL ERROR: No historical OHLCV data was found for the scan date: {scan_date.strftime('%Y-%m-%d')}.")
        print("Pipeline cannot continue. Please specify a valid trading day with the --endDate argument.")
        return

    high_score_stocks = score_df[score_df['Score'] >= CONFIG['min_score_threshold']].copy()
    if high_score_stocks.empty:
        print(f"No stocks found with a score of {CONFIG['min_score_threshold']} or higher. Exiting."); return
    print(f"Found {len(high_score_stocks)} stocks with a score of {CONFIG['min_score_threshold']} or higher to analyze.")

    print("Calculating generic uptrend and volume metrics for high-score stocks...")
    high_score_stocks['Generic_Prior_Up'] = [calculate_generic_uptrend(df[df['ticker'] == t.lower()], scan_date) for t in high_score_stocks['Ticker']]

    scan_day_data['ticker_upper'] = scan_day_data['ticker'].str.upper()
    high_score_stocks = pd.merge(high_score_stocks, scan_day_data[['ticker_upper', 'open', 'close', 'volume', 'volume_ma50']], left_on='Ticker', right_on='ticker_upper', how='left')
    high_score_stocks['RVOL'] = high_score_stocks['volume'] / high_score_stocks['volume_ma50']
    high_score_stocks.rename(columns={'volume_ma50': 'AVG_Vol'}, inplace=True)

    rvol_candidates = high_score_stocks.dropna(subset=['open','close'])
    if not rvol_candidates.empty:
         rvol_candidates = rvol_candidates[rvol_candidates['close'] > rvol_candidates['open']].copy()
    else:
         rvol_candidates = pd.DataFrame()


    print("\n" + "="*80); print("--- [3B] SCANNING FOR PATTERNS (C/H, STAIR, RVOL) ---"); print("="*80)
    if debug:
        print("--- DEBUG MODE ENABLED: Detailed pattern scanning logs will be shown. ---")

    all_patterns = []
    for ticker in tqdm(high_score_stocks['Ticker'], desc="Scanning Patterns"):
        ticker_df = df[df['ticker'] == ticker.lower()].sort_values(by='date').reset_index(drop=True)
        if ticker_df.empty:
            if debug: print(f"\nDEBUG [{ticker}]: No historical data found. Skipping.")
            continue
        
        # --- START OF DEBUG LOGGING ---
        if debug:
            scan_day_row = ticker_df[ticker_df['date'] == scan_date]
            if scan_day_row.empty:
                print(f"\nDEBUG [{ticker}]: No data for scan date {scan_date.strftime('%Y-%m-%d')}. Skipping.")
                continue
            
            scan_data = scan_day_row.iloc[0]
            close_price = scan_data.get('close')
            dema21 = scan_data.get('21dema')
            print(f"\nDEBUG [{ticker}]: Analyzing on {scan_date.strftime('%Y-%m-%d')}")
            if pd.notna(close_price) and pd.notna(dema21):
                print(f"  - Stair Pre-check: Close ({close_price:.2f}) > 21DEMA ({dema21:.2f})? {'PASS' if close_price > dema21 else 'FAIL'}")
            else:
                print(f"  - Stair Pre-check: Missing Close or 21DEMA data.")
        # --- END OF DEBUG LOGGING ---

        patterns = find_in_handle_patterns(ticker, ticker_df, scan_date) or \
                   find_stair_patterns(ticker, ticker_df, scan_date) or \
                   find_high_rvol(ticker, ticker_df, scan_date)
        if patterns:
            if debug: print(f"  >>> SUCCESS: Found pattern: {patterns[0].get('Pattern_Type')}")
            all_patterns.extend(patterns)
        elif debug:
            print(f"  - INFO: No C/H, Stair, or RVOL pattern found.")

    patterns_dict = {p['Ticker']: p for p in all_patterns}
    print(f"\nScan complete. Found patterns/signals for {len(patterns_dict)} high-score stocks.")
    # The rest of the function remains the same...
    print("Sorting results and preparing categories for the report...")
    cah_tickers = [t for t, p in patterns_dict.items() if p.get('Pattern_Type') in ['Completed C/H', 'On Going C/H']]
    cah_df = high_score_stocks[high_score_stocks['Ticker'].isin(cah_tickers)].copy()
    if not cah_df.empty:
        cah_df['Pattern_Type'] = cah_df['Ticker'].map(lambda t: patterns_dict[t].get('Pattern_Type'))
        top_25_cah_df = pd.concat([cah_df[cah_df['Pattern_Type'] == 'Completed C/H'].sort_values(by='RVOL', ascending=False), cah_df[cah_df['Pattern_Type'] == 'On Going C/H'].sort_values(by='RVOL', ascending=False)]).head(25).reset_index(drop=True)
    else: top_25_cah_df = pd.DataFrame()
    stair_tickers = [t for t, p in patterns_dict.items() if p.get('Pattern_Type') in ['Completed Stair', 'On Going Stair']]
    stair_df = high_score_stocks[high_score_stocks['Ticker'].isin(stair_tickers)].copy()
    if not stair_df.empty:
        stair_df['Pattern_Type'] = stair_df['Ticker'].map(lambda t: patterns_dict[t].get('Pattern_Type'))
        top_25_stair_df = pd.concat([stair_df[stair_df['Pattern_Type'] == 'Completed Stair'].sort_values(by='RVOL', ascending=False), stair_df[stair_df['Pattern_Type'] == 'On Going Stair'].sort_values(by='RVOL', ascending=False)]).head(25).reset_index(drop=True)
    else: top_25_stair_df = pd.DataFrame()
    pt_tickers = [t for t, p in patterns_dict.items() if 'low_price_target' in p]
    pt_df = high_score_stocks[high_score_stocks['Ticker'].isin(pt_tickers)].copy()
    if not pt_df.empty:
        pt_df['Low_PT_Pct'] = pt_df.apply(lambda row: ((patterns_dict[row['Ticker']]['low_price_target'] / get_dynamic_breakout_entry(patterns_dict[row['Ticker']], scan_date, df[df['ticker'] == row['Ticker'].lower()])) - 1) * 100 if get_dynamic_breakout_entry(patterns_dict[row['Ticker']], scan_date, df[df['ticker'] == row['Ticker'].lower()]) > 0 else 0, axis=1)
        top_25_low_pt_df = pt_df.sort_values(by='Low_PT_Pct', ascending=False).head(25).reset_index(drop=True)
    else: top_25_low_pt_df = pd.DataFrame()

    category_definitions = [
        {'title': 'Top 25 High RVOL Stocks Summary (by RVOL)', 'sort_by': 'RVOL', 'ascending': False, 'short_name': 'Top 25 RVOL', 'source_df': rvol_candidates},
        {'title': 'Top 25 Low Profit Target Stocks Summary (by Low PT %)', 'short_name': 'Top 25 Low PT', 'df': top_25_low_pt_df},
        {'title': 'Top 25 High Score Stocks Summary (by Score)', 'sort_by': 'Score', 'ascending': False, 'short_name': 'Top 25 Score', 'source_df': high_score_stocks},
        {'title': 'Top 25 High Perf Quarter Stocks Summary (by Perf Q)', 'sort_by': 'Perf Quart', 'ascending': False, 'short_name': 'Top 25 Perf Q', 'source_df': high_score_stocks},
        {'title': 'Top 25 High AVG Volume Stocks Summary (by AVG Vol)', 'sort_by': 'AVG_Vol', 'ascending': False, 'short_name': 'Top 25 AVG Vol', 'source_df': high_score_stocks},
        {'title': 'Top 25 High Sales Q/Q Stocks Summary (by Sales Q/Q)', 'sort_by': 'Sales Q/Q', 'ascending': False, 'short_name': 'Top 25 Sales Q/Q', 'source_df': high_score_stocks},
        {'title': 'Top 25 High Perf Month Stocks Summary (by Perf M)', 'sort_by': 'Perf Month', 'ascending': False, 'short_name': 'Top 25 Perf M', 'source_df': high_score_stocks},
        {'title': 'Top 25 C/H Pattern Stocks Summary (by RVOL)', 'short_name': 'Top 25 C/H', 'df': top_25_cah_df},
        {'title': 'Top 25 Stair Pattern Stocks Summary (by RVOL)', 'short_name': 'Top 25 Stair', 'df': top_25_stair_df},
    ]
    categories = []
    for cat_def in category_definitions:
        if 'df' in cat_def:
            if not cat_def['df'].empty: categories.append({k: v for k, v in cat_def.items()})
        elif 'sort_by' in cat_def and cat_def['sort_by'] in cat_def['source_df'].columns:
            sorted_df = cat_def['source_df'].dropna(subset=[cat_def['sort_by']]).sort_values(by=cat_def['sort_by'], ascending=cat_def['ascending']).head(25).reset_index(drop=True)
            if not sorted_df.empty: categories.append({**{k: v for k, v in cat_def.items() if k != 'source_df'}, 'df': sorted_df})
    generate_pdf_report(categories, patterns_dict, df, scan_date)

# ==============================================================================
# --- MAIN ORCHESTRATION ---
# ==============================================================================
def main(args_list=None): # Add args_list=None parameter
    """
    Main function to orchestrate the entire pipeline.
    """
    parser = argparse.ArgumentParser(description="A comprehensive stock analysis pipeline for scanning, fetching data, and identifying patterns.")
    parser.add_argument('--endDate', type=str, required=False, help="The end date for historical analysis in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument('--ticker', type=str, nargs='+', required=False, help="One or more specific ticker symbols to analyze, bypassing the Finviz scan.")
    # --- START OF NEW ARGUMENT ---
    parser.add_argument('--debug', action='store_true', help="Enable detailed debug logging during pattern scanning.")
    # --- END OF NEW ARGUMENT ---
    args = parser.parse_args(args_list) 

    end_date_intent = None
    if args.endDate:
        try:
            end_date_intent = datetime.strptime(args.endDate, '%Y-%m-%d')
        except ValueError:
            print("Error: Incorrect date format for --endDate. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
        end_date_intent = datetime.now()

    start_time = time.time()

    score_df = run_scanner_and_scorer(args)
    if score_df is None or score_df.empty:
        print("\nPipeline stopped: No stocks were found matching the initial screening criteria.")
        sys.exit(0)

    historical_df = fetch_historical_data_and_indicators(score_df, args.endDate)
    if historical_df is None or historical_df.empty:
        print("\nPipeline stopped: Failed to retrieve historical OHLCV data for the selected stocks.")
        sys.exit(0)

    actual_scan_date = historical_df['date'].max()
    actual_scan_date = actual_scan_date.normalize()

    if actual_scan_date.date() != end_date_intent.date():
        print("-" * 80)
        print(f"INFO: Requested scan date was {end_date_intent.strftime('%Y-%m-%d')}, which is not a trading day.")
        print(f"Adjusting analysis to use the latest available trading day: {actual_scan_date.strftime('%Y-%m-%d')}")
        print("-" * 80)

    # --- Pass the new debug flag to the function ---
    run_pattern_analysis_and_generate_report(historical_df, score_df, actual_scan_date, args.debug)

    elapsed_time = time.time() - start_time
    print("-" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS.")
    print("-" * 80)

if __name__ == "__main__":
    main()