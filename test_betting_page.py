#!/usr/bin/env python3
"""Test script to verify the LLM Betting Model page loads correctly"""

import pandas as pd
import sys

def test_load_data():
    """Test loading the betting data"""
    print("Testing NBA data load...")
    nba_url = 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bet_picks_evaluated.csv'
    
    try:
        df = pd.read_csv(nba_url)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Successfully loaded {len(df)} NBA bets")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Models: {df['model'].unique().tolist()}")
        print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"✗ Failed to load NBA data: {e}")
        return False
    
    print("\nTesting NCAAB data load...")
    ncaab_url = 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/ncaab_bet_picks_evaluated.csv'
    
    try:
        df = pd.read_csv(ncaab_url)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Successfully loaded {len(df)} NCAAB bets")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Models: {df['model'].unique().tolist()}")
    except Exception as e:
        print(f"✗ Failed to load NCAAB data: {e}")
        return False
    
    return True

def test_upcoming_bets():
    """Test loading upcoming bets"""
    print("\nTesting upcoming NBA bets load...")
    
    nba_files = {
        'v2': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bets_v2.txt',
        'v2_perp': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bets_v2_perp.txt',
        'gemini': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bets_gemini.txt'
    }
    
    all_bets = []
    for model, url in nba_files.items():
        try:
            df = pd.read_csv(url)
            df['model'] = model
            all_bets.append(df)
            print(f"✓ Loaded {len(df)} bets from {model}")
        except Exception as e:
            print(f"✗ Failed to load {model}: {e}")
            return False
    
    combined = pd.concat(all_bets, ignore_index=True)
    print(f"✓ Combined {len(combined)} total NBA upcoming bets")
    print(f"  Columns: {combined.columns.tolist()[:10]}...")
    
    print("\nTesting upcoming NCAAB bets load...")
    
    ncaab_files = {
        'claude': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/cbb_bets_claude.txt',
        'perp': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/cbb_bets_perp.txt',
        'gemini': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/cbb_bets_gemini.txt'
    }
    
    all_bets = []
    for model, url in ncaab_files.items():
        try:
            df = pd.read_csv(url)
            df['model'] = model
            all_bets.append(df)
            print(f"✓ Loaded {len(df)} bets from {model}")
        except Exception as e:
            print(f"✗ Failed to load {model}: {e}")
            return False
    
    combined = pd.concat(all_bets, ignore_index=True)
    print(f"✓ Combined {len(combined)} total NCAAB upcoming bets")
    
    return True

def test_calculations():
    """Test the calculation logic"""
    print("\nTesting calculations...")
    nba_url = 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bet_picks_evaluated.csv'
    
    try:
        df = pd.read_csv(nba_url)
        df['date'] = pd.to_datetime(df['date'])
        
        # Test cumulative payout calculation
        df_sorted = df.sort_values(['model', 'date', 'rank']).copy()
        df_sorted['cumulative_payout'] = df_sorted.groupby('model')['bet_payout'].cumsum()
        
        print(f"✓ Cumulative payout calculation successful")
        
        # Test win rate calculation
        df['win'] = (df['bet_result'] == 'win').astype(int)
        win_rate = df['win'].sum() / len(df) * 100
        print(f"✓ Overall win rate: {win_rate:.2f}%")
        
        # Test model stats
        model_stats = df.groupby('model').agg({
            'win': ['sum', 'count'],
            'bet_payout': 'sum'
        }).reset_index()
        
        print(f"✓ Model aggregation successful")
        
    except Exception as e:
        print(f"✗ Calculation failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("LLM Betting Model Page Test")
    print("=" * 60)
    
    success = True
    success = test_load_data() and success
    success = test_upcoming_bets() and success
    success = test_calculations() and success
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)

