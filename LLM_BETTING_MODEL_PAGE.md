# LLM Betting Model Visualization Page

## Overview
This Streamlit page visualizes betting results from your LLM betting model repository. It provides comprehensive analytics and visualizations to track model performance over time.

## Features

### 1. **Overall Performance Summary**
- Total bets placed
- Overall win rate
- Total payout (in units)
- Return on Investment (ROI)

### 2. **Model Performance Over Time**
- Interactive line chart showing cumulative payout for each model
- Helps identify which model is performing best over time
- Includes a reference line at 0 to easily see profit/loss

### 3. **Daily Performance**
- Grouped bar chart showing daily performance by model
- Compare how different models perform on the same days

### 4. **Model Comparison**
- Win rate visualization by model
- Detailed statistics table showing:
  - Total wins
  - Total bets
  - Total payout
  - Win rate percentage
  - ROI percentage

### 5. **Date-Specific Results**
- Select any date to view detailed bet information
- Color-coded table (green for wins, red for losses)
- Summary metrics for the selected date
- Breakdown by model for that specific date

## Data Sources

The page loads data from two CSV files in your llm_betting_model repository:

- **NBA**: `https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bet_picks_evaluated.csv`
- **NCAAB**: `https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/ncaab_bet_picks_evaluated.csv`

## Data Structure

The CSV files should contain the following columns:
- `rank`: Ranking of the bet
- `model`: Model name (e.g., 'v2', 'v2_perp', 'claude', 'gemini', 'perp')
- `date`: Date of the game
- `game_id`: Unique game identifier
- `match`: Teams playing (e.g., "Grizzlies vs Pelicans")
- `home_score`: Final home team score
- `away_score`: Final away team score
- `pick`: The bet pick (e.g., "Grizzlies -3.5", "Over 214.5")
- `odds`: Betting odds (e.g., -115, +150)
- `units`: Units wagered
- `bet_result`: 'win' or 'loss'
- `bet_payout`: Payout amount (positive for wins, negative for losses)

## Usage

1. Navigate to the "🎲 LLM Betting Model" page in the Streamlit sidebar
2. Select the sport (NBA or NCAAB) from the dropdown
3. View the overall performance metrics and charts
4. Use the date selector to view detailed results for specific dates
5. Compare model performance using the various visualizations

## Technical Details

- Data is cached for 1 hour to improve performance
- All visualizations use Plotly for interactivity
- The page uses the Futura font family to match your existing pages
- Color scheme matches your existing Streamlit app design

## Models Tracked

### NBA
- v2
- v2_perp

### NCAAB
- perp
- claude
- gemini

## Test Results

The page has been tested and verified to:
- ✓ Successfully load NBA data (397 bets)
- ✓ Successfully load NCAAB data (151 bets)
- ✓ Calculate cumulative payouts correctly
- ✓ Calculate win rates accurately (53.40% overall for NBA)
- ✓ Aggregate model statistics properly
