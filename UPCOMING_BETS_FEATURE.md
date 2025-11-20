# LLM Betting Model - Upcoming Bets Feature

## Overview
Added a new "Upcoming Bets" tab that displays pending and in-progress bets from your LLM betting models, with full integration to show completed results.

## New Features

### 🔮 Upcoming Bets Tab (First Tab)

This new tab provides a comprehensive view of all bets across all models, including:

#### **Status Tracking**
- **⏳ Upcoming**: Bets that haven't started yet
- **🎮 In Progress**: Games currently being played
- **✅/❌ Completed**: Finished games with win/loss indicators

#### **Smart Filtering**
1. **Status Filter**: Show only upcoming, in-progress, or completed bets
2. **Model Filter**: Filter by specific models (v2, v2_perp, gemini, claude, perp)
3. **Confidence Filter**: Slider to show only bets above a certain confidence threshold

#### **Summary Metrics**
- Count of upcoming bets
- Count of in-progress bets
- Count of completed bets
- Average confidence percentage

#### **Expandable Bet Cards**
Each bet is displayed as an expandable card showing:

**Card Title:**
- Status emoji (⏳/🎮/✅/❌)
- Model name
- Matchup
- Pick and odds
- Confidence percentage
- Result (if completed)

**Expanded Details:**
- **Left Column:**
  - Game matchup
  - Start time (formatted)
  - Pick details
  - Odds
  - Units wagered
  - Confidence percentage
  - Final score (if completed)
  - Result and payout (if completed)

- **Right Column:**
  - Current status
  - Model name
  - Bet rank
  - Predicted score (if available)

- **Betting Rationale:**
  - Full reasoning displayed in an easy-to-read info box
  - Handles long text gracefully

#### **Data Integration**
- Automatically matches upcoming bets with evaluated results
- Shows which bets have been completed
- Displays actual outcomes and payouts for completed bets
- Merges data from multiple model files

## Data Sources

### NBA
- `nba_bets_v2.txt` - v2 model bets
- `nba_bets_v2_perp.txt` - v2_perp model bets
- `nba_bets_gemini.txt` - gemini model bets

### NCAAB (CBB)
- `cbb_bets_claude.txt` - claude model bets
- `cbb_bets_perp.txt` - perp model bets
- `cbb_bets_gemini.txt` - gemini model bets

## Technical Implementation

### New Functions
1. **`load_upcoming_bets(sport)`**
   - Loads all txt files for the selected sport
   - Concatenates data from multiple models
   - Adds model identifier to each bet
   - Caches for 10 minutes (more frequent updates than evaluated bets)

### Data Matching
- Creates unique bet keys using `game_id + pick`
- Matches upcoming bets with evaluated results
- Determines status based on:
  - Has result → "Completed"
  - Start time passed but no result → "In Progress"
  - Start time in future → "Upcoming"

### User Experience Improvements
- **Expandable cards** prevent information overload
- **Color-coded status** for quick visual scanning
- **Filtering options** to focus on relevant bets
- **Readable reasoning** in dedicated info boxes
- **Real-time status** based on game start times

## Usage

1. Navigate to the "🔮 Upcoming Bets" tab (first tab)
2. Use filters to narrow down bets:
   - Select status (Upcoming/In Progress/Completed)
   - Choose specific models
   - Set minimum confidence threshold
3. Click on any bet card to expand and see full details
4. Read the betting rationale to understand the model's reasoning
5. For completed bets, see the actual result and payout

## Benefits

✅ **Proactive Betting**: See upcoming bets before games start
✅ **Live Tracking**: Monitor in-progress games
✅ **Complete History**: View all bets in one place
✅ **Model Transparency**: Read detailed reasoning for each pick
✅ **Confidence-Based**: Filter by confidence to focus on high-conviction bets
✅ **Multi-Model View**: Compare picks across different models
✅ **Integrated Results**: Seamlessly see outcomes for completed bets
