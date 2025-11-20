# Prompts Tab Feature

## Overview
Added a new "Prompts" tab that allows users to view and easily copy the betting prompts used by each model.

## Features

### 📝 Prompts Tab (6th Tab)

This tab provides easy access to all betting model prompts with the following features:

#### **Model Selection**
- Dropdown to select which model's prompt to view
- Models available based on selected sport:
  - **NBA**: gemini, v2, v2_perp
  - **NCAAB**: claude, gemini, perp

#### **Prompt Information**
- **Filename**: Shows the exact prompt file name (e.g., `nba_prompt_gemini.txt`)
- **Character count**: Total characters in the prompt
- **Line count**: Number of lines in the prompt

#### **Easy Copy Functionality**
Two ways to copy the prompt:

1. **Text Area Method** (Primary)
   - Large scrollable text area displaying the full prompt
   - Click inside and use keyboard shortcuts:
     - `Ctrl+A` / `Cmd+A` to select all
     - `Ctrl+C` / `Cmd+C` to copy

2. **Copy Button Method** (Alternative)
   - Click the "📋 Copy to Clipboard" button
   - Displays the prompt in a code block with Streamlit's built-in copy button
   - Provides visual confirmation with success message

#### **Additional Information**
- **Tips section**: Clear instructions on how to copy the prompt
- **Available prompts expander**: Shows all prompts available for the current sport

## Data Sources

Prompts are loaded from GitHub:
```
https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/prompts/{sport}_prompt_{model}.txt
```

### Available Prompts:
- `nba_prompt_gemini.txt`
- `nba_prompt_v2.txt`
- `nba_prompt_v2_perp.txt`
- `ncaab_prompt_claude.txt`
- `ncaab_prompt_gemini.txt`
- `ncaab_prompt_perp.txt`

## Technical Implementation

### New Functions
1. **`load_prompt(sport, model)`**
   - Loads prompt text from GitHub
   - Uses requests library to fetch the file
   - Caches for 1 hour
   - Returns the full prompt text or error message

2. **`get_available_models(sport)`**
   - Returns list of available models for a sport
   - NBA: ['gemini', 'v2', 'v2_perp']
   - NCAAB: ['claude', 'gemini', 'perp']

### User Experience
- **Responsive layout**: Text area adjusts to show content
- **Clear instructions**: Tips on how to copy
- **Multiple copy methods**: Accommodates different user preferences
- **Sport-aware**: Only shows models available for selected sport
- **Cached loading**: Fast prompt retrieval with 1-hour cache

## Usage

1. Navigate to the **📝 Prompts** tab (6th tab)
2. Select a model from the dropdown
3. View the prompt in the text area
4. Copy using either method:
   - **Method 1**: Click in text area → `Ctrl+A`/`Cmd+A` → `Ctrl+C`/`Cmd+C`
   - **Method 2**: Click "📋 Copy to Clipboard" button → Use copy button in code block

## Benefits

✅ **Easy Access**: All prompts in one place
✅ **Quick Copy**: Multiple ways to copy to clipboard
✅ **Model Transparency**: See exactly what prompts each model uses
✅ **Sport-Specific**: Automatically shows relevant models
✅ **Metadata**: See prompt size and line count
✅ **User-Friendly**: Clear instructions and visual feedback
