import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title='LLM Betting Model Results',
    page_icon='🎲',
    layout='wide'
)

@st.cache_data(ttl=3600)
def load_betting_data(sport):
    """Load betting data from GitHub"""
    if sport == 'NBA':
        url = 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bet_picks_evaluated.csv'
    else:  # NCAAB
        url = 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/ncaab_bet_picks_evaluated.csv'
    
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=600)  # Cache for 10 minutes since these update more frequently
def load_upcoming_bets(sport):
    """Load upcoming/in-progress bets from txt files"""
    
    if sport == 'NBA':
        model_files = {
            'v2': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bets_v2.txt',
            'v2_perp': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bets_v2_perp.txt',
            'gemini': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/nba_bets_gemini.txt'
        }
    else:  # NCAAB (cbb)
        model_files = {
            'claude': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/cbb_bets_claude.txt',
            'perp': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/cbb_bets_perp.txt',
            'gemini': 'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/data/cbb_bets_gemini.txt'
        }
    
    all_bets = []
    for model, url in model_files.items():
        try:
            df = pd.read_csv(url)
            df['model'] = model
            all_bets.append(df)
        except Exception as e:
            st.warning(f"Could not load {model} bets: {e}")
    
    if all_bets:
        combined_df = pd.concat(all_bets, ignore_index=True)
        combined_df['start_time'] = pd.to_datetime(combined_df['start_time'])
        
        # Convert numeric columns to proper types
        numeric_columns = ['rank', 'odds', 'units', 'confidence_pct', 'game_id']
        for col in numeric_columns:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Combine duplicate timestamp columns if they exist
        # Look for columns that might be duplicate timestamps
        timestamp_cols = [col for col in combined_df.columns if 'timestamp' in col.lower()]
        
        if len(timestamp_cols) > 1:
            # Combine them by taking the first non-null value
            combined_df['timestamp_combined'] = combined_df[timestamp_cols[0]].fillna(combined_df[timestamp_cols[1]])
            
            # If there are more than 2, keep filling
            for i in range(2, len(timestamp_cols)):
                combined_df['timestamp_combined'] = combined_df['timestamp_combined'].fillna(combined_df[timestamp_cols[i]])
            
            # Drop the original timestamp columns and rename the combined one
            combined_df = combined_df.drop(columns=timestamp_cols)
            combined_df = combined_df.rename(columns={'timestamp_combined': 'timestamp'})
        
        return combined_df
    else:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_prompt(sport, model):
    """Load prompt text file for a specific sport and model"""
    
    # Map sport names to prompt file prefixes
    sport_prefix = 'nba' if sport == 'NBA' else 'ncaab'
    
    # Construct the URL
    url = f'https://raw.githubusercontent.com/aaroncolesmith/llm_betting_model/refs/heads/main/prompts/{sport_prefix}_prompt_{model}.txt'
    
    try:
        import requests
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error loading prompt: {e}"

def get_available_models(sport):
    """Get list of available models for a sport"""
    if sport == 'NBA':
        return ['gemini', 'v2', 'v2_perp']
    else:  # NCAAB
        return ['claude', 'gemini', 'perp']




def plot_model_performance_over_time(df, sport):
    """Create line chart showing cumulative performance by model"""
    df_sorted = df.sort_values(['model', 'date', 'rank']).copy()
    
    # Calculate cumulative payout for each model
    df_sorted['cumulative_payout'] = df_sorted.groupby('model')['bet_payout'].cumsum()
    
    # Get the last entry for each date/model combination for cleaner visualization
    df_daily = df_sorted.groupby(['model', 'date']).agg({
        'cumulative_payout': 'last',
        'bet_payout': 'sum',
        'units': 'sum'
    }).reset_index()
    
    fig = px.line(df_daily, 
                  x='date', 
                  y='cumulative_payout',
                  color='model',
                  title=f'{sport} Model Performance Over Time (Cumulative Payout)',
                  labels={'cumulative_payout': 'Cumulative Payout (Units)', 
                         'date': 'Date',
                         'model': 'Model'},
                  template='plotly_white')
    
    fig.update_layout(
        font_family='Futura',
        height=600,
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Cumulative Payout (Units)'
    )
    
    fig.update_traces(mode='lines+markers', line=dict(width=3))
    
    # Add a horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def plot_daily_performance(df, sport):
    """Create bar chart showing daily performance by model"""
    df_daily = df.groupby(['model', 'date']).agg({
        'bet_payout': 'sum',
        'units': 'sum'
    }).reset_index()
    
    fig = px.bar(df_daily,
                 x='date',
                 y='bet_payout',
                 color='model',
                 title=f'{sport} Daily Performance by Model',
                 labels={'bet_payout': 'Daily Payout (Units)',
                        'date': 'Date',
                        'model': 'Model'},
                 template='plotly_white',
                 barmode='group')
    
    fig.update_layout(
        font_family='Futura',
        height=500,
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Daily Payout (Units)'
    )
    
    # Add a horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def plot_win_rate_by_model(df, sport):
    """Create bar chart showing win rate by model"""
    df['win'] = (df['bet_result'] == 'win').astype(int)
    
    model_stats = df.groupby('model').agg({
        'win': ['sum', 'count'],
        'bet_payout': 'sum'
    }).reset_index()
    
    model_stats.columns = ['model', 'wins', 'total_bets', 'total_payout']
    model_stats['win_rate'] = (model_stats['wins'] / model_stats['total_bets'] * 100).round(2)
    model_stats['roi'] = ((model_stats['total_payout'] / model_stats['total_bets']) * 100).round(2)
    
    # Create subplot with win rate and ROI
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=model_stats['model'],
        y=model_stats['win_rate'],
        name='Win Rate (%)',
        marker_color='lightblue',
        text=model_stats['win_rate'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'{sport} Win Rate by Model',
        xaxis_title='Model',
        yaxis_title='Win Rate (%)',
        font_family='Futura',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig, model_stats

def main():
    st.title('🎲 LLM Betting Model Results')
    # st.markdown('---')
    
    c1,c2 = st.columns(2)
    # Sport selection
    sport = c1.selectbox('Select Sport', ['NBA', 'NCAAB'])
    
    # Load data
    with st.spinner(f'Loading {sport} betting data...'):
        df = load_betting_data(sport)
    
    c2.success(f'Loaded {len(df):,} bets from {df["date"].min().strftime("%Y-%m-%d")} to {df["date"].max().strftime("%Y-%m-%d")}')
    
    # Overall Statistics
    st.subheader('📊 Overall Performance Summary')
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_bets = len(df)
    total_wins = len(df[df['bet_result'] == 'win'])
    total_payout = df['bet_payout'].sum()
    total_units_wagered = df['units'].sum()
    
    with col1:
        st.metric('Total Bets', f'{total_bets:,}')
    with col2:
        st.metric('Win Rate', f'{(total_wins/total_bets*100):.1f}%')
    with col3:
        st.metric('Total Payout', f'{total_payout:.2f} units')
    with col4:
        roi = (total_payout / total_units_wagered * 100) if total_units_wagered > 0 else 0
        st.metric('ROI', f'{roi:.1f}%')
    
    st.markdown('---')
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        '🔮 Upcoming Bets',
        '📈 Model Performance Over Time',
        '📅 Daily Performance', 
        '🏆 Model Comparison',
        '🗓️ Results by Date',
        '📝 Prompts',
        '🔧 Debug'
    ])
    
    # Tab 1: Upcoming Bets
    with tab1:
        st.subheader('Upcoming & In-Progress Bets')
        
        # Load upcoming bets
        with st.spinner('Loading upcoming bets...'):
            df_upcoming = load_upcoming_bets(sport)
        
        if len(df_upcoming) > 0:
            # Load evaluated bets to check status
            df_evaluated = df
            
            # Create a unique identifier for matching
            df_upcoming['bet_key'] = df_upcoming['game_id'].astype(str) + '_' + df_upcoming['pick'].astype(str)
            df_evaluated['bet_key'] = df_evaluated['game_id'].astype(str) + '_' + df_evaluated['pick'].astype(str)
            
            # Mark which bets have been evaluated
            df_upcoming['evaluated'] = df_upcoming['bet_key'].isin(df_evaluated['bet_key'])
            
            # Merge with evaluated results to get outcome
            df_upcoming = df_upcoming.merge(
                df_evaluated[['bet_key', 'bet_result', 'bet_payout', 'home_score', 'away_score']],
                on='bet_key',
                how='left',
                suffixes=('', '_eval')
            )
            
            # Timezone selection
            col_tz1, col_tz2 = st.columns([1, 3])
            with col_tz1:
                timezone = st.selectbox(
                    'Timezone',
                    ['PT', 'CT', 'ET'],
                    index=0  # Default to PT
                )
            
            # Map timezone to pytz timezone
            tz_map = {
                'PT': 'America/Los_Angeles',
                'CT': 'America/Chicago',
                'ET': 'America/New_York'
            }
            
            # Convert start times to selected timezone
            import pytz
            selected_tz = pytz.timezone(tz_map[timezone])
            df_upcoming['start_time_local'] = df_upcoming['start_time'].dt.tz_convert(selected_tz)
            
            # Get current time in selected timezone
            now_local = pd.Timestamp.now(tz=selected_tz)
            
            # Add status column based on local time
            df_upcoming['status'] = df_upcoming.apply(
                lambda row: 'Completed' if pd.notna(row['bet_result']) 
                else ('In Progress' if row['start_time_local'] < now_local else 'Upcoming'),
                axis=1
            )
            
            # Filter options
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status_filter = st.multiselect(
                    'Filter by Status',
                    ['Upcoming', 'In Progress', 'Completed'],
                    default=['Upcoming', 'In Progress']
                )
            
            with col2:
                model_filter = st.multiselect(
                    'Filter by Model',
                    df_upcoming['model'].unique().tolist(),
                    default=df_upcoming['model'].unique().tolist()
                )
            
            with col3:
                min_confidence = st.slider(
                    'Minimum Confidence %',
                    min_value=0,
                    max_value=100,
                    value=0
                )
            
            with col4:
                sort_by = st.selectbox(
                    'Sort By',
                    ['Start Time', 'Model', 'Confidence'],
                    index=0  # Default to Start Time
                )
            
            # Apply filters
            df_filtered = df_upcoming[
                (df_upcoming['status'].isin(status_filter)) &
                (df_upcoming['model'].isin(model_filter)) &
                (df_upcoming['confidence_pct'] >= min_confidence)
            ].copy()
            
            # Apply sorting
            if sort_by == 'Start Time':
                df_filtered = df_filtered.sort_values(['start_time_local', 'confidence_pct'], ascending=[True, False])
            elif sort_by == 'Model':
                df_filtered = df_filtered.sort_values(['model', 'start_time_local'], ascending=[True, True])
            else:  # Confidence
                df_filtered = df_filtered.sort_values(['confidence_pct', 'start_time_local'], ascending=[False, True])
            
            st.markdown(f'**Showing {len(df_filtered)} bets** (Current time: {now_local.strftime("%Y-%m-%d %I:%M %p %Z")})')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric('Upcoming', len(df_filtered[df_filtered['status'] == 'Upcoming']))
            with col2:
                st.metric('In Progress', len(df_filtered[df_filtered['status'] == 'In Progress']))
            with col3:
                st.metric('Completed', len(df_filtered[df_filtered['status'] == 'Completed']))
            with col4:
                avg_conf = df_filtered['confidence_pct'].mean()
                st.metric('Avg Confidence', f'{avg_conf:.1f}%')
            
            # Display bets in an expandable format
            for idx, row in df_filtered.iterrows():
                # Status emoji
                status_emoji = {
                    'Upcoming': '⏳',
                    'In Progress': '🎮',
                    'Completed': '✅' if pd.notna(row['bet_result']) and row['bet_result'] == 'win' else '❌'
                }
                
                # Create expander title
                result_text = ''
                if row['status'] == 'Completed':
                    result_text = f" - **{row['bet_result'].upper()}** ({row['bet_payout']:+.2f} units)"
                
                expander_title = f"{status_emoji[row['status']]} [{row['model']}] {row['match']} - {row['pick']} ({row['odds']}) - {row['confidence_pct']}% confidence{result_text}"
                
                with st.expander(expander_title):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Game:** {row['match']}")
                        st.markdown(f"**Start Time:** {row['start_time_local'].strftime('%Y-%m-%d %I:%M %p %Z')}")
                        st.markdown(f"**Pick:** {row['pick']}")
                        st.markdown(f"**Odds:** {row['odds']}")
                        st.markdown(f"**Units:** {row['units']}")
                        st.markdown(f"**Confidence:** {row['confidence_pct']}%")
                        
                        if row['status'] == 'Completed':
                            st.markdown(f"**Final Score:** {row['home_score']:.0f} - {row['away_score']:.0f}")
                            st.markdown(f"**Result:** {row['bet_result'].upper()}")
                            st.markdown(f"**Payout:** {row['bet_payout']:+.2f} units")
                    
                    with col2:
                        st.markdown(f"**Status:** {row['status']}")
                        st.markdown(f"**Model:** {row['model']}")
                        st.markdown(f"**Rank:** {row['rank']}")
                        if pd.notna(row.get('predicted_score')):
                            st.markdown(f"**Predicted Score:** {row['predicted_score']}")
                    
                    # Reason in a nice box
                    st.markdown("---")
                    st.markdown("**Betting Rationale:**")
                    st.info(row['reason'])
            
        else:
            st.info('No upcoming bets found.')

    
    # Tab 2: Model Performance Over Time
    with tab2:
        st.subheader('Model Performance Over Time')
        fig_cumulative = plot_model_performance_over_time(df, sport)
        st.plotly_chart(fig_cumulative, use_container_width=True)
        
        st.markdown("""
        **How to read this chart:**
        - Each line represents a different model's cumulative payout over time
        - The y-axis shows total units won/lost
        - Lines above 0 indicate profit, below 0 indicate loss
        - Steeper upward slopes indicate better recent performance
        """)
    
    # Tab 3: Daily Performance
    with tab3:
        st.subheader('Daily Performance')
        fig_daily = plot_daily_performance(df, sport)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        st.markdown("""
        **How to read this chart:**
        - Each bar represents a model's performance on a specific day
        - Positive values (above 0) indicate profit for that day
        - Negative values (below 0) indicate loss for that day
        - Compare bars side-by-side to see which model performed best on each day
        """)
    
    # Tab 4: Model Comparison
    with tab4:
        st.subheader('Model Comparison')
        
        fig_win_rate, model_stats = plot_win_rate_by_model(df, sport)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(fig_win_rate, use_container_width=True)
        
        with col2:
            st.markdown('**Model Statistics**')
            # Format the dataframe for display
            display_stats = model_stats.copy()
            display_stats['total_payout'] = display_stats['total_payout'].apply(lambda x: f'{x:.2f}')
            display_stats.columns = ['Model', 'Wins', 'Total Bets', 'Total Payout', 'Win Rate (%)', 'ROI (%)']
            st.dataframe(display_stats, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Key Metrics:**
        - **Win Rate**: Percentage of bets that won
        - **ROI (Return on Investment)**: Percentage return on units wagered
        - **Total Payout**: Net units won/lost (positive = profit, negative = loss)
        """)
    
    # Tab 5: Results by Date
    with tab5:
        st.subheader('Results by Date')
        
        # Get available dates
        available_dates = sorted(df['date'].dt.date.unique(), reverse=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_date = st.selectbox(
                'Select Date',
                available_dates,
                format_func=lambda x: x.strftime('%Y-%m-%d')
            )
        
        # Filter data for selected date
        df_date = df[df['date'].dt.date == selected_date].copy()
        
        if len(df_date) > 0:
            st.markdown(f'**Showing {len(df_date)} bets for {selected_date.strftime("%Y-%m-%d")}**')
            
            # Summary metrics for the selected date
            col1, col2, col3, col4 = st.columns(4)
            
            date_wins = len(df_date[df_date['bet_result'] == 'win'])
            date_payout = df_date['bet_payout'].sum()
            date_units = df_date['units'].sum()
            
            with col1:
                st.metric('Bets', len(df_date))
            with col2:
                st.metric('Wins', date_wins)
            with col3:
                st.metric('Win Rate', f'{(date_wins/len(df_date)*100):.1f}%')
            with col4:
                st.metric('Net Payout', f'{date_payout:.2f} units')
            
            # Display table
            st.markdown('**Bet Details**')
            
            # Format the display dataframe
            display_df = df_date[['rank', 'model', 'match', 'pick', 'odds', 'units', 
                                  'home_score', 'away_score', 'bet_result', 'bet_payout']].copy()
            
            # Add color coding for wins/losses
            def color_result(val):
                if val == 'win':
                    return 'background-color: #90EE90'
                else:
                    return 'background-color: #FFB6C6'
            
            # Sort by model and rank
            display_df = display_df.sort_values(['model', 'rank'])
            
            # Format numeric columns
            display_df['odds'] = display_df['odds'].apply(lambda x: f'{x:+.0f}' if x > 0 else f'{x:.0f}')
            display_df['bet_payout'] = display_df['bet_payout'].apply(lambda x: f'{x:+.2f}')
            
            # Rename columns for better display
            display_df.columns = ['Rank', 'Model', 'Match', 'Pick', 'Odds', 'Units', 
                                 'Home Score', 'Away Score', 'Result', 'Payout']
            
            st.dataframe(
                display_df.style.applymap(color_result, subset=['Result']),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Model breakdown for the date
            st.markdown('**Performance by Model for Selected Date**')
            
            model_date_stats = df_date.groupby('model').agg({
                'bet_result': lambda x: (x == 'win').sum(),
                'game_id': 'count',
                'bet_payout': 'sum',
                'units': 'sum'
            }).reset_index()
            
            model_date_stats.columns = ['Model', 'Wins', 'Total Bets', 'Net Payout', 'Units Wagered']
            model_date_stats['Win Rate'] = (model_date_stats['Wins'] / model_date_stats['Total Bets'] * 100).round(1)
            model_date_stats['Net Payout'] = model_date_stats['Net Payout'].round(2)
            
            st.dataframe(model_date_stats, use_container_width=True, hide_index=True)
        else:
            st.info(f'No bets found for {selected_date.strftime("%Y-%m-%d")}')
    
    # Tab 6: Prompts
    with tab6:
        st.subheader('📝 Model Prompts')
        
        st.markdown('**View and copy the prompts used by each betting model.**')
        
        # Get available models for the selected sport
        available_models = get_available_models(sport)
        
        # Model selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_model = st.selectbox(
                'Select Model',
                available_models,
                format_func=lambda x: x.upper()
            )
        
        # Load the prompt
        with st.spinner(f'Loading {selected_model} prompt...'):
            prompt_text = load_prompt(sport, selected_model)
        
        # Extract timestamp from prompt
        import re
        
        # Try multiple patterns to find timestamp
        timestamp_patterns = [
            r'`timestamp`:\s*use the time of this prompt\s*--\s*([\d\-\s:.]+)',
            r'timestamp:\s*([\d\-\s:.]+)',
            r'Prompt created:\s*([\d\-\s:.]+)',
            r'Generated:\s*([\d\-\s:.]+)'
        ]
        
        timestamp_display = None
        for pattern in timestamp_patterns:
            timestamp_match = re.search(pattern, prompt_text, re.IGNORECASE)
            if timestamp_match:
                try:
                    timestamp_str = timestamp_match.group(1).strip()
                    # Parse the timestamp
                    timestamp_dt = pd.to_datetime(timestamp_str)
                    
                    # Convert to Pacific Time
                    import pytz
                    utc_tz = pytz.UTC
                    pt_tz = pytz.timezone('America/Los_Angeles')
                    
                    # Assume the timestamp is in UTC, convert to PT
                    if timestamp_dt.tzinfo is None:
                        timestamp_dt = utc_tz.localize(timestamp_dt)
                    
                    timestamp_pt = timestamp_dt.astimezone(pt_tz)
                    
                    # Format as MM/DD H:MMAM/PM (cross-platform compatible)
                    month = timestamp_pt.month
                    day = timestamp_pt.day
                    hour = timestamp_pt.hour
                    minute = timestamp_pt.minute
                    
                    # Convert to 12-hour format
                    if hour == 0:
                        hour_12 = 12
                        am_pm = 'AM'
                    elif hour < 12:
                        hour_12 = hour
                        am_pm = 'AM'
                    elif hour == 12:
                        hour_12 = 12
                        am_pm = 'PM'
                    else:
                        hour_12 = hour - 12
                        am_pm = 'PM'
                    
                    timestamp_display = f'{month}/{day} {hour_12}:{minute:02d}{am_pm}'
                    break  # Found a timestamp, stop searching
                except Exception:
                    continue  # Try next pattern
        
        # Display prompt info
        if timestamp_display:
            # Show 4 columns with timestamp
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric('Prompt File', f'{sport.lower()}_prompt_{selected_model}.txt')
            with col2:
                st.metric('Created', timestamp_display)
            with col3:
                st.metric('Characters', f'{len(prompt_text):,}')
            with col4:
                st.metric('Lines', f'{len(prompt_text.splitlines()):,}')
        else:
            # Show 3 columns without timestamp
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric('Prompt File', f'{sport.lower()}_prompt_{selected_model}.txt')
            with col2:
                st.metric('Characters', f'{len(prompt_text):,}')
            with col3:
                st.metric('Lines', f'{len(prompt_text.splitlines()):,}')

        
        # Copy button and prompt display
        st.markdown('---')
        
        # Create a text area with the prompt
        st.text_area(
            'Prompt Content',
            value=prompt_text,
            height=400,
            help='Click inside and use Ctrl+A (Cmd+A on Mac) to select all, then Ctrl+C (Cmd+C) to copy',
            key=f'prompt_{sport}_{selected_model}'
        )
        
        # Add a copy button using Streamlit's built-in functionality
        if st.button('📋 Copy to Clipboard', key=f'copy_btn_{sport}_{selected_model}'):
            # Use st.code with a copy button
            st.code(prompt_text, language='text')
            st.success('✅ Prompt displayed above with copy button!')
        
        # Additional info
        st.markdown('---')
        st.markdown('**Tips:**')
        st.info("""
        - **Select all text**: Click in the text area and press `Ctrl+A` (Windows/Linux) or `Cmd+A` (Mac)
        - **Copy**: Press `Ctrl+C` (Windows/Linux) or `Cmd+C` (Mac)
        - **Alternative**: Click the "Copy to Clipboard" button to see the prompt in a code block with a built-in copy button
        """)
        
        # Show all available prompts for this sport
        with st.expander('View all available prompts for this sport'):
            st.markdown(f'**Available {sport} prompts:**')
            for model in available_models:
                st.markdown(f'- `{sport.lower()}_prompt_{model}.txt`')

    
    # Tab 7: Debug
    with tab7:
        st.subheader('Debug - Raw Data View')
        
        st.markdown('**Use this tab to inspect the raw data and troubleshoot any issues.**')
        
        # Show evaluated bets
        st.markdown('---')
        st.markdown('### Evaluated Bets (Results)')
        st.markdown(f'**Total rows:** {len(df)}')
        
        with st.expander('View Evaluated Bets DataFrame'):
            st.dataframe(df, use_container_width=True, height=400)
        
        with st.expander('View Evaluated Bets Column Info'):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Show upcoming bets
        st.markdown('---')
        st.markdown('### Upcoming Bets (Raw)')
        
        df_upcoming_debug = load_upcoming_bets(sport)
        
        if len(df_upcoming_debug) > 0:
            st.markdown(f'**Total rows:** {len(df_upcoming_debug)}')
            
            with st.expander('View Upcoming Bets DataFrame'):
                st.dataframe(df_upcoming_debug, use_container_width=True, height=400)
            
            with st.expander('View Upcoming Bets Column Info'):
                col_info = pd.DataFrame({
                    'Column': df_upcoming_debug.columns,
                    'Type': df_upcoming_debug.dtypes.values,
                    'Non-Null Count': df_upcoming_debug.count().values,
                    'Null Count': df_upcoming_debug.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
            
            with st.expander('View Start Times (UTC)'):
                time_df = df_upcoming_debug[['model', 'match', 'start_time']].copy()
                time_df = time_df.sort_values('start_time')
                st.dataframe(time_df, use_container_width=True)
            
            with st.expander('View Start Times (with Timezone Conversion)'):
                import pytz
                pt_tz = pytz.timezone('America/Los_Angeles')
                time_df_tz = df_upcoming_debug[['model', 'match', 'start_time']].copy()
                time_df_tz['start_time_pt'] = time_df_tz['start_time'].dt.tz_convert(pt_tz)
                time_df_tz['current_time_pt'] = pd.Timestamp.now(tz=pt_tz)
                time_df_tz['time_diff_hours'] = (time_df_tz['start_time_pt'] - time_df_tz['current_time_pt']).dt.total_seconds() / 3600
                time_df_tz = time_df_tz.sort_values('start_time_pt')
                st.dataframe(time_df_tz, use_container_width=True)
        else:
            st.info('No upcoming bets data available')


if __name__ == '__main__':
    main()
