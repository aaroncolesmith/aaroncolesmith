import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_nba_games():
    """Load NBA games data from GitHub."""
    df = pd.read_parquet(
        'https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_games.parquet',
        engine='pyarrow'
    )
    return df


@st.cache_data(ttl=3600)
def load_nba_box_scores():
    """Load NBA box scores data from GitHub."""
    df = pd.read_parquet(
        'https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_box_scores.parquet',
        engine='pyarrow'
    )
    df = df.loc[df['mp'] > 0]
    
    # Convert to numeric and calculate derived stats
    df['gmsc'] = pd.to_numeric(df['gmsc'])
    df['bpm'] = pd.to_numeric(df['bpm'])
    df['ortg'] = pd.to_numeric(df['ortg'])
    df['drtg'] = pd.to_numeric(df['drtg'])
    df['missed_shots'] = (df['fga'].fillna(0) - df['fg'].fillna(0)) + (df['fta'].fillna(0) - df['ft'].fillna(0))
    df['all_stat'] = df['pts'] + df['trb'] + df['ast']

    # Calculate DARKO Lite metric
    df["darko_lite"] = (
        0.25 * df["bpm"] +
        0.20 * (df["ortg"] - df["drtg"]) +
        0.15 * df["usg_pct"] +
        0.15 * df["ts_pct"].fillna(0) +
        0.10 * df["ast_pct"] +
        0.10 * df["trb_pct"] +
        0.05 * df["stl_pct"] +
        0.05 * df["blk_pct"] -
        0.05 * df["tov_pct"].fillna(0)
    )

    return df


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def aggregate_box_scores(df):
    """Aggregate box scores by player."""
    df = df.groupby(['player']).agg(
        team=('team', lambda x: ', '.join(set(x))),
        mp=('mp', 'sum'),
        gp=('game_id', 'nunique'),
        fg=('fg', 'sum'),
        fga=('fga', 'sum'),
        threep=('3p', 'sum'),
        threepa=('3pa', 'sum'),
        ft=('ft', 'sum'),
        fta=('fta', 'sum'),
        pts=('pts', 'sum'),
        ast=('ast', 'sum'),
        trb=('trb', 'sum'),
        stl=('stl', 'sum'),
        blk=('blk', 'sum'),
        orb=('orb', 'sum'),
        drb=('drb', 'sum'),
        bpm=('bpm', 'sum'),
        tov=('tov', 'sum'),
        gmsc=('gmsc', 'mean'),
        orgt=('ortg', 'mean'),
        drtg=('drtg', 'mean'),
        usg_pct=('usg_pct', 'mean'),
        ts_pct=('ts_pct', 'mean'),
        efg_pct=('efg_pct', 'mean'),
        plus_minus=('+/-', 'sum'),
        missed_shots=('missed_shots', 'sum'),
        all_stat=('all_stat', 'sum'),
        wins=('win', 'sum'),
        losses=('loss', 'sum'),
        playoff_games=('playoff_game', 'sum'),
        playoff_wins=('playoff_win', 'sum'),
        playoff_losses=('playoff_loss', 'sum'),
        darko_lite=('darko_lite', 'mean'),
    ).reset_index()
    
    # Calculate per-game and per-minute stats
    df['fg_pct'] = df['fg'] / df['fga']
    df['3p_pct'] = df['threep'] / df['threepa']
    df['ft_pct'] = df['ft'] / df['fta']
    df['ppg'] = df['pts'] / df['gp']
    df['apg'] = df['ast'] / df['gp']
    df['rpg'] = df['trb'] / df['gp']
    df['spg'] = df['stl'] / df['gp']
    df['bpg'] = df['blk'] / df['gp']
    df['ppm'] = df['pts'] / df['mp']
    df['apm'] = df['ast'] / df['mp']
    df['rpm'] = df['trb'] / df['mp']
    df['spm'] = df['stl'] / df['mp']
    df['tovpm'] = df['tov'] / df['mp']
    df['plus_minus_per_min'] = df['plus_minus'] / df['mp']
    df['missed_shots_per_game'] = df['missed_shots'] / df['gp']
    df['win_pct'] = df['wins'] / df['gp']
    df['playoff_win_pct'] = df['playoff_wins'] / df['playoff_games']

    return df


def add_game_outcomes(df, games_df):
    """Add win/loss columns to box score data."""
    df['playoff_game'] = np.where(df['game_type'] == 'Playoffs', 1, 0)
    df['win'] = np.select(
        [
            (df['team'] == df['home_team']) & (df['home_score'] > df['visitor_score']),
            (df['team'] == df['home_team']) & (df['home_score'] < df['visitor_score']),
            (df['team'] != df['home_team']) & (df['home_score'] < df['visitor_score']),
            (df['team'] != df['home_team']) & (df['home_score'] > df['visitor_score'])
        ],
        [1, 0, 1, 0],
    )
    df['loss'] = np.where(df['win'] == 0, 1, 0)
    df['playoff_win'] = np.where(df['playoff_game'] == 1, df['win'], 0)
    df['playoff_loss'] = np.where(df['playoff_game'] == 1, 1 - df['win'], 0)
    return df


def rename_columns(df):
    """Rename columns to be more user-friendly."""
    df.rename(columns={
        "3par": "3pa Rate",
        "ts_pct": "True Shot Pct",
        "pts": "Points",
        "ast": "Assists",
        "efg_pct": "Eff FG Pct",
        "drtg": "Def Rtg",
        "ortg": "Off Rtg",
        "3par": "3pa Rate",
        "ft": "Free Throws",
        "fta": "Free Throws Attempted",
        "stl_pct": "Steal Pct",
        "player": "Player",
        "team": "Team",
        "mp": "Minutes",
        "fg": "Field Goals",
        "fga": "Field Goals Attempted",
        "fg_pct": "Field Goal Pct",
        "ftp_pct": "Free Throw Pct",
        "trb_pct": "Total Rebound Pct",
        "trb": "Rebounds",
        "stl": "Steals",
        "bpm": "Box +/-",
        "drb": "Def Reb",
        "orb": "Off Reb",
        "pf": "Fouls",
        "tov": "Turnovers",
        "ftr": "FT Rate",
        "gmsc": "Game Score",
        "3p_pct": '3p Pct',
        'ppm': 'Points Per Minute',
        'apm': 'Assists Per Minute',
        'rpm': 'Rebounds Per Minute',
        'spm': 'Steals Per Minute',
        'ppg': 'Points Per Game',
        'apg': 'Assists Per Game',
        'rpg': 'Rebounds Per Game',
        'spg': 'Steals Per Game',
        'bpg': 'Blocks Per Game',
        'threep': '3 Pointers Made',
        'threepa': '3 Pointers Attempted',
        'gp': 'Games Played',
        'blk': 'Blocks',
        'orgt': 'Offensive Rating',
        'drtg': 'Defensive Rating',
        'ft_pct': 'Free Throw Pct',
        "usg_pct": "Usage Rate",
        'darko_lite': 'Darko Lite'
    }, inplace=True)
    
    return df


def get_num_cols(df):
    """Separate numeric and non-numeric columns."""
    num_cols = []
    non_num_cols = []
    for col in df.columns:
        if col in ['date', 'last_game']:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
        else:
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except:
                non_num_cols.append(col)
    return num_cols, non_num_cols


# ============================================================================
# CLUSTERING AND VISUALIZATION FUNCTIONS
# ============================================================================

def perform_clustering(df, num_cols):
    """
    Perform PCA and K-Means clustering on the dataframe.
    Returns: df with cluster assignments, pca object, feature importance dataframe
    """
    df_work = df.copy()
    df_features = df_work[num_cols].copy()
    
    # Handle invalid values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(0)
    df_features = df_features.clip(lower=-1e10, upper=1e10)
    
    # Scale features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_features)
    
    # PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=2, n_init=10)
    cluster_labels = kmeans.fit_predict(x_pca)
    
    # Add results to dataframe
    df_work['Cluster'] = cluster_labels.astype(int)
    df_work['Cluster_x'] = x_pca[:, 0]
    df_work['Cluster_y'] = x_pca[:, 1]
    
    # Calculate feature importance (absolute value of PCA components)
    feature_importance = pd.DataFrame({
        'Feature': num_cols,
        'PC1': np.abs(pca.components_[0]),
        'PC2': np.abs(pca.components_[1])
    })
    feature_importance['Importance'] = feature_importance['PC1'] + feature_importance['PC2']
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return df_work, pca, feature_importance


def generate_color_map(df, color_col):
    """Generate a color map for the color column."""
    hex_colors = ['#ffd900', '#ff2a00', '#35d604', '#59ffee', '#1d19ff', '#ff19fb']
    
    def generate_random_hex_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    unique_values = df[color_col].unique()
    
    while len(hex_colors) < len(unique_values):
        hex_colors.append(generate_random_hex_color())
    
    discrete_color_map = {value: hex_colors[i] for i, value in enumerate(unique_values)}
    return discrete_color_map


def create_cluster_scatter(df, color_col, hover_cols, highlighted_players=None, feature_importance_df=None):
    """Create clustering scatter plot with feature direction labels."""
    df = df.copy()
    df[color_col] = df[color_col].astype('str')
    
    discrete_color_map = generate_color_map(df, color_col)
    
    if highlighted_players is None or len(highlighted_players) == 0:
        opacity_values = 0.75
        scatter_kwargs = {
            'color': color_col,
            'color_discrete_map': discrete_color_map,
            'category_orders': {color_col: df.sort_values(color_col, ascending=True)[color_col].unique().tolist()},
            'opacity': opacity_values
        }
    else:
        opacity_values = np.where(df['Player'].isin(highlighted_players), 0.9, 0.2)
        scatter_kwargs = {'opacity': opacity_values}
    
    fig = px.scatter(
        df,
        x='Cluster_x',
        y='Cluster_y',
        hover_data=hover_cols,
        template='simple_white',
        **scatter_kwargs
    )
    
    # Add player labels for highlighted players
    if highlighted_players is not None and len(highlighted_players) > 0:
        for _, row in df[df['Player'].isin(highlighted_players)].iterrows():
            fig.add_annotation(
                x=row['Cluster_x'],
                y=row['Cluster_y'] + 0.015,
                text=row['Player'],
                bgcolor="gray",
                opacity=0.85,
                showarrow=False,
                font=dict(size=12, color="#ffffff")
            )
    
    # Add feature direction labels if feature importance is provided
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        # Get the feature names and their PC1/PC2 components
        feature_labels = feature_importance_df.copy()
        
        # Create a secondary clustering of features to find representative ones
        feature_coords = feature_labels[['PC1', 'PC2']].copy()
        
        # Scale the coordinates
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_coords)
        
        # PCA on features
        pca_features = PCA(n_components=2)
        feature_pca = pca_features.fit_transform(feature_scaled)
        
        # K-means to cluster features
        num_clusters = min(3, len(feature_labels))
        if len(feature_labels) >= 2:
            kmeans_features = KMeans(n_clusters=num_clusters, random_state=2, n_init=10)
            feature_labels['cluster'] = kmeans_features.fit_predict(feature_pca)
            feature_labels['pca_x'] = feature_pca[:, 0]
            feature_labels['pca_y'] = feature_pca[:, 1]
            
            # Calculate distance from origin for each feature
            feature_labels['distance_from_origin'] = np.sqrt(
                feature_labels['PC1']**2 + feature_labels['PC2']**2
            )
            
            # Get key features: those at extremes of PC1 and PC2
            key_features = pd.DataFrame()
            key_features = pd.concat([key_features, feature_labels.nlargest(3, 'PC1')])
            key_features = pd.concat([key_features, feature_labels.nsmallest(3, 'PC1')])
            key_features = pd.concat([key_features, feature_labels.nlargest(3, 'PC2')])
            key_features = pd.concat([key_features, feature_labels.nsmallest(3, 'PC2')])
            key_features = key_features.drop_duplicates(subset=['Feature'])
            
            # Calculate scaling factors to position labels
            x_range = df['Cluster_x'].max() - df['Cluster_x'].min()
            y_range = df['Cluster_y'].max() - df['Cluster_y'].min()
            
            pc1_range = key_features['PC1'].max() - key_features['PC1'].min()
            pc2_range = key_features['PC2'].max() - key_features['PC2'].min()
            
            x_factor = (x_range * 0.7) / pc1_range if pc1_range > 0 else 1
            y_factor = (y_range * 0.7) / pc2_range if pc2_range > 0 else 1
            
            # Round for grouping
            key_features['PC1_rounded'] = key_features['PC1'].round(1)
            key_features['PC2_rounded'] = key_features['PC2'].round(1)
            
            # Group nearby features together
            feature_groups = key_features.groupby(['PC1_rounded', 'PC2_rounded']).agg(
                features=('Feature', lambda x: '<br>'.join(x)),
                pc1_mean=('PC1', 'mean'),
                pc2_mean=('PC2', 'mean')
            ).reset_index()
            
            # Add annotations for feature groups
            for _, row in feature_groups.iterrows():
                x_pos = row['pc1_mean'] * x_factor
                y_pos = row['pc2_mean'] * y_factor
                
                # Constrain to plot boundaries
                x_pos = max(df['Cluster_x'].min(), min(x_pos, df['Cluster_x'].max()))
                y_pos = max(df['Cluster_y'].min(), min(y_pos, df['Cluster_y'].max()))
                
                fig.add_annotation(
                    x=x_pos,
                    y=y_pos,
                    text=row['features'],
                    showarrow=False,
                    opacity=0.25,
                    font=dict(color="black", size=12),
                    bgcolor="rgba(255, 255, 255, 0.5)",
                    borderpad=4
                )
    
    fig.update_layout(
        legend_title_text=color_col,
        font_family='Futura',
        height=700,
        font_color='black',
        title="Player Clustering Analysis"
    )
    
    fig.update_traces(
        mode='markers',
        marker=dict(size=16, line=dict(width=2, color='DarkSlateGrey'))
    )
    
    fig.update_xaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    fig.update_yaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    
    # Fix hover template
    for i in range(len(fig.data)):
        default_template = fig.data[i].hovertemplate
        updated_template = default_template.replace('=', ': ')
        fig.data[i].hovertemplate = updated_template
    
    return fig


def create_feature_importance_bar(feature_importance_df, top_n=15):
    """Create bar chart showing feature importance."""
    top_features = feature_importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Most Important Statistics for Clustering',
        template='simple_white',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        font_family='Futura',
        height=600,
        font_color='black',
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    fig.update_traces(
        marker=dict(line=dict(color='navy', width=2))
    )
    
    return fig


def create_radial_comparison(df, selected_players, key_stats, fill=True):
    """Create radial chart comparing selected players."""
    if len(selected_players) == 0:
        return None
    
    df_selected = df[df['Player'].isin(selected_players)].copy()
    
    # Scale the key stats
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_selected[key_stats])
    scaled_cols = [f'{col}_scaled' for col in key_stats]
    df_selected[scaled_cols] = scaled_values
    
    # Melt for plotting
    df_melted = pd.melt(
        df_selected,
        id_vars=['Player'],
        value_vars=scaled_cols,
        var_name='Statistic',
        value_name='Value'
    )
    
    df_melted_unscaled = pd.melt(
        df_selected,
        id_vars=['Player'],
        value_vars=key_stats,
        var_name='Statistic',
        value_name='Value_Unscaled'
    )
    
    df_melted_unscaled['Statistic'] = df_melted_unscaled['Statistic'].str.replace('_scaled', '')
    df_melted['Statistic'] = df_melted['Statistic'].str.replace('_scaled', '')
    df_melted = pd.merge(
        df_melted,
        df_melted_unscaled,
        left_on=['Player', 'Statistic'],
        right_on=['Player', 'Statistic'],
        how='left'
    )
    
    fig = px.line_polar(
        df_melted,
        r='Value',
        theta='Statistic',
        color='Player',
        line_close=True,
        template='simple_white',
        hover_data=['Value_Unscaled'],
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    if fill:
        fig.update_traces(
            fill='toself',
            mode='lines+markers',
            marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
            line=dict(width=4)
        )
    else:
        fig.update_traces(
            mode='lines+markers',
            marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
            line=dict(width=4)
        )
    
    fig.update_layout(
        title=f"Player Comparison: {', '.join(selected_players)}",
        font_family='Futura',
        height=700,
        font_color='black',
        showlegend=True
    )
    
    return fig


def create_top_players_bar(df, stat_name, color_col='Cluster'):
    """Create bar chart showing top players for a specific stat."""
    # Get possible hover fields
    possible_hover_fields = ['Team', 'team', 'Position', 'position', 'pos', 'Cluster']
    hover_fields = [col for col in possible_hover_fields if col in df.columns]
    hover_data = {col: True for col in hover_fields}
    
    # Create a copy and sort by stat
    df_plot = df.copy().sort_values(stat_name, ascending=False)
    
    fig = px.bar(
        df_plot,
        x='Player',
        y=stat_name,
        hover_data=hover_data,
        template='simple_white',
        color_discrete_sequence=['#1f77b4']  # Single color for all bars
    )
    
    fig.update_xaxes(categoryorder='total descending')
    fig.update_traces(
        marker=dict(
            color='#1f77b4',
            line=dict(color='navy', width=2)
        )
    )
    
    fig.update_layout(
        height=700,
        title=f"{stat_name} - Top Players",
        font=dict(family='Futura', size=12, color='black'),
        showlegend=False
    )
    
    # Fix hover template
    for i in range(len(fig.data)):
        default_template = fig.data[i].hovertemplate
        updated_template = default_template.replace('=', ': ')
        fig.data[i].hovertemplate = updated_template
    
    return fig


# ============================================================================
# VIEW FUNCTIONS (Single Game, Date Range, Player Comparison)
# ============================================================================

def single_game_view(games_df, box_scores_df):
    """Single game visualization view."""
    st.header('🏀 Single Game Analysis')
    
    # Date and game selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        date = st.date_input(
            "Select a date",
            value=pd.to_datetime(games_df.date.max()),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(games_df.date.max()),
            key='single_game_date'
        )
    
    date = pd.to_datetime(date)
    games_df['game_str'] = (
        games_df['date'].dt.strftime('%Y-%m-%d') + ' - ' +
        games_df['visitor_team'] + ' ' +
        games_df['visitor_score'].astype('str') + '-' +
        games_df['home_score'].astype('str') + ' ' +
        games_df['home_team']
    )
    
    games = games_df.loc[games_df.date == date]['game_str'].tolist()
    
    with col2:
        if len(games) == 0:
            st.warning("No games found for this date.")
            return
        game_select = st.selectbox('Select a game:', games, key='single_game_select')
    
    game_id = games_df.loc[games_df.game_str == game_select].game_id.min()
    df = box_scores_df.loc[box_scores_df.game_id == game_id].copy()
    
    # Clean and prepare data
    try:
        df['+/-'] = pd.to_numeric(df['+/-'].astype('str').str.replace('+', ''))
    except:
        pass
    
    df = rename_columns(df)
    num_cols, non_num_cols = get_num_cols(df)
    
    # Configuration form
    with st.form(key='single_game_form'):
        st.subheader("Configure Analysis")
        
        num_cols_select = st.multiselect(
            'Select statistics for clustering analysis',
            num_cols,
            default=num_cols[:10] if len(num_cols) > 10 else num_cols
        )
        
        non_num_cols_select = st.multiselect(
            'Select columns for hover data',
            non_num_cols,
            default=['Player'] if 'Player' in non_num_cols else []
        )
        
        color_col = 'Team' if 'Team' in df.columns else 'team'
        
        mp_filter = st.slider(
            'Filter by minimum minutes played',
            min_value=0.0,
            max_value=float(df['Minutes'].max()),
            value=0.0,
            step=1.0
        )
        
        submit_button = st.form_submit_button(label='🔍 Analyze Game', type='primary')
    
    # Store form data in session state
    if submit_button:
        st.session_state['single_game_submitted'] = True
        st.session_state['single_game_data'] = {
            'df': df.query("Minutes > @mp_filter").copy(),
            'num_cols_select': num_cols_select,
            'non_num_cols_select': non_num_cols_select,
            'color_col': color_col
        }
    
    # Check if we have submitted data
    if st.session_state.get('single_game_submitted') and len(st.session_state.get('single_game_data', {}).get('num_cols_select', [])) > 1:
        data = st.session_state['single_game_data']
        df_filtered = data['df']
        num_cols_select = data['num_cols_select']
        non_num_cols_select = data['non_num_cols_select']
        color_col = data['color_col']
        
        # Perform clustering
        df_clustered, pca, feature_importance = perform_clustering(df_filtered, num_cols_select)
        
        # Create three tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "📊 Clustering View",
            "📈 Feature Importance",
            "🎯 Player Comparison"
        ])
        
        with tab1:
            st.subheader("Player Clustering")
            fig = create_cluster_scatter(
                df_clustered,
                color_col,
                non_num_cols_select + num_cols_select[:5],
                feature_importance_df=feature_importance
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Clustering Summary:** {len(df_clustered)} players grouped into 5 clusters based on selected statistics.")
        
        with tab2:
            st.subheader("Most Impactful Statistics")
            fig = create_feature_importance_bar(feature_importance)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("This chart shows which statistics contribute most to the clustering model. Higher values indicate greater importance.")
        
        with tab3:
            st.subheader("Compare Players")
            
            # Get top stats for comparison
            top_stats = feature_importance.head(15)['Feature'].tolist()
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                players_to_compare = st.multiselect(
                    'Select players to compare (2-6 recommended)',
                    df_clustered['Player'].sort_values().tolist(),
                    default=df_clustered.nlargest(3, 'Minutes')['Player'].tolist(),
                    key='single_game_players'
                )
            
            with col2:
                fill_checkbox = st.checkbox(
                    'Fill areas',
                    value=True,
                    help='Uncheck to see hover data more easily',
                    key='single_game_fill'
                )
            
            # Stat selection for radial chart
            stats_for_comparison = st.multiselect(
                'Select statistics to include in comparison (8-12 recommended)',
                top_stats,
                default=top_stats[:8],
                key='single_game_stats_select'
            )
            
            if len(players_to_compare) > 0 and len(stats_for_comparison) > 0:
                fig = create_radial_comparison(df_clustered, players_to_compare, stats_for_comparison, fill=fill_checkbox)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one player and one statistic to compare.")
        
        # NEW: Add a fourth tab for bar charts of top features
        st.markdown("---")
        st.subheader("🏆 Top Players by Most Important Statistics")
        st.markdown("Explore which players rank highest for each key statistic identified by the clustering model.")
        
        # Get top features
        top_features = feature_importance.head(10)['Feature'].tolist()
        
        # Create tabs for each top feature
        if len(top_features) > 0:
            feature_tabs = st.tabs(top_features)
            for i, tab in enumerate(feature_tabs):
                with tab:
                    stat_name = top_features[i]
                    fig = create_top_players_bar(df_clustered, stat_name, color_col)
                    st.plotly_chart(fig, use_container_width=True)


def date_range_view(games_df, box_scores_df):
    """Date range visualization view."""
    st.header('📅 Date Range Analysis')
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start date",
            value=pd.to_datetime(games_df.date.max()) - pd.DateOffset(months=1),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(games_df.date.max()),
            key='date_range_start'
        )
    
    with col2:
        end_date = st.date_input(
            "End date",
            value=pd.to_datetime(games_df.date.max()),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(games_df.date.max()),
            key='date_range_end'
        )
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter and prepare data
    df = box_scores_df.loc[(box_scores_df.date >= start_date) & (box_scores_df.date <= end_date)].copy()
    df = pd.merge(df.loc[df.player != 'Eddie Johnson'], games_df).sort_values(by=['date', 'player'])
    df = add_game_outcomes(df, games_df)
    df = aggregate_box_scores(df)
    df = rename_columns(df)
    
    num_cols, non_num_cols = get_num_cols(df)
    
    # Configuration form
    with st.form(key='date_range_form'):
        st.subheader("Configure Analysis")
        
        num_cols_select = st.multiselect(
            'Select statistics for clustering analysis',
            num_cols,
            default=num_cols[:10] if len(num_cols) > 10 else num_cols
        )
        
        non_num_cols_select = st.multiselect(
            'Select columns for hover data',
            non_num_cols,
            default=['Player'] if 'Player' in non_num_cols else []
        )
        
        show_player_select = st.multiselect(
            'Highlight specific players on the graph',
            df['Player'].unique().tolist()
        )
        
        mp_filter = st.slider(
            'Filter by minimum minutes played',
            min_value=0.0,
            max_value=float(df['Minutes'].max()),
            value=float(df['Minutes'].quantile(0.25)),
            step=100.0
        )
        
        submit_button = st.form_submit_button(label='🔍 Analyze Date Range', type='primary')
    
    # Store form data in session state
    if submit_button:
        df_filtered = df.query("Minutes > @mp_filter").copy()
        st.session_state['date_range_submitted'] = True
        st.session_state['date_range_data'] = {
            'df_filtered': df_filtered,
            'num_cols_select': num_cols_select,
            'non_num_cols_select': non_num_cols_select,
            'show_player_select': show_player_select,
            'start_date': start_date,
            'end_date': end_date
        }
    
    # Check if we have submitted data
    if st.session_state.get('date_range_submitted') and len(st.session_state.get('date_range_data', {}).get('num_cols_select', [])) > 1:
        data = st.session_state['date_range_data']
        df_filtered = data['df_filtered']
        num_cols_select = data['num_cols_select']
        non_num_cols_select = data['non_num_cols_select']
        show_player_select = data['show_player_select']
        saved_start = data['start_date']
        saved_end = data['end_date']
        
        # Perform clustering
        df_clustered, pca, feature_importance = perform_clustering(df_filtered, num_cols_select)
        
        # Create three tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "📊 Clustering View",
            "📈 Feature Importance",
            "🎯 Player Comparison"
        ])
        
        with tab1:
            st.subheader("Player Clustering")
            fig = create_cluster_scatter(
                df_clustered,
                'Cluster',
                non_num_cols_select + num_cols_select[:5],
                highlighted_players=show_player_select if show_player_select else None,
                feature_importance_df=feature_importance
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Analysis Period:** {saved_start.strftime('%Y-%m-%d')} to {saved_end.strftime('%Y-%m-%d')} | **Players:** {len(df_clustered)}")
        
        with tab2:
            st.subheader("Most Impactful Statistics")
            fig = create_feature_importance_bar(feature_importance)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Statistics are ranked by their contribution to the PCA components used in clustering.")
        
        with tab3:
            st.subheader("Compare Players")
            
            # Get top stats for comparison
            top_stats = feature_importance.head(15)['Feature'].tolist()
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                players_to_compare = st.multiselect(
                    'Select players to compare (2-6 recommended)',
                    df_clustered['Player'].sort_values().tolist(),
                    default=df_clustered.nlargest(3, 'Points Per Game')['Player'].tolist() if 'Points Per Game' in df_clustered.columns else [],
                    key='date_range_players'
                )
            
            with col2:
                fill_checkbox = st.checkbox(
                    'Fill areas',
                    value=True,
                    help='Uncheck to see hover data more easily',
                    key='date_range_fill'
                )
            
            # Stat selection for radial chart
            stats_for_comparison = st.multiselect(
                'Select statistics to include in comparison (8-12 recommended)',
                top_stats,
                default=top_stats[:8],
                key='date_range_stats'
            )
            
            if len(players_to_compare) > 0 and len(stats_for_comparison) > 0:
                fig = create_radial_comparison(df_clustered, players_to_compare, stats_for_comparison, fill=fill_checkbox)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one player and one statistic to compare.")
        
        # NEW: Add bar charts for top features
        st.markdown("---")
        st.subheader("🏆 Top Players by Most Important Statistics")
        st.markdown("Explore which players rank highest for each key statistic identified by the clustering model.")
        
        # Get top features
        top_features = feature_importance.head(10)['Feature'].tolist()
        
        # Create tabs for each top feature
        if len(top_features) > 0:
            feature_tabs = st.tabs(top_features)
            for i, tab in enumerate(feature_tabs):
                with tab:
                    stat_name = top_features[i]
                    fig = create_top_players_bar(df_clustered, stat_name, 'Cluster')
                    st.plotly_chart(fig, use_container_width=True)


def player_comparison_view(games_df, box_scores_df):
    """Player comparison visualization view."""
    st.header('👤 Player Comparison')
    
    # Prepare data
    df = pd.merge(
        box_scores_df.loc[box_scores_df.player != 'Eddie Johnson'],
        games_df
    ).sort_values(by=['date', 'player'])
    
    df = add_game_outcomes(df, games_df)
    
    # Get player list sorted by all-around stats
    players = (
        df.groupby('player')
        .agg(all_stat=('all_stat', 'mean'))
        .sort_values('all_stat', ascending=False)
        .reset_index()['player']
        .tolist()
    )
    
    # Player and analysis type selection
    col1, col2 = st.columns(2)
    
    with col1:
        player = st.selectbox('Select a player to analyze', players)
    
    with col2:
        analysis_type = st.selectbox(
            'Analysis based on:',
            ['Games Played', 'Date Range']
        )
    
    # Configuration form
    with st.form(key='player_comparison_form'):
        st.subheader(f"Analyzing: **{player}**")
        
        if analysis_type == 'Games Played':
            games_played = df.loc[df.player == player].game_id.nunique()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                games_played_select = st.slider(
                    'Through first N games',
                    10,
                    games_played,
                    games_played
                )
            
            minutes_played = int(df.loc[df.player == player].head(games_played_select).mp.sum())
            minutes_played_half = int(minutes_played / 2)
            
            with col2:
                minutes_played_select = st.slider(
                    f'Min minutes (player: {minutes_played})',
                    0,
                    minutes_played,
                    minutes_played_half
                )
            
            with col3:
                old_players_select = st.checkbox(
                    'Include older players',
                    value=True,
                    help="Older players don't have advanced stats like Usage Rate"
                )
            
            df_filtered = df.groupby('player').head(games_played_select)
            
        else:  # Date Range
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Start date",
                    value=pd.to_datetime(df.loc[df['player'] == player].date.max()) - pd.DateOffset(months=1),
                    min_value=pd.to_datetime('1966-02-19'),
                    max_value=pd.to_datetime(df.loc[df['player'] == player].date.max())
                )
            
            with col2:
                end_date = st.date_input(
                    "End date",
                    value=pd.to_datetime(df.loc[df['player'] == player].date.max()),
                    min_value=pd.to_datetime('1966-02-19'),
                    max_value=pd.to_datetime(df.loc[df['player'] == player].date.max())
                )
            
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df_filtered = df.loc[(df.date >= start_date) & (df.date <= end_date)]
            
            player_mp = int(df_filtered.loc[df_filtered.player == player].mp.sum())
            player_mp_half = int(player_mp / 2)
            player_gp = int(df_filtered.loc[df_filtered.player == player].game_id.nunique())
            player_gp_half = int(player_gp / 2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                minutes_played_select = st.slider(
                    f'Min minutes (player: {player_mp})',
                    0,
                    player_mp,
                    player_mp_half
                )
            
            with col2:
                games_played_select = st.slider(
                    f'Min games (player: {player_gp})',
                    0,
                    player_gp,
                    player_gp_half
                )
            
            old_players_select = True
        
        # Aggregate and filter
        df_agg = aggregate_box_scores(df_filtered)
        df_agg = df_agg.loc[df_agg['gp'] >= (games_played_select * 0.75)].copy()
        df_agg = rename_columns(df_agg)
        
        if minutes_played_select:
            df_agg = df_agg.query("Minutes > @minutes_played_select")
        
        if old_players_select:
            df_agg = df_agg.loc[(df_agg['Minutes'] >= minutes_played_select)].fillna(0)
        else:
            df_agg = df_agg.loc[
                (df_agg['Minutes'] >= minutes_played_select) &
                (df_agg['Usage Rate'].notnull())
            ].fillna(0)
        
        num_cols, non_num_cols = get_num_cols(df_agg)
        
        num_cols_select = st.multiselect(
            'Select statistics for analysis',
            num_cols,
            default=num_cols[:10] if len(num_cols) > 10 else num_cols
        )
        
        non_num_cols_select = st.multiselect(
            'Select columns for hover data',
            non_num_cols,
            default=['Player'] if 'Player' in non_num_cols else []
        )
        
        submit_button = st.form_submit_button(label='🔍 Find Similar Players', type='primary')
    
    if submit_button and len(num_cols_select) > 1:
        # Perform clustering
        df_clustered, pca, feature_importance = perform_clustering(df_agg, num_cols_select)
        
        # Find closest players
        player_data = df_clustered[df_clustered['Player'] == player][['Cluster_x', 'Cluster_y']].values
        
        if len(player_data) > 0:
            df_clustered['distance'] = np.sqrt(
                (df_clustered['Cluster_x'] - player_data[0][0])**2 +
                (df_clustered['Cluster_y'] - player_data[0][1])**2
            )
            df_clustered = df_clustered.sort_values('distance')
            
            # Create three tabs for different visualizations
            tab1, tab2, tab3 = st.tabs([
                "📊 Clustering View",
                "📈 Feature Importance",
                "🎯 Player Comparison"
            ])
            
            with tab1:
                st.subheader(f"Players Similar to {player}")
                
                # Show highlighted player
                fig = create_cluster_scatter(
                    df_clustered,
                    'Cluster',
                    non_num_cols_select + num_cols_select[:5],
                    highlighted_players=[player] + df_clustered.head(10)['Player'].tolist(),
                    feature_importance_df=feature_importance
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show closest players table
                st.subheader("Most Similar Players")
                closest_df = df_clustered[df_clustered['Player'] != player].head(10)[
                    ['Player', 'Team', 'Games Played', 'Points Per Game', 'Assists Per Game', 'Rebounds Per Game']
                    if all(col in df_clustered.columns for col in ['Team', 'Games Played', 'Points Per Game', 'Assists Per Game', 'Rebounds Per Game'])
                    else ['Player', 'distance']
                ]
                st.dataframe(closest_df, use_container_width=True, hide_index=True)
            
            with tab2:
                st.subheader("Most Impactful Statistics")
                fig = create_feature_importance_bar(feature_importance)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("These statistics had the greatest influence in finding similar players.")
            
            with tab3:
                st.subheader("Compare with Similar Players")
                
                # Get top stats for comparison
                top_stats = feature_importance.head(15)['Feature'].tolist()
                
                # Default to player + top 3 similar
                default_comparison = [player] + df_clustered[df_clustered['Player'] != player].head(3)['Player'].tolist()
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    players_to_compare = st.multiselect(
                        'Select players to compare',
                        df_clustered['Player'].tolist(),
                        default=default_comparison
                    )
                
                with col2:
                    fill_checkbox = st.checkbox(
                        'Fill areas',
                        value=True,
                        help='Uncheck to see hover data more easily',
                        key='player_comp_fill'
                    )
                
                # Stat selection for radial chart
                stats_for_comparison = st.multiselect(
                    'Select statistics to include in comparison (8-12 recommended)',
                    top_stats,
                    default=top_stats[:8],
                    key='player_comp_stats'
                )
                
                if len(players_to_compare) > 0 and len(stats_for_comparison) > 0:
                    fig = create_radial_comparison(df_clustered, players_to_compare, stats_for_comparison, fill=fill_checkbox)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select at least one player and one statistic to compare.")
            
            # NEW: Add bar charts for top features
            st.markdown("---")
            st.subheader("🏆 Top Players by Most Important Statistics")
            st.markdown("Explore which players rank highest for each key statistic in the comparison set.")
            
            # Get top features
            top_features = feature_importance.head(10)['Feature'].tolist()
            
            # Create tabs for each top feature
            if len(top_features) > 0:
                feature_tabs = st.tabs(top_features)
                for i, tab in enumerate(feature_tabs):
                    with tab:
                        stat_name = top_features[i]
                        fig = create_top_players_bar(df_clustered, stat_name, 'Cluster')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Player '{player}' not found in the filtered dataset.")


# ============================================================================
# MAIN APP
# ============================================================================

def app():
    """Main application function."""
    st.set_page_config(
        page_title='NBA Visualization Pt 2 | aaroncolesmith.com',
        page_icon='🏀',
        layout='wide'
    )
    
    # Title and description
    st.title("🏀 NBA Player Statistics Visualization (Part 2)")
    st.markdown("""
    Explore NBA player performance through clustering analysis, feature importance, and visual comparisons.
    Choose your analysis type below to get started.
    """)
    
    # Load data
    with st.spinner("Loading NBA data..."):
        games_df = load_nba_games()
        box_scores_df = load_nba_box_scores()
    
    # Main navigation using tabs
    tab1, tab2, tab3 = st.tabs([
        "🎮 Single Game",
        "📅 Date Range",
        "👤 Player Comparison"
    ])
    
    with tab1:
        single_game_view(games_df, box_scores_df)
    
    with tab2:
        date_range_view(games_df, box_scores_df)
    
    with tab3:
        player_comparison_view(games_df, box_scores_df)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source:** Basketball Reference | **Updated:** Hourly")


if __name__ == "__main__":
    app()
