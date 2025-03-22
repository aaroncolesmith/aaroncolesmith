from datetime import datetime, date
import streamlit as st
import plotly_express as px
import pandas as pd
import plotly.io as pio
pio.templates.default = "simple_white"



def app():
    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
    )
    df = pd.read_parquet('https://github.com/aaroncolesmith/data_action_network/raw/refs/heads/main/data/trank_db_merged.parquet')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%m/%d/%Y')
    df['bet_advice'] = df['bet_advice'].fillna('no advice')

    today = today = date.today().strftime('%m/%d/%Y')
    date_range = df['date'].unique().tolist()
    ix = date_range.index(today)
    date_select = st.selectbox(
        label = 'Select a date',
        options = date_range,
        index=ix
    )

    filtered_df = df.loc[df.date == date_select]
    st.write(filtered_df)



    col_list = filtered_df.columns.tolist()

    c1,c2,c3 = st.columns(3)
    x_axis = c1.selectbox(
        'X Axis',
        col_list,
        col_list.index('ttq')
        )

    y_axis = c2.selectbox(
        'Y Axis',
        col_list,
        col_list.index('spread_diff')
        )

    color_select = c3.selectbox(
        'Color',
        col_list,
        col_list.index('bet_advice')
        )
    
    hover_data = st.multiselect(
        'Hover Data',
        col_list,
        ['t_rank_line']
    )

    fig = px.scatter(
    filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_select,
        text = 'matchup',
            hover_data=hover_data,
            # width=1200,
            height=900)
    fig.update_traces(marker=dict(
        # color='lightblue',        # Fill color of the bars
        size=12,
        line=dict(color='navy',
                width=2)  # Outline color and thickness
    )
    )
    fig.update_layout(
            font=dict(
            family='Futura',  # Set font to Futura
            size=12,          # You can adjust the font size if needed
            color='black'     # Font color
        ),
        # yaxis=dict(categoryorder='total ascending'),  # Order the categories by total value,
        # width=1200,
        # title='% of Units by Operational Opportunity'
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)



if __name__ == "__main__":
    #execute
    app()