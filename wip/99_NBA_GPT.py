import streamlit as st
import pandas as pd
import requests
import numpy as np

@st.cache_data(ttl=3600)
def load_nba_games():
    df = pd.read_parquet('https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_games.parquet', engine='pyarrow')

    return df

@st.cache_data(ttl=3600)
def load_nba_box_scores():
    df = pd.read_parquet('https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_box_scores.parquet', engine='pyarrow')
    df = df.loc[df['mp'] > 0]
    
    df['gmsc'] = pd.to_numeric(df['gmsc'])
    df['bpm'] = pd.to_numeric(df['bpm'])
    df['ortg'] = pd.to_numeric(df['ortg'])
    df['drtg'] = pd.to_numeric(df['drtg'])
    df['missed_shots'] = (df['fga'].fillna(0) - df['fg'].fillna(0))+(df['fta'].fillna(0) - df['ft'].fillna(0))
    df['all_stat'] = df['pts'] + df['trb'] + df['ast']

    df["darko_lite"] = (
    0.25 * df["bpm"] +
    0.20 * (df["ortg"] - df["drtg"]) +
    0.15 * df["usg_pct"] +
    0.15 * df["ts_pct"].fillna(0) +  # Fill missing TS% with 0 to avoid NaNs
    0.10 * df["ast_pct"] +
    0.10 * df["trb_pct"] +
    0.05 * df["stl_pct"] +
    0.05 * df["blk_pct"] -
    0.05 * df["tov_pct"].fillna(0)
    )

    return df


def combine_nba_data(d1,d2):

    d = pd.merge(d2.loc[d2.player!='Eddie Johnson'],d1).sort_values(by=['date', 'player']).copy()
    
    d['playoff_game'] = np.where(d['game_type'] == 'Playoffs', 1, 0)
    d['win'] = np.select(
        [
            (d['team'] == d['home_team']) & (d['home_score'] > d['visitor_score']),
            (d['team'] == d['home_team']) & (d['home_score'] < d['visitor_score']),
            (d['team'] != d['home_team']) & (d['home_score'] < d['visitor_score']),
            (d['team'] != d['home_team']) & (d['home_score'] > d['visitor_score'])
        ],
        [
            1,  
            0,  
            1,  
            0   
        ],
    )
    d['loss'] = np.where(d['win'] == 0, 1, 0)
    d['playoff_win'] = np.where(d['playoff_game'] == 1, d['win'], 0)
    d['playoff_loss'] = np.where(d['playoff_game'] == 1, 1-d['win'], 0)


    return d


def ask_ollama(prompt, model="llama3"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def generate_code_from_question(question, df):
    schema_description = "\n".join([f"- '{col}'" for col in df.columns])
    prompt = f"""
You are a Python data assistant. The user has a pandas DataFrame called `df` with these columns:
{schema_description}

Write clean Python code (no print statements, no imports) that answers the user's question:
"{question}"

Store the result in a variable called `result`.

Be sure to look into the data to ensure the question is being answered correctly. The user may not type the exact name of a player. For example, they may ask about Lebron instead of LeBron James. Use the data to find the correct player name.

Only return the code. No explanation or markdown.
"""
    return ask_ollama(prompt)


def generate_natural_response(question, result, model="llama3"):
    prompt = f"""
You are a helpful NBA data assistant.

The user asked: "{question}"

The result of the query is: {result}

Respond with a single, clear sentence that presents the result in a conversational tone. Do not restate the full question unless necessary. Keep it short and direct. But be sure to include the result number so that the user knows how many of the questioned statitstic.
"""
    return ask_ollama(prompt, model=model)




def app():

    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    

    st.title("NBA GPT")

    d1 = load_nba_games()
    d2 = load_nba_box_scores()

    df=combine_nba_data(d1,d2)


    st.write(df.head(10))

    question = st.text_input("Ask a question about NBA stats")

    if question:
        code = generate_code_from_question(question, df)
        st.code(code, language="python")
        try:
            local_vars = {"df": df}
            exec(code, {}, local_vars)
            result = local_vars.get("result")
            if result is not None:
                st.write("Result:", result)
                natural_response = generate_natural_response(question, result)
                st.markdown(f"**Answer:** {natural_response}")
            else:
                st.warning("No result returned. Check the generated code.")
        except Exception as e:
            st.error(f"Error running code: {e}")



if __name__ == "__main__":
    #execute
    app()
