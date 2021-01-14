import streamlit as st

def app():

    st.write("""
    # Aaron Cole Smith
    Hi, I'm Aaron. I am a data-driven problem solver who believes that any problem can be solved with hard work, creativity and technology. I have worked a number of different roles, but have mostly found success working as a Product Manager and working with data.
    My most successful project was working at [Olive](https://oliveai.com) where I was an early product hire brought in to build the platform to build Olive's AI workforce.
    I'm also very active in building side projects, mainly centered around data-related apps. Feel free to check some of these out on the sidebar, like NBA Clusters where you can see how different NBA players / careers cluster based on similar stats.""")

    st.image('./images/nba_clusters.gif',caption='Preview of NBA Clusters -- feel free to check it out!',use_column_width=True)

    st.write("""If you have any questions / thoughts, feel free to reach out to me via [email](mailto:aaroncolesmith@gmail.com), [LinkedIn](https://linkedin.com/in/aaroncolesmith) or [Twitter](https://www.twitter.com/aaroncolesmith).

    """)
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=portfolio&ea=about">',unsafe_allow_html=True)
