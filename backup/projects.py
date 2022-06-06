import streamlit as st

def app():
    st.write("""
    # Projects
    ### Olive - Process Mining / Process Mapping Product | September 2018 - July 2019
    **Problem:** Process discovery and process documentation were very difficult and time consuming for our team. We would go into meetings with customers and struggle to identify what were some good areas for automation and we would try to document existing processes, but interviews would only uncover a small set of the entire scope of the process.

    **Solution:** We built Pupil which was a Process Discovery & Process Mapping product. Pupil would ingest process related data like system logs and using machine learning and process mining algorithms, it would cluster processes together. We could then display those processes in a 3d scatter visualization that would allow users to group similar sets of processes together. Finally, they could output those groups to a process map which could be handed off to the automation team to build an Olive.""")
    st.image('./images/pupil.gif',caption='Pupil was used for Process Mining & Process Discovery',use_column_width=True)

    st.write("""

    ---

    ### Olive - Process Automation Platform | July 2017 - May 2018
    **Problem:** As a company, we realized that we had product market fit, but we needed to build a platform to enable our team to efficiently scale operations of building and deploying Olive.

    **Solution:** We built a process automation platform, speficially for our internal team to build Olives. Built the platform specifically for our team and for our industry so that we could build healthcare automations as quickly, resiliently & easily as possible.""")
    st.image('./images/mimic.gif',caption='The OliveBuilder platform was used to build & deploy thousands of automated workers',use_column_width=True)

    st.write("""
    ---

    ### Olive - Data Analytics Project | July 2016 - February 2017
    **Problem:** Our team at Olive was not harnessing data to maximize our team's operations. We were making decisions based on gut feel or instinct solely as relying on data as a valuable input into the decision-making process.

    **Solution:** As I had a strong data background as well as a strong understanding of our internal data schema, our COO tasked me with harnessing our data to provide intelligence to our staff to operate more efficiently. This was done by connecting multiple datasets together and working with leaders from across the company to ensure everyone was using our data to most efficiently align their team's operations.
    """)
    st.image('./images/grow.png',caption='An example of a data dashboard used to visualize company performance metrics & goals',use_column_width=True)

    st.write("""
    ---

    ### Olive - Connect - Hospital Registration System | January 2016 - July 2016
    **Problem:** We had a legacy product that was built on old technology that was very buggy. It was also built on a single-tenant architecture which meant that costs for supporting the product were growing at an unsupportable rate.

    **Solution:** Went all-in on rewriting and redesigning our existing product so that we had a much more stable future. I led the product requirements and product design while working very closely with engineering leadership to ensure we were meeting project objectives of keeping future support costs to a minimum.
    """)

    st.image('./images/queue.png',caption='Queue was our patient registration system used across 500 health systems', use_column_width=True)

    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=portfolio&ea=projects">',unsafe_allow_html=True)
