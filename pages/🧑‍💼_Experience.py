import streamlit as st

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )

def app():
    st.write("""
    # Work Experience

    ### [Nike](https://www.nike.com)
    #### Senior Product Manager | 2021 - Present
    - Worked within the Commercial Analytics organization to shift Nike into a more technology and data-driven approach to their Supply Chain
    - Partnered with Data Science and Machine Learning teams to build enhancmenents and customizations to the Supply Chain to improve the efficiency of the organization
    - Collaborated with Business Development stakeholders to ensure they were able to visualize and understand the Supply Chain data as well as be able to increase their efficieny and make changes based on data outputs

    ---

    ### [Openpath](https://openpath.com)
    #### Senior Product Manager | 2020-2021
    - Built and launched two new hardware products, a Video Reader and a Video Intercom Reader, which immediately become the most popular access control products on the market
    - Working with the advanced concepts team to start building new products and integration to support new use cases like integrating video feeds into the portal to provide a holistic security monitoring solutions
    - Introduced data driven processes and principles to the product team so that we start to build products and make decisions backed by data

    ---

    ### [SafeRide Health](https://saferidehealth.com)
    #### Product Manager | 2020
    - Built the technology platform to enable end-to-end patient transportation to care, supporting a number of different
    use cases supporting Health Plans, Health Networks, NEMT Providers & ridesharing solutions
    - Automated existing Operations’ processes to enable the organization to scale without having to increase team size
    - Introduced software development processes to enable a much more collaborative product / engineering culture

    ---
    ### [Olive](https://oliveai.com)
    #### Product Manager - AI & Process Automation Platform | 2017 - 2020
    - Built the software platform used to build and train our artificial intelligence solution, Olive, which allows an
    automation engineer to utilize a wide range RPA functionality and integrate with state of the art technology like CV,
    OCR and machine intelligence
    - Developed from the ground up a state of the art process mining / intelligence product to greatly reduce the time to
    analyze and document processes prior to building out automations
    - Increased automation build velocity by 500% by eliciting user feedback to implement usability enhancements

    #### Director - Data & Analytics | 2016 - 2018
    - Built a data analytics solution to analyze our data across multiple data sources to provide the entire company with
    actionable intelligence and insights to influence product and business strategy
    - Shifted the company focus to make more data-driven decisions using the metrics and tools recently created

    #### Product Manager - Patient Registration & Internal Tools | 2015 - 2016
    - Developed the next-gen platform for our flagship product with a mindset of increasing scalability and engagement
    - Collaborated cross-functionally with leadership to ensure the internal tools developed would support all necessary
    job functions to allow CrossChx to grow at the rate it needed to meet investors’ demands

    ---
    ### [Cardinal Solutions Group](http://www.cardinalsolutions.com/)
    #### Senior Consultant – Product Management | 2014 - 2015
    - Hired as a Senior Consultant to the Agile Project Services helping mentor newer employees in both Agile/Scrum methodologies as well as working as a Business Analyst and Product Owner
    - Developed the requirements and the content model for key components to enable a new version of a product to be marketed and sold as an innovative educational solution
    - Collaborated with product stakeholders to build a comprehensive product backlog and roadmap to enable to the development of critical enhancements and cutting-edge features to enable approximately $1 billion in sales
    - Managed the workload for the development team to create incremental product updates towards the roadmap

    ---
    ### [Accenture](https://www.accenture.com/us-en)
    #### Consultant / Demand Planning Team Lead | 2011 - 2014
    - Analyzed, designed, and tested new functionality for DLA’s ERP solution which included enhancements to SAP (ECC 6.0) and JDA/Manugistics solutions as well as interfaces to Business Intelligence and Reporting systems
    - Managed the Demand Planning team to ensure all resources had a sufficient amount of work assigned
    - As the JDA Demand Planning subject matter expert, provided guidance on all new features being built to assess
    the potential impact to downstream data interfaces
    - Worked with key client contacts to maintain our current relationship while also working with our delivery leads to
    explore and expand our business relationship
    """)
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=portfolio&ea=experience">',unsafe_allow_html=True)
