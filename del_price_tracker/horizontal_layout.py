import pandas as pd
import streamlit as st

st.markdown('## Title')
st.sidebar.markdown('## This is the sidebar')


#st.grid.[element_name](element_properties, element_order, row_num, width, height)
st.grid.markdown('First row and first element of the grid', 0, 0)
st.grid.markdown('First row and second element of the grid', 1, 0)
st.grid.markdown('First row and third element of the grid', 2, 0)
st.grid.markdown('Second row and first element of the grid', 0, 1)
st.grid.markdown('Second row and second element of the grid', 1, 1)
st.grid.markdown('Second row and third element of the grid', 2, 1, width=500, height=500)

#this probably complicates the st.grid solution
#are 0 & 2 plotly_chart or grid parameters?
st.grid.plotly_chart(fig,0,2)



#With st.grid initiates the Table
#Each st.cell is meant to include an element
#st.cell(col, row, width, height)
with st.grid():
    with st.cell(col=0, row=1):
        st.plotly_chart(rolling_avg_fig)
    with st.cell(1, 0):
        st.write(bar_chart_fig)
    with st.cell(0,1,width=800):
        st.write(scatter_plot_fig)
    with st.cell(1,1,width=200):
        st.write(dashboard_description())

st.markdown('# This is my title')
#This represents the 2nd row (below title) and first element
st.plotly_chart(fig1, row=1, col=0)
st.plotly_chart(fig2, row=1, col=1)
#Next row down, with a customized width per element
st.plotly_chart(fig3, row=2, col=0, width=800)
st.markdown(text, row=2, col=1, width=400)



st.grid(
    (st.write('First row and first element of the grid'),
    st.write('First row and second element of the grid')),
    (st.write('Second row and first element of the grid'),
    st.write('Second row and second element of the grid'))
)

st.layout((st.plotly_chart(fig),width=1000))

with st.row(400, 600, 800):
    st.plotly_chart(fig1)
    st.plotly_chart(fig1)
    st.plotly_chart(fig1)

#First row, each element is equally sized
#st.row(st.cell(element, width, height))
st.row(
    st.cell(st.plotly_chart(fig1)),
    st.cell(st.plotly_chart(fig2))
)
#Second row, 1st element to take be 800px wide and my 2nd element 400 px wide
st.row(
    st.cell(st.plotly_chart(fig3), width=800),
    st.cell(description(), width=400)
)



with st.row():
    st.plotly_chart(fig)
    st.plotly_chart(fig2)


elements = (['First row and first element of the grid', height = 200, width = 100, row = 0])

st.grid(elements)
