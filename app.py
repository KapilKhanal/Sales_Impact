# Streamlit 
import streamlit as st
from PIL import Image
import plotly
import plotly.graph_objs as go
# Python
import numpy as np
import pandas as pd
import datetime as dt
#from pandas_profiling import ProfileReport
# Dependencies
import dataIngestion as di
import config as config
import kmeans_clustering as kc
import causalImpact as cimpact
import RFM as rfm
import matplotlib.pyplot as plt


data_path = "data/Sales_df.csv"

@st.cache()
def read_file(data_path):
    original = pd.read_csv(data_path, index_col =[0], parse_dates=[config.DATE_COL])
    df = di.remove_na(original,config.COLS_WITH_NA)
    df = di.remove_negative(df,config.NEGATIVE_COL)
    return df


data = read_file(data_path)

@st.cache()
def get_rfm():
    rfm = pd.read_csv('data/rfmtable.csv')
    return rfm

PAGES = (
    "Home",
    "Description",
    "Customer Segmentation",
    "Impact Analysis"
)

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio('options',PAGES)
    if selection == 'Description':
        # upload = st.file_uploader("Choose a csv file", type="csv")
        st.markdown("This data is collected from the UCI Machine Learning repository for demonstration purposes.")
        st.markdown("The project is focused on segmenting customers and analyzing the effect of intervention such as discounts, loyality programs in these different kinds of customers")
        
        # Show Dataset
        if st.checkbox("Preview DataFrame"):
            if st.button("Head"):
                st.write(data.head())
                st.write(data.dtypes)
            if st.button("Tail"):
                st.write(data.tail())
         # Show Summary of Dataset
        if st.checkbox("Show Summary of Dataset"):
            st.write(data.describe())
    elif selection == 'Customer Segmentation':
        st.markdown("""**Cluster analysis** uses mathematical models to discover groups of similar customers based on the smallest variations among customers within each group.
                 The goal of cluster analysis in marketing is to accurately segment customers in order to achieve more effective customer marketing via personalization. A common cluster analysis method is a mathematical algorithm known as k-means cluster analysis, sometimes referred to as scientific segmentation
                 The clusters definitions change every time the clustering algorithm runs, ensuring that the groups always accurately reflect the current state of the data.""", unsafe_allow_html=True)
        now = dt.date(config.REFERENCE_YEAR ,config.REFERENCE_Month,config.REFERENCE_day)
        # Create rfm table
        rfmtable = rfm.calculate_rfm(data,config.GROUP_BY_COL,config.LIST_COL_AGG,now)
        # Turn rfm into a matrix
        st.write(rfmtable.head())
        df_normalized = di.normalise_table(rfmtable)
        # Get Best N of clusters
        st.write("If using new data and want to run a new model, check below")
        if st.checkbox("Run new model"):
            matrix = kc.get_matrix(df_normalized)
            kmeans_clusters = kc.give_num_clusters(matrix, config.MIN_CLUSTER, config.MAX_CLUSTER)
            st.pyplot(kmeans_clusters['Plot'])
            Best_N = kmeans_clusters['Best_N']
            st.write(f"Optimum number of cluster selected is: {Best_N}")
        else:
            Best_N = 4
            st.write("The default model has 4 clusters")
        rfm_with_labels = get_rfm()
        #rfm_with_labels = kc.get_df_with_labels(Best_N, df_normalized)
        st.write(rfm_with_labels['cluster'].value_counts())
        st.write(rfm_with_labels.head())
        st.pyplot(kc.plot_clusters(rfm_with_labels)['Plot'])
    elif selection == 'Impact Analysis':
        st.title("Impact Analysis")
        now = dt.date(config.REFERENCE_YEAR ,config.REFERENCE_Month,config.REFERENCE_day)
        rfmtable = rfm.calculate_rfm(data,config.GROUP_BY_COL,config.LIST_COL_AGG,now)
        df_normalized = di.normalise_table(rfmtable)
        matrix = kc.get_matrix(df_normalized)
        kmeans_clusters = kc.give_num_clusters(matrix, config.MIN_CLUSTER, config.MAX_CLUSTER)
        Best_N = kmeans_clusters['Best_N']
        st.write("Best num cluster selected is: ",Best_N)
        rfm_with_labels = kc.get_df_with_labels(Best_N, df_normalized)
        merged_df = di.join_rfm_orginial(data, rfm_with_labels, config.JOIN_ON_COL)
        selected_cluster = st.selectbox("Select Cluster to examine",range(Best_N))
        if selected_cluster == 0:
            before_ci_df = di.give_cluster_df(merged_df, 0)
            Dates = cimpact.get_st_dates(before_ci_df)
            stpre, stmax = Dates['stpre'].date(), Dates['stmax'].date()
            st.write(f'Intervention dates should be between {stpre} and {stmax}')
            st.write(f'A good post-intervention date anywhere from 1 day to 1 week after the intervention date.')
            d = st.date_input("Select intervention date", Dates['stpre'])
            e = st.date_input("Select post-intervention date", Dates['stpost'])
            intervention_date = pd.to_datetime(d)
            post_intervention_date = pd.to_datetime(e)
            ci = cimpact.causal_impact(before_ci_df, intervention_date,post_intervention_date)
            Impact = cimpact.plot_ci(ci)
            st.pyplot(Impact['Plot'])
            st.write(Impact['Report'])
        else:
            before_ci_df = di.give_cluster_df(merged_df, selected_cluster)
            Dates = cimpact.get_st_dates(before_ci_df)
            stpre, stmax = Dates['stpre'].date(), Dates['stmax'].date()
            st.write(f'Intervention dates should be between {stpre} and {stmax}')
            st.write(f'A good post-intervention date anywhere from 1 day to 1 week after the intervention date.')
            d = st.date_input("Select intervention date", Dates['stpre'])
            e = st.date_input("Select post-intervention date", Dates['stpost'])
            intervention_date = pd.to_datetime(d)
            post_intervention_date = pd.to_datetime(e)
            ci = cimpact.causal_impact(before_ci_df, intervention_date,post_intervention_date)
            Impact = cimpact.plot_ci(ci)
            st.pyplot(Impact['Plot'])
            st.write(Impact['Report'])
            
            
if __name__ == "__main__":
    main()
