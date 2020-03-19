# Streamlit 
import streamlit as st
import plotly
import plotly.graph_objs as go
# Python
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
from pandas_profiling import ProfileReport
import time
# Dependencies
from src.data import config as config
from src.data import dataIngestion as di
from src.features import kmeans_clustering as kc
from src.features import causalImpact as cimpact
from src.features.report_generator import generate_html_report
from src.features import RFM as rfm


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

data_path = "data/interim/Sales_df.csv"

@st.cache()
def read_file(data_path):
    original = pd.read_csv(data_path, index_col =[0], parse_dates=[config.DATE_COL])
    df = di.remove_na(original,config.COLS_WITH_NA)
    df = di.remove_negative(df,config.NEGATIVE_COL)
    return df


data = read_file(data_path)

@st.cache()
def get_rfm():
    rfm = pd.read_csv('data/interim/rfmtable.csv')
    return rfm

now = dt.date(config.REFERENCE_YEAR ,config.REFERENCE_Month,config.REFERENCE_day)

PAGES = (
    "Description",
    "Customer Segmentation",
    "Impact Analysis"
)

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio('options',PAGES)
    report = st.sidebar.button('Generate segmentation report')
    if report:
        generate_html_report('reports/sales_impact_report.ipynb')
    if selection == 'Description':
        st.markdown("### This data set is from the UCI machine learning repository for the demonstration of this project.")
        st.markdown("### It is transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.")           
        st.markdown('### The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.')            
        st.markdown("### Our project focuses on the segmentation of customers and analyzing the effect of intervention such as discounts, loyality programs in different customer groups.")
        # Show Dataset
        if st.checkbox("Preview DataFrame"):
            if st.button("Head"):
                st.write(data.head())
            if st.button("Tail"):
                st.write(data.tail())
        # Show Summary of Dataset
        if st.checkbox("Show Summary of Dataset"):
            st.write(data.dtypes)
            st.write(data.describe())
        st.write('The profile report generates an explaratory data analysis of the dataframe.')
        st.warning('Run it only once.')
        if st.checkbox('Generate Profiling Report'):
            st.info('Generating report .....')
            profile = ProfileReport(data, minimal=True)
            profile.to_file(output_file="reports/df_report.html")
    elif selection == 'Customer Segmentation':
        st.markdown("""
                 **Cluster analysis** uses mathematical models to discover groups of similar customers based on the smallest variations among customers within each group.
                 The goal of cluster analysis in marketing is to accurately segment customers in order to achieve more effective customer marketing via personalization. 
                 A common cluster analysis method is a mathematical algorithm known as k-means cluster analysis, sometimes referred to as scientific segmentation
                 The clusters definitions change every time the clustering algorithm runs, ensuring that the groups always accurately reflect the current state of the data.
                    """)
        now = dt.date(config.REFERENCE_YEAR ,config.REFERENCE_Month,config.REFERENCE_day)
        # Create rfm table
        rfmtable = rfm.calculate_rfm(data,config.GROUP_BY_COL,config.LIST_COL_AGG,now)
        df_normalized = di.normalise_table(rfmtable)
        st.info("If you want to run a new model, check the box below")
        # Get Best N of clusters in new model
        if st.checkbox("Run new model"):
            matrix = kc.get_matrix(df_normalized)
            kmeans_clusters = kc.give_num_clusters(matrix, config.MIN_CLUSTER, config.MAX_CLUSTER)
            st.pyplot(kmeans_clusters['Plot'])
            Best_N = kmeans_clusters['Best_N']
            st.write(f"Optimum number of cluster selected is: {Best_N}")
            rfm_with_labels = kc.get_df_with_labels(Best_N, df_normalized)
        else:
            Best_N = 4
            st.write("The default model selects 4 clusters.")
            rfm_with_labels = get_rfm()
        st.write('Cluster count for each customer group.')
        st.write(rfm_with_labels['cluster'].value_counts())
        st.pyplot(kc.plot_clusters(rfm_with_labels)['Plot'])
    elif selection == 'Impact Analysis':
        st.title("Impact Analysis")
        st.markdown("""It is an approach to estimating the causal effect of a designed intervention on a time series. 
                    In our case, how can the introduction of a discount during the holidays affect the total sale of a customer group? 
                    Causal impact uses a structural Bayesian time-series model to estimate how the response metric (sales) might have evolved after the intervention (discount) if the intervention had not occurred.""")
        # now = dt.date(config.REFERENCE_YEAR ,config.REFERENCE_Month,config.REFERENCE_day)
        # rfmtable = rfm.calculate_rfm(data,config.GROUP_BY_COL,config.LIST_COL_AGG,now)
        # df_normalized = di.normalise_table(rfmtable)
        # matrix = kc.get_matrix(df_normalized)
        # kmeans_clusters = kc.give_num_clusters(matrix, config.MIN_CLUSTER, config.MAX_CLUSTER)
        # Best_N = kmeans_clusters['Best_N']
        # st.write("Best num cluster selected is: ",Best_N)
        # rfm_with_labels = kc.get_df_with_labels(Best_N, df_normalized)
        Best_N = 4
        rfm_with_labels = get_rfm()
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
