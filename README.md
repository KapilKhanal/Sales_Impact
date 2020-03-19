# Project Structure
```bash
├── __init__.py
├── sales_dashboard.py <- Streamlit dashboard.
├── data 
│   ├── __init__.py
│   ├── interim <- Intermediate data that has been transformed.
│   │   ├── Sales_df.csv
│   │   └── rfmtable.csv
│   ├── processed <- The final, canonical data sets for modeling.
│   │   ├── Joined_df.csv
│   └── raw <- The original, immutable data dump.
│       ├── CustomerTable.csv
│       ├── ItemsTable.csv
│       ├── TransactionsTable.csv
│       ├── Online_Retail.xlsx
│       ├── SalesDatabase.db
├── database 
│   ├── __init__.py
│   ├── creating_tables.py
│   ├── join_write.py
│   └── queries.py
├── notebooks <- Jupyter notebooks for experiments.
│   ├── EDA.ipynb
│   ├── RFM.ipynb
│   ├── Retail.ipynb
│   └── salesImpactResearch.ipynb
├── reports <- Generated HTML analysis.
│   ├── __init__.py
│   ├── df_report.html
│   ├── sales_impact_report.html
│   ├── sales_impact_report.ipynb
│   └── template.ipynb
├── requirement.txt
└── src <- Source code for use in this project.
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── config.py
    │   ├── dataIngestion.py
    │   └── main.py
    └── features
        ├── __init__.py
        ├── RFM.py
        ├── causalImpact.py
        ├── kmeans_clustering.py
        └── report_generator.py
```

# Sales_Impact
This project looks at how can the introduction of a discount during the holidays affect the total sale of customer groups within a timeframe of a year. The statistical techniques used are:

**RFM analysis** (recency, frenquency, monetary) to analyse customer behavior by examining their transaction history such as

    - how recently a customer has purchased (recency)
    - how often they purchase (frequency)
    - how much the customer spends (monetary)
    
RFM helps us identify customers who are more likely to respond to promotions, 
    
**K-means** to segment customers into various categories.

**Causal impact analysis** to study the impact of discounts within customer groups.





