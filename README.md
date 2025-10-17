# cmse_830fds
Mid semester Project README File

**Why I chose my dataset:**
- I chose my datasets because I was very interested in how governments can impact the lives of their citizens. Based on this idea, I narrowed my scope down a bit to how governments (through expenditure) can impact the health of their citizens. I then set out to find a dataset with information on the health of citizens in different countries over the course of at least 10 years. Next, I went to the World Bank Group DataBank and downloaded a dataset that contained information on government expenditure for individual countries.

**What I've learned from IDA/EDA:**
- Based on the IDA and EDA that I've done so far, I have learned that my datasets span different timeframes (2000-2015 vs. 2000-2024). I have also learned that these datasets include information about the same countries (except for a few), which is why I chose them. I have learned about the distribution of the data in both datasets using histograms and boxplots. I have also examined some trends and correlations between variables using correlation matrices, scatter plots, and line plots.

**What preprocessing steps I've completed:**
- Removed unwanted or unneeded variables from both datasets
- Dropped rows that contained unwanted info
- Changed variable data types
- Renamed variables to names with a common format
- Merged datasets to produce one final dataframe
- Encoded region variable in preparation for visualization
- Imputed values for missing health expenditure data using stochastic regression
  
**What I've tried with Streamlit so far:**
- So far in Streamlit, I have used some basic plots from my EDA and explored some features like filters, pages, and tabs. 
