#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv("Udacity_AZDIAS_Subset.csv", sep=";")

# Load in the feature summary file.
feat_info = pd.read_csv("AZDIAS_Feature_Summary.csv", sep=";")


# In[3]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
print(azdias.shape, feat_info.shape)


# In[4]:


azdias.head(11)


# In[5]:


feat_info.head()


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[6]:


# Identify missing or unknown data values and convert them to NaNs.
feat_info["missing_or_unknown_list"] = feat_info["missing_or_unknown"].apply(lambda x: x.replace("[","").replace("]","").split(","))
feat_info_list = feat_info[["attribute","missing_or_unknown_list"]].to_dict("split")["data"]
feat_info_dict = {a[0]: a[1:] for a in feat_info_list}
for a in feat_info_dict:
    try:
        feat_info_dict[a] = [int(n) for n in feat_info_dict[a][0]]
    except:
        feat_info_dict[a] = feat_info_dict[a][0]
    azdias[a] = azdias[a].replace(to_replace = feat_info_dict[a], value=np.nan)


# In[7]:


azdias.head(11)


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[8]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.
missing = azdias.isnull().sum().sort_values(ascending=False)
print(missing)


# In[9]:


print(missing[missing == 0].shape[0])
print(missing[missing > 0].shape[0])


# In[10]:


# Investigate patterns in the amount of missing data in each column.
missing.hist(bins=20,color="blue")
plt.xlabel("Missing values")
plt.ylabel("Number of columns")


# In[11]:


missing.sort_values(ascending=True).plot.barh(figsize=(10,30),color="blue")


# In[12]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
missing_col_drop = missing[missing > 200000].index.tolist()
azdias.drop(missing_col_drop,axis=1,inplace=True)
print(azdias.shape)


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# 
# 24 columns did not have any missing values, while 61 had at least one. The histogram showed that there were 6 columns with more than 200,000 missing values. With the horizontal barplot I identified these colums: TITEL_KZ, AGER_TYP, KK_KUNDENTYP, KBA05_BAUMAX, GEBURTSJAHR and ALTER_HH. These columns were deleted.

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[13]:


# How much data is missing in each row of the dataset?
missing_row = azdias.isnull().sum(axis=1)
missing_row.hist(bins=20,color="blue")
plt.xlabel("Missing values")
plt.ylabel("Number of rows")


# In[14]:


print(missing_row[missing_row > 0].shape[0])
print(missing_row[missing_row == 0].shape[0])


# In[15]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
azdias_low = azdias[missing_row <= 10].copy()
azdias_high = azdias[missing_row > 10].copy()
print(azdias.shape[0], azdias_low.shape[0], azdias_high.shape[0])


# In[16]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
def compare_chart(col):
    ax1, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
    ax1 = plt.subplot(1,2,1)
    ax1.set_title("Dataset with low number of missing values")
    sns.countplot(azdias_low[col], ax=ax1)
    ax2 = plt.subplot(1,2,2)
    ax2.set_title("Dataset with high number of missing values")
    sns.countplot(azdias_high[col], ax=ax2)

for c in ["HEALTH_TYP", "REGIOTYP", "RELAT_AB", "ANREDE_KZ", "ANZ_PERSONEN"]:
    compare_chart(c)
    plt.show()


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# If we look at the missing data in rows we can tell that there are 268,012 observations with at least 1 missing value and 623,209 without any. I decided to define the threshold as 10, so observations with 10 or less missing values are regarded to have a low number of missing values (n=780,153), while those having more than 10 are regarded to have many (n=111,068).
# 
# Regarding the comparison between individuals (with a high and low level of missing values), based on the data dictionary i selected three personal-level (ANREDE KZ - gender, ANZ_PERSONEN - "number of adults in household", HEALTH_TYP - health status), one regional-level (REGIOTYP - neighborhood type) and one community-level (RELAT_AB - "share of unemployment relative to county") features. The distributions look similar for the two datasets in ANREDE KZ, ANZ_PERSONEN, REGIOTYP, RELAT_AB, but look a bit different for HEALTH_TYP. "Jaunty hedonists" (or value 3) has a higher relative frequency in the dataset with low number of missing values than in the other one.
# 
# So based on these selected features the two datasets can be regarded as rather similar to each other.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[17]:


# How many features are there of each data type?
feat_info_filtered = feat_info[~feat_info["attribute"].isin(missing_col_drop)]
print(feat_info_filtered.iloc[:,2].value_counts().sort_values(ascending=False))


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[18]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
binary = []
multi = []
for a in feat_info_filtered.loc[feat_info_filtered["type"] == "categorical","attribute"]:
    if azdias_low[a].nunique(dropna=True) == 2:
        binary.append(a)
    elif azdias_low[a].nunique(dropna=True) > 2:
        multi.append(a)
print(binary)
print(multi)
for c in binary:
    print(azdias_low[c].unique())


# In[19]:


# Re-encode categorical variable(s) to be kept in the analysis.
for df in [azdias, azdias_low, azdias_high]:
    df["OST_WEST_KZ"] = df["OST_WEST_KZ"].map({"W": 0, "O": 1})
    df.drop(multi, axis=1, inplace=True)
    print(df.shape)


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# First of all I searched for the 21 categorical features. 4 of them had numeric values, so they were kept, 1 had non-numeric values, so was recoded and the 16 multi-categorical features were deleted.
# 
# All these steps were done for the azdias_low (to be used in later analysis), the azdias (original) and the azdias_high (other divided) datasets.

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[20]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
print(azdias_low["PRAEGENDE_JUGENDJAHRE"].unique())
PRAEGENDE_JUGENDJAHRE_dict = {1:[40,0],2:[40,1],3:[50,0],4:[50,1],5:[60,0],6:[60,1],                              7:[60,1],8:[70,0],9:[70,1],10:[80,0],11:[80,1],12:[80,0],13:[80,1],14:[90,0],15:[90,1]}

for df in [azdias, azdias_low, azdias_high]:
    df["decade"] = df["PRAEGENDE_JUGENDJAHRE"].apply(lambda x: np.nan if np.isnan(x) else PRAEGENDE_JUGENDJAHRE_dict[x][0])
    df["movement"] = df["PRAEGENDE_JUGENDJAHRE"].apply(lambda x: np.nan if np.isnan(x) else PRAEGENDE_JUGENDJAHRE_dict[x][1])
    df.drop("PRAEGENDE_JUGENDJAHRE", axis=1, inplace=True)
    print(df.shape)


# In[21]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
for df in [azdias, azdias_low, azdias_high]:
    df["wealth"] = df["CAMEO_INTL_2015"].str[0]
    df["life_stage"] = df["CAMEO_INTL_2015"].str[1]
    df.drop("CAMEO_INTL_2015", axis=1, inplace=True)
    print(df.shape)


# In[22]:


for c in ["decade", "movement", "wealth", "life_stage"]:
    print(azdias_low[c].value_counts())


# In[23]:


mixed = [a for a in feat_info_filtered.loc[feat_info_filtered["type"] == "mixed","attribute"]]
print(mixed)
mixed.remove("PRAEGENDE_JUGENDJAHRE")
mixed.remove("CAMEO_INTL_2015")
for df in [azdias, azdias_low, azdias_high]:
    df.drop(mixed, axis=1, inplace=True)
    print(df.shape)


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# From PRAEGENDE_JUGENDJAHRE I created two new features: decade (interval scale) and movement (0-mainstream, 1-avantgarde). Then I dropped the original feature. 
# 
# From CAMEO_INTL_2015 I created two new features: wealth (higher score being less wealth) and life stage. Then I dropped the original feature.
# 
# I dropped the other four mixed_value features.
# 
# All these steps were done for the azdias_low (to be used in later analysis), the azdias (original) and the azdias_high (other divided) datasets.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[24]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)


# In[25]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[26]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    feat_info["missing_or_unknown_list"] = feat_info["missing_or_unknown"].apply(lambda x: x.replace("[","").replace("]","").split(","))
    feat_info_list = feat_info[["attribute","missing_or_unknown_list"]].to_dict("split")["data"]
    feat_info_dict = {k[0]: k[1:] for k in feat_info_list}
    for a in feat_info_dict:
        try:
            feat_info_dict[a] = [int(n) for n in feat_info_dict[a][0]]
        except:
            feat_info_dict[a] = feat_info_dict[a][0]
        df[a] = df[a].replace(to_replace = feat_info_dict[a], value=np.nan)
    
    # remove selected columns and rows, ...
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_col_drop = ['TITEL_KZ', 'AGER_TYP', 'KK_KUNDENTYP', 'KBA05_BAUMAX', 'GEBURTSJAHR', 'ALTER_HH']
    df = df.drop(missing_col_drop,axis=1).copy()
    missing_row = df.isnull().sum(axis=1)
    df = df[missing_row <= 10]    
    
    # select, re-encode, and engineer column values.
    binary = []
    multi = []
    feat_info_filtered = feat_info[~feat_info["attribute"].isin(missing_col_drop)]
    for a in feat_info_filtered.loc[feat_info_filtered["type"] == "categorical","attribute"]:
        if df[a].nunique(dropna=True) == 2:
            binary.append(a)
        elif df[a].nunique(dropna=True) > 2:
            multi.append(a)
    df["OST_WEST_KZ"] = df["OST_WEST_KZ"].map({"W": 0, "O": 1})
    df.drop(multi, axis=1, inplace=True)
    PRAEGENDE_JUGENDJAHRE_dict = {1:[40,0],2:[40,1],3:[50,0],4:[50,1],5:[60,0],6:[60,1],                              7:[60,1],8:[70,0],9:[70,1],10:[80,0],11:[80,1],12:[80,0],13:[80,1],14:[90,0],15:[90,1]}
    df["decade"] = df["PRAEGENDE_JUGENDJAHRE"].apply(lambda x: np.nan if np.isnan(x) else PRAEGENDE_JUGENDJAHRE_dict[x][0])
    df["movement"] = df["PRAEGENDE_JUGENDJAHRE"].apply(lambda x: np.nan if np.isnan(x) else PRAEGENDE_JUGENDJAHRE_dict[x][1])
    df.drop("PRAEGENDE_JUGENDJAHRE", axis=1, inplace=True)
    df["wealth"] = df["CAMEO_INTL_2015"].str[0]
    df["life_stage"] = df["CAMEO_INTL_2015"].str[1]
    df.drop("CAMEO_INTL_2015", axis=1, inplace=True)
    mixed = [a for a in feat_info_filtered.loc[feat_info_filtered["type"] == "mixed","attribute"]]
    mixed.remove("PRAEGENDE_JUGENDJAHRE")
    mixed.remove("CAMEO_INTL_2015")
    df.drop(mixed, axis=1, inplace=True)
    print(df.shape)
    
    # Return the cleaned dataframe.
    return df
    


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[27]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
imputer = Imputer()
azdias_clean = imputer.fit_transform(azdias_low)


# In[28]:


# Apply feature scaling to the general population demographics data.
scaler = StandardScaler()
azdias_analysis = scaler.fit_transform(azdias_clean)


# In[29]:


azdias_final = pd.DataFrame(azdias_analysis, columns=azdias_low.columns)
print(azdias_final.shape)


# ### Discussion 2.1: Apply Feature Scaling
# 
# I used the standard scaler as suggested and imputed the data with the mean along features (the default setting). I was considering deleting missing data, but since I kept only rows with a low number of missing values, I thought that imputing data would not mess up my results.

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[30]:


# Apply PCA to the data.
pca = PCA(random_state=42)
X_pca = pca.fit_transform(azdias_final)


# In[31]:


# Investigate the variance accounted for by each principal component.
print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_.sum())
explained_variance = pd.Series(pca.explained_variance_ratio_)
explained_variance_sum = explained_variance.cumsum()
plt.bar(x=explained_variance.index, height=explained_variance, label="Variance")
plt.plot(explained_variance_sum, label="Cumulative variance")
plt.legend(loc='upper left', borderpad=0.3)


# In[32]:


# Re-apply PCA to the data while selecting for number of components to retain.
print(explained_variance_sum[explained_variance_sum > 0.8].index)
n_components = explained_variance_sum[explained_variance_sum > 0.8].index[0] + 1
print(n_components)
pca = PCA(n_components, random_state=42)
X_pca = pca.fit_transform(azdias_final)
X_azdias_pca = pd.DataFrame(X_pca)
print(X_azdias_pca.shape)
print(pca.explained_variance_ratio_.sum())


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# I decided to keep 22 principal components as they explain more than 80% of the variance.

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[33]:


components_azdias_pca = pd.DataFrame(pca.components_)
components_azdias_pca.columns = azdias_final.columns
print(components_azdias_pca)
def show_association(df, component):
    component = df.iloc[component]
    print(component.sort_values(ascending=False))


# In[34]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
show_association(components_azdias_pca, 0)


# In[35]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.
show_association(components_azdias_pca, 1)


# In[36]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.
show_association(components_azdias_pca, 2)


# The top 3 components are shown with the top 5 negative and positive features for each. In the parantheses I copied the name of the feature from the dictionary and I also made a comment about the scale of those original features because that can help in the interpretation.
# 
# For the first component the biggest positive weights are with: PLZ8_ANTG3 ("Number of 6-10 family houses in the PLZ8 region", higher values mean higher number of these families), PLZ8_ANTG4 ("Number of 6-10 family houses in the PLZ8 region", higher values mean higher number of these families), wealth (higher values mean lower wealth), HH_EINKOMMEN_SCORE ("Estimated household net income", higher values mean lower income) and ORTSGR_KLS9 ("Size of community", higher values mean higher size). The biggest negative weights are with: MOBI_REGIO ("Movement patterns", higher values mean lower movement), PLZ8_ANTG1 ("Number of 1-2 family houses in the PLZ8 region", higher values mean higher number of these families), KBA05_ANTG1 ("Number of 1-2 family houses in the microcell", higher values mean higher number of these families), FINANZ_MINIMALIST ("Financial typology - low financial interest", higher values mean lower), KBA05_GBZ ("Number of buildings in the microcell", higher values mean more buildings). The results are somewhat confusing. Based on the features this component represents population density and wealth. The positive weights suggest that those scoring high on this component are less wealthy (wealth and income are measured on a reversed scale originally, those scoring high on those features have less income and wealth) and live in more densely populated areas. The negative weights tell a slightly different story (e.g. high number of buildings in the microcell having a negative sign).
# 
# For the second component the biggest positive weights are with: ALTERSKATEGORIE_GROB ("Estimated age based on given name analysis", higher values mean higher estimated age), FINANZ_VORSORGER ("Financial typology - be brepared", higher values mean lower), SEMIO_ERL ("Personality typology - event-oriented", higher values mean lower affinity), SEMIO_LUST ("Personality typology - sensual-minded", higher values mean lower affinity) and RETOURTYP_BK_S ("Return type", higher values mean less returns). The biggest negative weights are with: SEMIO_REL ("Personality typology - religious", higher values mean lower affinity), decade (the decade when was born, higher values mean younger generations), FINANZ_SPARER ("Financial typology - money-saver", higher values mean lower), SEMIO_TRADV ("Personality typology - tradional-minded", higher values mean lower affinity), SEMIO_PFLICHT ("Personality typology - dutiful", higher values mean lower affinity). Based on the features this component represents age and personality. We have to take into account the scaling of the features, based on that those scoring high on this component are older, more traditional-minded and religious, less sensual-minded and make less returns.
# 
# For the third component the biggest positive weights are with: SEMIO_VERT ("Personality typology - dreamful", higher values mean lower affinity), SEMIO_SOZ ("Personality typology - socially-minded", higher values mean lower affinity), SEMIO_FAM ("Personality typology - family-minded", higher values mean lower affinity), SEMIO_KULT ("Personality typology - cultural-minded", higher values mean lower affinity) and FINANZ_MINIMALIST ("Financial typology - low financial interest", higher values mean lower). The biggest negative weights are with: ANREDE_KZ ("gender", 1 means male, 2 means female), SEMIO_KAEM ("Personality typology - combative attitude", higher values mean lower affinity), SEMIO_DOM ("Personality typology - dominant-minded", higher values mean lower affinity), SEMIO_KRIT ("Personality typology - critical-minded", higher values mean lower affinity), SEMIO_RAT ("Personality typology - rational", higher values mean lower affinity). Based on the features this component represents mainly personality. We have to take into account the scaling of the features, based on that those scoring high on this component are more rational, critical, dominant, combative and less socially-or family-minded.

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[37]:


# Over a number of different cluster counts...
kmeans_scores = []
for i in range(1,16):
    kmeans = KMeans(i, random_state=42)
    # run k-means clustering on the data and...
    kmeans.fit(X_azdias_pca) 
    # compute the average within-cluster distances.
    score = np.abs(kmeans.score(X_azdias_pca))
    print(score)
    kmeans_scores.append(score)    


# In[38]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.
k_means_scores_list = pd.Series(kmeans_scores)
plt.plot(k_means_scores_list,color="blue",marker="o")
plt.xticks(np.arange(0, 15, 1))


# In[39]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
n_clusters = 9
kmeans = KMeans(n_clusters, random_state=42)
kmeans.fit(X_azdias_pca)
azdias_pred = kmeans.predict(X_azdias_pca)
print(pd.Series(azdias_pred).value_counts().sort_values(ascending=False))


# ### Discussion 3.1: Apply Clustering to General Population
# 
# Based on the elbow method I decided to segment the population into 9 clusters.

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[40]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')


# In[41]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.
customers_low = clean_data(customers)
print(customers.shape, customers_low.shape)
customers_clean = imputer.transform(customers_low)
customers_analysis = scaler.transform(customers_clean)
customers_final = pd.DataFrame(customers_analysis)
X_pca = pca.transform(customers_final)
X_customers_pca = pd.DataFrame(X_pca)
print(X_customers_pca.shape)
customers_pred = kmeans.predict(X_customers_pca)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[42]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.
azdias_clusters = pd.Series(azdias_pred).value_counts().sort_index()
customers_clusters = pd.Series(customers_pred).value_counts().sort_index()
clusters = pd.concat([azdias_clusters,customers_clusters], axis=1).reset_index()
clusters.drop("index",axis=1,inplace=True)
clusters.columns = ["general","customers"]
clusters["general_ratio"] = clusters["general"] / clusters["general"].sum(axis=0)
clusters["customers_ratio"] = clusters["customers"] / clusters["customers"].sum(axis=0)
clusters["diff"] = clusters["general_ratio"] - clusters["customers_ratio"]
print(clusters["diff"].idxmax(axis=1))
print(clusters["diff"].idxmin(axis=1))
print(clusters)
sns.barplot(x=clusters.index, y="general_ratio",data=clusters)
plt.show()
sns.barplot(x=clusters.index, y="customers_ratio",data=clusters)
plt.show()


# In[43]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?
print(X_azdias_pca.shape, azdias_pred.shape, X_customers_pca.shape, customers_pred.shape)
azdias_full = pd.concat([X_azdias_pca, pd.Series(azdias_pred)],axis=1).reset_index()
customers_full = pd.concat([X_customers_pca, pd.Series(customers_pred)],axis=1).reset_index()
azdias_agg = azdias_full.groupby(azdias_full.iloc[:,23]).agg("mean")
customers_agg = customers_full.groupby(customers_full.iloc[:,23]).agg("mean")
print(azdias_agg)
azdias_4 = azdias_agg.loc[4,:]
customers_4 = customers_agg.loc[4,:]
print(azdias_4, customers_4)


# In[44]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?
azdias_2 = azdias_agg.loc[2,:]
customers_2 = customers_agg.loc[2,:]
print(azdias_2, customers_2)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# Based on the comparison, if we have 9 clusters, those from the fifth cluster are overrepresented in the customer data compared to the general, while those in the third cluster are underrepresented.
# 
# If we look at the mean value of the components for those in the fith cluster, we can see that they score high on the third component and low on the first component. We already discussed these components. Scoring low on the first component means that someone is wealthier and lives in less populated areas, those scoring high on the third component are more rational, critical, dominant, combative and less socially-or family-minded. The company should target this group as they are overrepresented in the customer data.
# 
# If we look at the mean value of the components for those in the third cluster, we can see that they are the opposite of the third cluster as they score low on the third component and high on the first component. Scoring high on the first component means that someone is less wealthy and lives in more densely populated areas, while those scoring low on the third component are less rational, critical, dominant, combative and more socially-or family-minded. The company should not regard this group as its target group as they are underrepresented in the customer data.

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




