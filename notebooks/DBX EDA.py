# Databricks notebook source
# MAGIC %md # Aim

# COMMAND ----------

# MAGIC %md # 1. Setup

# COMMAND ----------

# MAGIC %md ## 1.1 Installs

# COMMAND ----------

# MAGIC %pip install google-cloud-storage gcsfs imblearn -U scikit-learn

# COMMAND ----------

# MAGIC %md ## 1.2 Imports

# COMMAND ----------

import os
import json
import numpy as np

import pandas as pd
from google.cloud import storage

import pyspark.sql.functions as F
from pyspark.sql.functions import col, expr, udf, struct
from pyspark.sql.window import Window

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as Pipeline_imb

from xgboost import XGBClassifier

# plotting
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# COMMAND ----------

# MAGIC %md ## 1.3 Utils

# COMMAND ----------

def kill_time(mins=60):

    """
    Keeps cluster alive for 60 mins by default 
    """

    import time

    for i in range(mins):
        time.sleep(60)

    return

# COMMAND ----------

def get_dict_value(mapping):
    """
    Used to extract the value from a dictionary given a pyspark column as a key.
    Returns a udf function (currying) that can be used with the following syntax:
        df.withColumn('new_col_name', (get_dict_value(mapping_dict)('col_name')))
        
    :param mapping: Dictionary to extract mapping from
    :type mapping: dict
    :returns: udf function requiring dataframe column name as parameter
    :rtype: udf function
    """
    
    from pyspark.sql.functions import col, udf
    
    def get_mapping_val(col):
        return mapping.get(col)
    return udf(get_mapping_val)

# COMMAND ----------

def check_duplicate_ids(df):
    """
    Return df with distinct counts of 'id column'
    """
    grouped_id_df = (
        df
        .groupby("id")
        .agg(F.sum(F.lit(1)).alias("cnt"))
        .orderBy(col("cnt").desc())
    )
    return grouped_id_df

# COMMAND ----------

def unpivot_df(df, new_col_name, new_value_col_name, col_name_to_row_list = None):
    """
    Unpivots dataframe, i.e. from columns to rows
    """
    # Column names to store as rows in a column
    if col_name_to_row_list is None:
        col_name_to_row_list = df.columns

    # Create string with '"col1", col1' format
    unpivot_col_string = ', '.join([f'"{col_name}", {col_name}' for col_name in col_name_to_row_list])

    new_col_name = new_col_name
    new_value_col_name = new_value_col_name

    # Create SQL expression using strings defined above
    unpivot_expr = f'stack({len(col_name_to_row_list)}, {unpivot_col_string}) as ({new_col_name}, {new_value_col_name})'

    # Apply SQL expression to generate columns
    unpivot_df = df.select(expr(unpivot_expr))

    return unpivot_df

# COMMAND ----------

def dataset_summary(df, counts_list = ["null", "group"]):
    """
    Print a quick summary of a dataset
    """
    print(f"Shape: {df.count()}, {len(df.columns)}")
    
    output_list = []

    if "null" in counts_list:

        col_list = [*df.columns, "total_count"]

        null_count_df = (
            df
            .withColumn("total_count", F.lit(None))
            .select(
                [F.count(F.when(
                F.isnan(c) |
                col(c).isNull() |
                col(c).contains('None') |
                col(c).contains('NULL') |
                (col(c) == '' )
                , c)).alias(c) for c in col_list]
            )
        )

        unpivoted_df = unpivot_df(null_count_df, "columns", "null_count", col_name_to_row_list = null_count_df.columns)

        # row_count = null_count_df.select("total_count").collect()[0]["total_count"]

        unpivoted_df = (
            unpivoted_df
            .withColumn("null_perc", col("null_count")/F.lit(df.count())*100)
        )

        output_list.append(unpivoted_df)

        # print("Null Count:")
        # unpivoted_df.display()

    if "group" in counts_list:
    #     print("Group Counts:")

        group_counts_df_list = []
        empty_window = Window.partitionBy()

        for col_name in df.columns:
            count_col_name = f"{col_name}_count"

            group_count_df = (
                df
                .groupBy(col_name)
                .agg(F.count(F.lit(1)).alias(count_col_name))
                .withColumn("perc_total", col(count_col_name)/F.sum(count_col_name).over(empty_window)*100)
            )

            group_counts_df_list.append(group_count_df)
        
        output_list.append(group_counts_df_list)

    return output_list
        

# COMMAND ----------

#udf definition    
def list_contains(value, contains_list):
    """
    Return 1 list contains items
    """
    if value in contains_list:
        return 1
    else:
        return 0
    
def list_contains_udf(contains_list):
    """
    UDF for list above to allow operation on spark dataframes
    """
    return udf(lambda l: list_contains(l, contains_list))

# COMMAND ----------

# MAGIC %md ## 1.3 Download Data

# COMMAND ----------

# Set paths and filenames
credential_json_path = "../credentials/interview task gcs credentials.json"

filename_list = [
    "brand_affinity",
    "demographics_responses",
    "psychographics_responses",
    "question_metadata",
]

gcs_path = "gs://datascience_interview/"

# Create client
storage_client = storage.Client.from_service_account_json(credential_json_path)

# COMMAND ----------

# Download data and store in dictionary
df_dict = {}

for filename in filename_list:
    df_path = os.path.join(gcs_path, filename + ".csv")

    df = spark.createDataFrame(
        pd.read_csv(
            df_path,
            storage_options={"token":credential_json_path}
        )
    )
    df_dict[filename] = df

# COMMAND ----------

# MAGIC %md # 2. Explore individual datasets
# MAGIC - From initial idea of data, the flow should be "How do people's survey answers and affinity for Ralph Lauren indicate their affinity for Superdry
# MAGIC - How good is the join between the datasets?
# MAGIC - Will I have to remove any features due to lack of coverage?
# MAGIC - What's the best imputation method for the situation?
# MAGIC - What's the ratio between categoric variables?

# COMMAND ----------

# MAGIC %md ## 2.1 Brand Affinity
# MAGIC Straightforward information detailing preference and existing customer
# MAGIC - Only Ralph Lauren and Superdry, maybe Ralph Lauren affinity will correlate with Superdry affinity
# MAGIC - No nulls
# MAGIC - ~60% rows are Superdry-related
# MAGIC   - Do all Ralph Lauren rows have a corresponding Superdry row?
# MAGIC     - Mix of ralph only, superdry only or both
# MAGIC - Majority of id's are unique (53179/77763 = 68%)
# MAGIC   - Are these duplicates across different brands? No, all single
# MAGIC - Quit a small proportion of customers that are current Superdry customers
# MAGIC   - Can try using consideration instead or under/oversample

# COMMAND ----------

dataset_name = "brand_affinity"
brand_affinity_df = df_dict[dataset_name]
print(f"Row Count: {brand_affinity_df.count()}")
brand_affinity_df.display()

# Get summary
null_df, group_count_df_list = dataset_summary(brand_affinity_df, ["group", "null"])

# Print null counts
display(null_df)

# Print group count dfs
for df in group_count_df_list:
    display(df)

# COMMAND ----------

# Distinct values

brand_affinity_summary_df = (
    brand_affinity_df
    .groupBy()
    .agg(
        F.countDistinct("id").alias("distinct_id"),
        F.countDistinct("brand").alias("distinct_brand"),
        F.sum(F.lit(1)).alias("total_rows")
    )
    .withColumn("distinct_id_perc", col("distinct_id")/col("total_rows")*100)
)

brand_affinity_summary_df.display()

# COMMAND ----------

display(
    brand_affinity_df
    .filter(col("brand") == "Superdry")
)

# COMMAND ----------

# Check for duplicate id's within the same brand
display(
    brand_affinity_df
    .groupby("id", "brand")
    .agg(F.sum(F.lit(1)).alias("cnt"))
    # .groupby("id")
    # .agg(F.max("cnt").alias("max_brand_count"))
    .filter(col("cnt") > 1)
)

# COMMAND ----------

# Check id coverage between brands
brand_affinity_ralph_df = brand_affinity_df.filter(col("brand") == "Ralph Lauren").select(col("id").alias("r_id"))
brand_affinity_super_df = brand_affinity_df.filter(col("brand") == "Superdry").select(col("id").alias("s_id"))

brand_affinity_joint_df = (
    brand_affinity_ralph_df.alias("r")
    .join(brand_affinity_super_df.alias("s"), how="outer", on=col("s.s_id") == col("r.r_id"))
)

display(
    brand_affinity_joint_df
    .withColumn("r_only", F.when(col("r_id").isNotNull() & col("s_id").isNull(), 1))
    .withColumn("s_only", F.when(col("s_id").isNotNull() & col("r_id").isNull(), 1))
    .withColumn("both", F.when(col("s_id").isNotNull() & col("r_id").isNotNull(), 1))
    .agg(
        F.count("r_id").alias("r_id_count"),
        F.count("s_id").alias("s_id_count"),
        F.sum("r_only").alias("r_only_sum"),
        F.sum("s_only").alias("s_only_sum"),
        F.sum("both").alias("both_sum"),
        F.sum(F.lit(1)).alias("total_rows")
    )
)

# COMMAND ----------

# MAGIC %md ## 2.2 Question Metadata
# MAGIC - Category_hierarchy_1 tells me which dataset to join against
# MAGIC   - 2 nulls I should look into though
# MAGIC - Type column for categorical
# MAGIC   - Numeric is just age, can bin this for conversion to categorical
# MAGIC - Are any of these aliases not actually present in the demo and psycho datasets?
# MAGIC - When profiling, can use category hierarchy columns
# MAGIC   - 4 seems good for psycho
# MAGIC - Ignore parent_category column
# MAGIC - Cat 1 = Psycho
# MAGIC   - Origin_source = attitudes
# MAGIC     - 5 point scale in name column is agreement with statement
# MAGIC     - 0-4, ordinal
# MAGIC     - Seems the alias for these starts with attitudes_agree
# MAGIC     - Can use this to batch processs
# MAGIC     - Maybe encode these
# MAGIC     - Origin_source = attitudes, parent_cat = Audiences, Cat_1 = Psycho, Category_hierarchy_2 = Attitudes, 3 = By statement
# MAGIC   - Origin_source = issues
# MAGIC     - Issues in alias
# MAGIC     - Importance in name
# MAGIC       - Issues_importance in alias
# MAGIC       - 0-3 importance to person, ordinal
# MAGIC     - Opinion in name
# MAGIC       - issues_scale in alias
# MAGIC       - mostly 0-4 on opinion, seems ordinal
# MAGIC         - some not, e.g. Sustainability in Retail has no 2
# MAGIC       - Some responses "don't know", either part of response 2 with "neither" or added as response = 11
# MAGIC   - Origin source = personality
# MAGIC     - personality_single in alias
# MAGIC     - Start from 1, some ordinal, some not
# MAGIC
# MAGIC - Cat 1 = null
# MAGIC   - origin_souce = datum
# MAGIC   - parent 1 or 0
# MAGIC   
# MAGIC - Cat 1 = Demo
# MAGIC   - Origin_source = datum
# MAGIC   - pdl at start of alias
# MAGIC   - Has numeric age 18 - 99 individual
# MAGIC     - Can bin this
# MAGIC   - Will likely need to encode these with subject-specific methods
# MAGIC   - Child age
# MAGIC     - has option for NA, 99
# MAGIC     - starts from 0
# MAGIC     - goes to 18
# MAGIC   - Grandchild age
# MAGIC     - seems to actually be binary "have them" yes or no
# MAGIC     - But question is "I don't have them", confusing
# MAGIC   - Army
# MAGIC     - Likely have to encode as categoric
# MAGIC     - 99 NA
# MAGIC   - Seems like some of these responses are the NA binary replies rather than actual option responses, have "Not applicable" in name
# MAGIC     - have number at end of alias, e.g. 99, 96
# MAGIC     - Need to consider how to handle these, likely some redundancies
# MAGIC     - Lots can be encoded with some though
# MAGIC     - Some have yes or no binary with details of specific answer in name, i.e. "is x important?"
# MAGIC       - e.g. Dream career, price increase
# MAGIC       - Can reduce these to a single column
# MAGIC       - Maybe identify with response_name having 2 rows with yes or no
# MAGIC       - Or only having 2 responses
# MAGIC     - A few high cardinality columns, might need a way to bin these to reduce sparseness
# MAGIC       - can create binary columns for applicable or not, e.g. police and army
# MAGIC       - Then check coverage and decide whether to use or not
# MAGIC
# MAGIC - For naming features
# MAGIC   - Null
# MAGIC     - Use alias
# MAGIC   - Demo
# MAGIC     - if response not in alias: alias + response or response_name
# MAGIC     - if in alias: use alias
# MAGIC   - Psycho
# MAGIC     - alias up to second underscore + formatted hier4 + mapped hier 5 + response

# COMMAND ----------

dataset_name = "question_metadata"
question_metadata_df = df_dict[dataset_name]
display(
    question_metadata_df
    .orderBy("Category_hierarchy_1", "origin_source", "name", "response")
    .select(
        "origin_source",
        "Category_hierarchy_1",
        "alias",
        "name",
        "response",
        "response_name",
        "description",
        "category",
        "Category_hierarchy_2",
        "Category_hierarchy_3",
        "Category_hierarchy_4",
        "Category_hierarchy_5",
    )
)

# Get summary
null_df, group_count_df_list = dataset_summary(question_metadata_df, ["group", "null"])

# Print null counts
display(null_df)

# Print group count dfs
for df in group_count_df_list:
    display(df)

# COMMAND ----------

# MAGIC %md ## 2.3 Demographics Responses
# MAGIC - Varying amounts of nulls up to almost 100%
# MAGIC - Some columns contain multiple values, some binary
# MAGIC   - Need to confirm with metadata, some may have just been NA for most people or were not answered
# MAGIC - Mix of ordinal and nominal
# MAGIC - No duplicate id's
# MAGIC - Can bin age and child age

# COMMAND ----------

# Overview
dataset_name = "demographics_responses"
demographics_responses_df = df_dict[dataset_name]
demographics_responses_df.display()

# Get summary
null_df = dataset_summary(demographics_responses_df, ["null"])[0]
null_df.display()

# COMMAND ----------

# Check for duplicate ids
display(check_duplicate_ids(demographics_responses_df))

# COMMAND ----------

# MAGIC %md ## 2.4 Psychographics Responses
# MAGIC - No duplicate id's
# MAGIC - Up to 50% nulls
# MAGIC - Similar to demographics, some ordinal and some nominal categorical variables
# MAGIC - Will rename columns for convenience

# COMMAND ----------

dataset_name = "psychographics_responses"
psychographics_responses_df = df_dict[dataset_name]
psychographics_responses_df.display()
display(dataset_summary(psychographics_responses_df, ["null"])[0])

# COMMAND ----------

display(check_duplicate_ids(psychographics_responses_df))

# COMMAND ----------

# MAGIC %md # 3. Preprocessing Data
# MAGIC
# MAGIC - Split affinity by brand
# MAGIC - Outer join on id's
# MAGIC - Filter out rows that don't have superdry current_customer
# MAGIC - Ralph lauren missing some ids, 54%

# COMMAND ----------

# MAGIC %md ## 3.1 Join data

# COMMAND ----------

brand_affinity_df.display()
psychographics_responses_df.display()
demographics_responses_df.display()
question_metadata_df.display()

# COMMAND ----------

# Left join on id from superdry, filter to s_current is not null so we have necessary information on the target variable

brand_affinity_ralph_df = (
    brand_affinity_df
    .filter(col("brand") == "Ralph Lauren")
    .select("id", col("Current_customer").alias("rl_current"), col("Consideration").alias("rl_consideration"))
)
brand_affinity_super_df = (
    brand_affinity_df
    .filter(col("brand") == "Superdry")
    .select("id", col("Current_customer").alias("s_current"), col("Consideration").alias("s_consideration"))
)

joint_df = ((
            brand_affinity_super_df
            .withColumn("s_id", col("id"))
            .drop("id")
        ).alias("s")
    .join((
            demographics_responses_df
            .withColumn("d_id", col("id"))
            .drop("id")
        ).alias("d"), how="outer", on = col("d.d_id") == col("s.s_id")
    )
    .join((
            brand_affinity_ralph_df
            .withColumn("r_id", col("id"))
            .drop("id")
        ).alias("r"), how="outer", on = col("r.r_id") == col("s.s_id")
    )
    .join((
            psychographics_responses_df
            .withColumn("p_id", col("id"))
            .drop("id")
        ).alias("p"), how="outer", on = col("p.p_id") == col("s.s_id")
    )
)

joint_df = joint_df.filter(col("s_current").isNotNull())

joint_df.display()

display(
    joint_df
    .withColumn("r_only", F.when(col("r_id").isNotNull() & col("s_id").isNull(), 1))
    .withColumn("s_only", F.when(col("s_id").isNotNull() & col("r_id").isNull(), 1))
    .withColumn("both", F.when(col("s_id").isNotNull() & col("r_id").isNotNull(), 1))
    .agg(
        F.count("r_id").alias("r_id_count"),
        F.count("d_id").alias("d_id_count"),
        F.count("p_id").alias("p_id_count"),
        # F.sum("r_only").alias("r_only_sum"),
        # F.sum("s_only").alias("s_only_sum"),
        # F.sum("both").alias("both_sum"),
        F.sum(F.lit(1)).alias("total_rows")
    )
    .withColumn("r_not_null_perc", col("r_id_count")/col("total_rows"))
)

# joint_df.display()

# COMMAND ----------

# MAGIC %md ## 3.2 Rename Psychographic columns

# COMMAND ----------

# Map a rank over unique hier 5 questions for psychographic responses, keep others the same
alias_window = Window.partitionBy("partition_col").orderBy("Category_hierarchy_5")

psycho_mapped_question_metadata_df = (
    question_metadata_df
    .withColumn("partition_col", F.concat_ws("", col("Parent_Category"), col("Category_hierarchy_1"), col("Category_hierarchy_2"), col("Category_hierarchy_3"), col("Category_hierarchy_4")))
    .withColumn("psycho_rank", F.rank().over(alias_window))
    .withColumn(
        "new_alias",
        F.when((col("Category_hierarchy_1") == "Psychographics") & (col("alias").contains("personality_single") == False), F.concat_ws("", F.regexp_extract(col("alias"), "^([^_]*_){2}", 0), F.regexp_replace(F.lower(col("Category_hierarchy_4")), "[\s*&*\-*]", "_"), F.lit("_"), col("psycho_rank")))
        .otherwise(col("alias"))
    )
)

psycho_mapping_df = (
    psycho_mapped_question_metadata_df
    .select("Category_hierarchy_1", "alias", "new_alias")
    .distinct()
)

psycho_mapped_question_metadata_df.display()

psycho_mapping_df.display()

# COMMAND ----------

# Create mapping dict
psychographics_alias_mapping_dict = {}

for i in psycho_mapping_df.filter(col("Category_hierarchy_1") == "Psychographics").collect():
    psychographics_alias_mapping_dict[i["alias"]] = i["new_alias"]

# Rename columns
new_cols = []
for item in joint_df.columns:
    if item in list(psychographics_alias_mapping_dict.keys()):
        new_cols.append(psychographics_alias_mapping_dict[item])
    else:
        new_cols.append(item)

mapped_joint_df = (
    joint_df
    .toDF(*new_cols)
)

mapped_joint_df.display()

# COMMAND ----------

# MAGIC %md ## 3.3 Deal with Nulls

# COMMAND ----------

col_list = [*mapped_joint_df.columns]

null_count_df = (
    mapped_joint_df
    .select(
        [F.count(F.when(
        F.isnan(c) |
        col(c).isNull() |
        col(c).contains('None') |
        col(c).contains('NULL') |
        (col(c) == '' )
        , c)).alias(c) for c in col_list]
    )
)

unpivoted_df = unpivot_df(null_count_df, "columns", "null_count", col_name_to_row_list = null_count_df.columns)

unpivoted_df = (
    unpivoted_df
    .withColumn("null_perc", col("null_count")/F.lit(mapped_joint_df.count())*100)
)

display(
    unpivoted_df
)

# COMMAND ----------

# Drop columns with > 60% nulls
high_null_df = (
    unpivoted_df
    .filter(col("null_perc") > 60)
)

high_null_df.display()

high_null_col_list = [i["columns"] for i in high_null_df.collect()]

null_drop_joint_df = mapped_joint_df.drop(*high_null_col_list)

print(len(null_drop_joint_df.columns))
print(len(joint_df.columns))

null_drop_joint_df.display()

# COMMAND ----------

# MAGIC %md ## 3.4 Flag ordinal or nominal

# COMMAND ----------

# display(
#     psycho_mapped_question_metadata_df
#     .filter(col("Category_hierarchy_1") == "Psychographics")
#     .orderBy("name", "response")
# )

encoding_flagged_question_metadata_df = (
    psycho_mapped_question_metadata_df
    .withColumn(
        "encoding_flag",
        F.when(
            ( # Flag psychographics
                (col("Category_hierarchy_1") == "Psychographics") 
                & (
                    (col("alias").contains("attitudes_agree"))
                    | (col("alias").contains("issues_importance"))
                    | (col("name") == ("Type of reader"))
                )
            )
            | ( # Flag demographics
                (col("Category_hierarchy_1") == "Demographics")
                & (
                    (col("alias") == "pdlc_age")
                    | (col("alias").contains("pdl_child_age"))
                )
            )
            ,
            "ordinal"
        )
        .otherwise(F.lit("nominal"))
    )
)

encoding_flagged_question_metadata_df.display()

# COMMAND ----------

encoding_flagged_question_metadata_df.select("name").distinct().orderBy("name").display()

# COMMAND ----------

# MAGIC %md ## 3.5 Correlation Matrix
# MAGIC - All columns have relatively low correlation with s_current other than consideration
# MAGIC - Checked correlation for consideration but also low so will stick with s_current

# COMMAND ----------

null_drop_joint_df_pd = null_drop_joint_df.toPandas()

df_corr = null_drop_joint_df_pd.corr()

df_corr = df_corr.where(np.tril(np.ones(df_corr.shape),k=0).astype(bool))

df_corr

# COMMAND ----------

fig = px.bar(df_corr.reset_index().sort_values(by="s_current", ascending=False), x="index", y="s_current")
fig['layout'].update(height = 1000, width = 1500)
fig.show()

fig = px.bar(df_corr.reset_index().sort_values(by="s_consideration", ascending=False), x="index", y="s_consideration")
fig['layout'].update(height = 1000, width = 1500)
fig.show()

# COMMAND ----------

fig = px.imshow(df_corr, text_auto=True)
fig['layout'].update(height = 1500, width = 1500)
fig.show()



# COMMAND ----------

# MAGIC %md ## 3.6 Group low proportion values
# MAGIC
# MAGIC Country
# MAGIC - Very high cardinality in country, so will look at group counts and either:
# MAGIC   - Encode high count and leave the rest as "other"
# MAGIC   - Group by continentUpon closer inspection
# MAGIC - 88% are UK so will change to UK and other
# MAGIC
# MAGIC Others
# MAGIC - Not as severe so will simply group any that have below 1% proportion into "other"

# COMMAND ----------

# MAGIC %md ### Country

# COMMAND ----------

empty_window = Window.partitionBy()

display(
    demographics_responses_df
    .groupby("pdl_country_of_birth")
    .agg(F.sum(F.lit(1)).alias("cnt"))
    .alias("d")
    .join((
            question_metadata_df
            .filter(col("alias") == "pdl_country_of_birth")
            .select("response", "response_name")
        ).alias("q"), how="left", on=col("q.response") == col("d.pdl_country_of_birth") 
    )
    .withColumn("perc_rows", col("cnt")/F.sum("cnt").over(empty_window))
    .orderBy(col("cnt").desc())
)

# COMMAND ----------

region_grouped_joint_df = (
    null_drop_joint_df
    .withColumn("pdl_country_of_birth_uk", F.when(col("pdl_country_of_birth") == 1, 1).when(col("pdl_country_of_birth") > 1, 0).otherwise(F.lit(None)))
    .drop("pdl_country_of_birth")
)

region_grouped_joint_df.display()

# COMMAND ----------

# MAGIC %md ### Other Columns

# COMMAND ----------

# Get nominal columns
nominal_cols = [i["new_alias"] for i in encoding_flagged_question_metadata_df.filter(col("encoding_flag") == "nominal").select("new_alias").distinct().collect()]
nominal_cols.remove("pdl_country_of_birth")

group_col_dict = {}

# Get values to group for each column
empty_window = Window.partitionBy()

for col_name in nominal_cols:
    if col_name in region_grouped_joint_df.columns:
        grouped_df = (
            region_grouped_joint_df
            .groupby(col_name)
            .agg(F.sum(F.lit(1)).alias("cnt"))
            .withColumn("perc_rows", col("cnt")/F.sum("cnt").over(empty_window))
        )

        grouped_df = grouped_df.filter(col("perc_rows") < 0.1) #< 0.01)
        col_values_to_group = [i[col_name] for i in grouped_df.collect()]

        group_col_dict[col_name] = col_values_to_group

group_col_dict

# COMMAND ----------

# Group < 1% values as 1000
grouped_counts_df = region_grouped_joint_df

for col_name, value_list in group_col_dict.items():
    grouped_counts_df = (
        grouped_counts_df
        .withColumn(
            col_name,
            F.when(list_contains_udf(value_list)(col(col_name)) == 1, 1000).otherwise(col(col_name))
        )
    )

# COMMAND ----------

# MAGIC %md ## 3.7 Random Forest Feature Importance

# COMMAND ----------


# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification

from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# COMMAND ----------

# Undersample, fill, impute, encode

# Defining the encoding variables
ordinal_columns = [i["new_alias"] for i in encoding_flagged_question_metadata_df.filter(col("encoding_flag") == "ordinal").select("new_alias").collect() if i["new_alias"] in list(x_train.columns)]
nominal_columns = [i["new_alias"] for i in encoding_flagged_question_metadata_df.filter(col("encoding_flag") == "nominal").select("new_alias").collect() if i["new_alias"] in list(x_train.columns)]
nominal_columns.append(["s_consideration", "rl_consideration", "rl_current"])
all_columns = [i["new_alias"] for i in encoding_flagged_question_metadata_df.select("new_alias").collect() if i["new_alias"] in list(x_train.columns)]

# Calling the encoder classes
fill_imputer = SimpleImputer(strategy="constant", fill_value=0)
knn_imputer = KNNImputer()
oh_encoder = OneHotEncoder(drop="if_binary", handle_unknown="infrequent_if_exist")

# Make column transformer
column_transform = make_column_transformer(
    # (fill_imputer, nominal_columns),
    # (knn_imputer, ordinal_columns),
    (oh_encoder, all_columns),
    remainder="passthrough"
    )

# Make undersampler
undersampler = RandomUnderSampler()

# Make classifier
xgb_classifier = XGBClassifier(objective="binary:logistic")

# Make pipeline
pipeline = Pipeline([("transformers", column_transform), ("xgb_classifier", xgb_classifier)])

# COMMAND ----------

# MAGIC %md # 4. Create model

# COMMAND ----------

# MAGIC %md ## 4.1 Split data

# COMMAND ----------

# Drop unnecessary columns
input_df = (
    grouped_counts_df
    .withColumn("target", col("s_current"))
    .drop("s_id", "d_id", "r_id", "p_id", "weights", "s_current")
)

# COMMAND ----------

# input_df.write.option('compression','snappy').mode('overwrite').parquet("s3a://prd-matchesfashion-datalake-scratch/ds-customer/brandon/hm_prop_input_df")
input_df = spark.read.parquet("s3a://prd-matchesfashion-datalake-scratch/ds-customer/brandon/hm_prop_input_df")
# # df_.write.option('compression','snappy').mode('overwrite').parquet('s3://path/to/file.parquet')

# COMMAND ----------

# Split and check counts
train_df, test_df = input_df.randomSplit(weights=[0.7, 0.3], seed=42)

print(train_df.count())
print(test_df.count())

train_df_pd = train_df.toPandas()
test_df_pd = test_df.toPandas()

x_train = train_df_pd.drop("target", axis = 1)
y_train = train_df_pd["target"]

x_test = test_df_pd.drop("target", axis = 1)
y_test = test_df_pd["target"]

# COMMAND ----------

# input_df.display()

# COMMAND ----------

# MAGIC %md ## 4.2 Set up pipeline
# MAGIC - Will undersample to balance classes
# MAGIC - Impute for ordinal, fill for nominal
# MAGIC - Will use KNN imputer as other survey answers are correlated

# COMMAND ----------

# Undersample, fill, impute, encode

# Defining the encoding variables
ordinal_columns = [i["new_alias"] for i in encoding_flagged_question_metadata_df.filter(col("encoding_flag") == "ordinal").select("new_alias").distinct().collect() if i["new_alias"] in list(x_train.columns)]
nominal_columns = [i["new_alias"] for i in encoding_flagged_question_metadata_df.filter(col("encoding_flag") == "nominal").select("new_alias").distinct().collect() if i["new_alias"] in list(x_train.columns)]
nominal_columns.append(["s_consideration", "rl_consideration", "rl_current"])
all_columns = [i["new_alias"] for i in encoding_flagged_question_metadata_df.select("new_alias").distinct().collect() if i["new_alias"] in list(x_train.columns)]

ordinal_columns

# Fill nominal nulls with 0
x_train[nominal_columns] = x_train[nominal_columns].fillna(0)
x_test[nominal_columns] = x_test[nominal_columns].fillna(0)

# # Calling the encoder classes
# fill_imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
# fill_pipeline = Pipeline(["fill_imputer", fill_imputer])

knn_imputer = KNNImputer()
oh_encoder = OneHotEncoder(drop="if_binary", handle_unknown="infrequent_if_exist")

# Make column transformer
column_transform = make_column_transformer(
    # (fill_imputer, nominal_columns),
    # (fill_pipeline, nominal_columns),
    (knn_imputer, ordinal_columns),
    (oh_encoder, all_columns),
    remainder="passthrough"
)

# Make undersampler
undersampler = RandomUnderSampler()

# Make classifier
xgb_classifier = XGBClassifier(objective="binary:logistic")

# Make pipeline
# pipeline = Pipeline([("undersampler", undersampler), ("transformers", column_transform), ("xgb_classifier", xgb_classifier)])

pipeline = Pipeline_imb([("transformers", column_transform), ("undersampler", undersampler), ("xgb_classifier", xgb_classifier)])

# COMMAND ----------

# Find best parameters

# param_grid = {
#     "xgb_classifier__learning_rate":[0.01, 0.3, 0.5],
#     "xgb_classifier__n_estimators": [1,2],#range(10, 200, 20),
#     "xgb_classifier__max_depth": [2,3],#range(2, 10 , 1),
# }

# # Intantiate GridSearchCV
# grid_search_cv = GridSearchCV(pipeline, param_grid, scoring = "f1", cv = 10)

# # Fit
# model = grid_search_cv.fit(x_train, y_train)

# print(model.best_params_)

# COMMAND ----------

# MAGIC %md # 5. Analyse results

# COMMAND ----------

# Predict on test data
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
predict_proba = pipeline.predict_proba(x_test)

output_df_pd = test_df_pd.copy(deep=True)

output_df_pd["y_pred"] = y_pred
# output_df_pd["predict_proba"] = predict_proba

# COMMAND ----------

predict_proba
# xgb_classifier.classes_

# COMMAND ----------

# cm
y_pred.sum()

# COMMAND ----------

# Get classification report
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True)
# fig['layout'].update(height = 1500, width = 1500)
fig.update_layout(xaxis_title="True", yaxis_title="Pred")
fig.show()


# COMMAND ----------

# MAGIC %md # 6. Profiling

# COMMAND ----------

# xgb_classifier.feature_importances_

# x_train_transformed = pipeline[:-2].transform(x_train)

# Access column names after transformation
# oh_encoder.feature_names_in_ndarray
# oh_encoder.get_feature_names_out()
list(column_transform.transformers_[0][1].get_feature_names_out())

# df = pd.DataFrame.sparse.from_spmatrix(x_train_transformed, columns=['A', 'B', 'C'])

# COMMAND ----------

empty_window = Window.partitionBy()

profiling_df = spark.createDataFrame(output_df_pd)
profiling_df.display()

thresholds = profiling_df.approxQuantile("predict_proba",[0.25, 0.75])

threshold_df = (
    profiling_df
    .withColumn("proba_flag", F.when(col("predict_proba") <= thresholds[0], F.lit("low")).when(col("predict_proba") >= thresholds[1], F.lit("high")))
)

threshold_df.display()

# profiling




# COMMAND ----------

# MAGIC %md # Misc

# COMMAND ----------

kill_time(120)

# COMMAND ----------


