import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from scipy import stats

# Load the datasets train, test and the additional datasets
app_train = pd.read_csv("./data/application_train.csv")
app_test = pd.read_csv("./data/application_test.csv")
prev_app = pd.read_csv("./data/previous_application.csv")
installments = pd.read_csv("./data/installments_payments.csv")
pos = pd.read_csv("./data/POS_CASH_balance.csv")
bureau = pd.read_csv("./data/bureau.csv")
bureau_bal = pd.read_csv("./data/bureau_balance.csv")
credit_card = pd.read_csv("./data/credit_card_balance.csv")

def transform_column(df, column_name):
    # Apply the transformation function to the selected column
    if column_name in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[column_name] = df[column_name].apply(lambda x: 1 if x == 'Y' else 0)
    elif column_name == 'CODE_GENDER':
        df[column_name] = df[column_name].apply(lambda x: np.nan if x == 'XNA' else x)

# Apply the transformation function to both train and test sets
transform_column(app_train, 'FLAG_OWN_CAR')
transform_column(app_test, 'FLAG_OWN_CAR')
transform_column(app_train, 'FLAG_OWN_REALTY')
transform_column(app_test, 'FLAG_OWN_REALTY')
transform_column(app_train, 'CODE_GENDER')
transform_column(app_test, 'CODE_GENDER')

def convert_dtype(df, target_dtype):
    # Select columns based on their current dtype
    columns_to_convert = df.select_dtypes(include=[target_dtype]).columns
    # Convert selected columns to the target dtype
    df[columns_to_convert] = df[columns_to_convert].astype(target_dtype)

# Define dtypes to convert to
dtypes_to_convert = {'float64': 'float32', 'int64': 'int32'}

# Convert dtypes for both train and test datasets
for dtype, new_dtype in dtypes_to_convert.items():
    convert_dtype(app_train, new_dtype)
    convert_dtype(app_test, new_dtype)

# impute the too big values with np.nan
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].apply(lambda x: np.nan if x > 350000 else x)
app_test['DAYS_EMPLOYED'] = app_test['DAYS_EMPLOYED'].apply(lambda x: np.nan if x > 350000 else x)
app_train['OWN_CAR_AGE'] = app_train['OWN_CAR_AGE'].apply(lambda x: np.nan if x == 64 or x == 65 else x)
app_test['OWN_CAR_AGE'] = app_test['OWN_CAR_AGE'].apply(lambda x: np.nan if x == 64 or x == 65 else x)

# define the bins for the age column
bin_edges = [0,21,26,36,46,56,66,100]

# Define the labels for the bins
bin_labels = ['0-20', '20-25', '26-35', '36-45', '46-55', '56-65', '65+']

# Create a new column with the age bins efficiently
age_bins_train = pd.cut(app_train['DAYS_BIRTH']/-365, bins=bin_edges, labels=bin_labels)
age_bins_test = pd.cut(app_test['DAYS_BIRTH']/-365, bins=bin_edges, labels=bin_labels)

combined_age_bins = pd.concat([age_bins_train, age_bins_test], axis=1)

# Assign the combined AGE column to both app_train and app_test
app_train['AGE'] = combined_age_bins.iloc[:, 0]
app_test['AGE'] = combined_age_bins.iloc[:, 0]


# Calculate the debt to income ratio
app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']  
app_test['AMT_ANNUITY'] / app_test['AMT_CREDIT']

# Calculate the credit to income ratio
app_train['CREDIT_INCOME_PERCENT'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
app_test['CREDIT_INCOME_PERCENT'] = app_test['AMT_CREDIT'] / app_test['AMT_INCOME_TOTAL']

# Calculate the annuity to income ratio
app_train['ANNUITY_INCOME_PERCENT'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
app_test['ANNUITY_INCOME_PERCENT'] = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL']

# Calculate the annuity percentage
annuity_percent_train = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL'] * 100
annuity_percent_test = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL'] * 100

# Calculate the number of days employed to the number of days of birth
app_train['DAYS_EMPLOYED_PERCENT'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
app_test['DAYS_EMPLOYED_PERCENT'] = app_test['DAYS_EMPLOYED'] / app_test['DAYS_BIRTH']

# Calculate the debt to income ratio
dti_train = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
dti_test = app_test['AMT_CREDIT'] / app_test['AMT_INCOME_TOTAL']

## Efficiently add calculated columns
app_train['DTI'] = pd.concat([dti_train], axis=1).iloc[:, 0]
app_test['DTI'] = pd.concat([dti_test], axis=1).iloc[:, 0]

app_train['ANNUITY_PERCENT'] = pd.concat([annuity_percent_train], axis=1).iloc[:, 0]
app_test['ANNUITY_PERCENT'] = pd.concat([annuity_percent_test], axis=1).iloc[:, 0]

app_train['DAYS_EMPLOYED_PERCENT'] = app_train['DAYS_EMPLOYED_PERCENT'].replace([np.inf, -np.inf], np.nan)
app_test['DAYS_EMPLOYED_PERCENT'] = app_test['DAYS_EMPLOYED_PERCENT'].replace([np.inf, -np.inf], np.nan)

# additional feature engineering
app_train['CREDIT_TERM'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
app_test['CREDIT_TERM'] = app_test['AMT_ANNUITY'] / app_test['AMT_CREDIT']

app_train['EXT_SOURCE_MEAN'] = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
app_test['EXT_SOURCE_MEAN'] = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

app_train['EXT_SOURCE_STD'] = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
app_test['EXT_SOURCE_STD'] = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)

app_train['EXT_SOURCE_STD'] = app_train['EXT_SOURCE_STD'].replace([np.inf, -np.inf], np.nan)
app_test['EXT_SOURCE_STD'] = app_test['EXT_SOURCE_STD'].replace([np.inf, -np.inf], np.nan)

app_train['EXT_SOURCE_STD'] = app_train['EXT_SOURCE_STD'].fillna(app_train['EXT_SOURCE_STD'].mean())
app_test['EXT_SOURCE_STD'] = app_test['EXT_SOURCE_STD'].fillna(app_test['EXT_SOURCE_STD'].mean())

app_train['EXT_SOURCE_MEAN'] = app_train['EXT_SOURCE_MEAN'].fillna(app_train['EXT_SOURCE_MEAN'].mean())
app_test['EXT_SOURCE_MEAN'] = app_test['EXT_SOURCE_MEAN'].fillna(app_test['EXT_SOURCE_MEAN'].mean())


# We will group the education types into 3 categories: High, Mid and Low
low = ['Lower secondary']
high = ['Academic degree','Higher education']
app_train['NAME_EDUCATION_TYPE'] = app_train['NAME_EDUCATION_TYPE'].apply(
    lambda x: 'High' if x in high else ('Low' if x in low else 'Mid'))
app_test['NAME_EDUCATION_TYPE'] = app_test['NAME_EDUCATION_TYPE'].apply(
    lambda x: 'High' if x in high else ('Low' if x in low else 'Mid'))

# We will group the education types into 3 categories: High, Mid and Low
high =['Unemployed','Maternity leave']
low =['Pension', 'State servant', 'Commercial associate']
vlow = ['Student','Businessman']

app_train['NAME_INCOME_TYPE'] = app_train['NAME_INCOME_TYPE'].apply(
    lambda x: 'Not working' if x in high else ('State related' if x in low else
                                              ('Low defaults' if x in vlow else x)) )

app_test['NAME_INCOME_TYPE'] = app_test['NAME_INCOME_TYPE'].apply(
    lambda x: 'Not working' if x in high else ('State related' if x in low else
                                              ('Low defaults' if x in vlow else x)) )

high = ['Laborers','Drivers','Low-skill laborers','Waiters/barmen staff','Security staff','Cooking staff']
medium = ['Realty agents','Sales staff', 'Cleaning staff']

app_train['OCCUPATION_TYPE'] = app_train['OCCUPATION_TYPE'].apply(
    lambda x: 'High risk' if x in high else ('Medium risk' if x in medium else 'Low risk'))

app_test['OCCUPATION_TYPE'] = app_test['OCCUPATION_TYPE'].apply(
    lambda x: 'High risk' if x in high else ('Medium risk' if x in medium else 'Low risk'))

high = ['Civil marriage', 'Single / not married']
medium = ['Separated','Married']

app_train['NAME_FAMILY_STATUS'] = app_train['NAME_FAMILY_STATUS'].apply(
    lambda x: 'High risk' if x in high else ('Medium risk' if x in medium else 'Low risk'))
app_test['NAME_FAMILY_STATUS'] = app_test['NAME_FAMILY_STATUS'].apply(
    lambda x: 'High risk' if x in high else ('Medium risk' if x in medium else 'Low risk'))

app_train['TARGET'] = app_train['TARGET']
group_0 = app_train[app_train['TARGET'] == 0]['DAYS_EMPLOYED']
group_1 = app_train[app_train['TARGET'] == 1]['DAYS_EMPLOYED']

group_0 = group_0.dropna()
group_1 = group_1.dropna()

t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)

app_train = app_train.drop(columns='TARGET')

# Prepare lists to hold the new column values
employed_mean_list = []
id_mean_list = []
reg_mean_list = []
phone_mean_list = []

# Calculate means and populate lists
for _, row in app_train.iterrows():
    employed_mean = 1 if row['DAYS_EMPLOYED'] > group_1.mean() else 0
    id_mean = 1 if row['DAYS_ID_PUBLISH'] > group_1.mean() else 0
    reg_mean = 1 if row['DAYS_REGISTRATION'] > group_1.mean() else 0
    phone_mean = 1 if row['DAYS_LAST_PHONE_CHANGE'] > group_1.mean() else 0
    
    employed_mean_list.append(employed_mean)
    id_mean_list.append(id_mean)
    reg_mean_list.append(reg_mean)
    phone_mean_list.append(phone_mean)

# Create DataFrames for training set
employed_mean_df = pd.DataFrame({'EMPLOYED_MEAN': employed_mean_list})
id_mean_df = pd.DataFrame({'ID_MEAN': id_mean_list})
reg_mean_df = pd.DataFrame({'REG_MEAN': reg_mean_list})
phone_mean_df = pd.DataFrame({'PHONE_MEAN': phone_mean_list})

# Concatenate all new columns together
new_columns_train = pd.concat([employed_mean_df, id_mean_df, reg_mean_df, phone_mean_df], axis=1)

# Add these new columns to the original DataFrame
app_train = pd.concat([app_train, new_columns_train], axis=1)

# Repeating the process for the test set
employed_mean_list = []
id_mean_list = []
reg_mean_list = []
phone_mean_list = []

# Calculate means and populate lists
for _, row in app_test.iterrows():
    employed_mean = 1 if row['DAYS_EMPLOYED'] > group_1.mean() else 0
    id_mean = 1 if row['DAYS_ID_PUBLISH'] > group_1.mean() else 0
    reg_mean = 1 if row['DAYS_REGISTRATION'] > group_1.mean() else 0
    phone_mean = 1 if row['DAYS_LAST_PHONE_CHANGE'] > group_1.mean() else 0
    
    employed_mean_list.append(employed_mean)
    id_mean_list.append(id_mean)
    reg_mean_list.append(reg_mean)
    phone_mean_list.append(phone_mean)

# Create DataFrames for test set
employed_mean_df = pd.DataFrame({'EMPLOYED_MEAN': employed_mean_list})
id_mean_df = pd.DataFrame({'ID_MEAN': id_mean_list})
reg_mean_df = pd.DataFrame({'REG_MEAN': reg_mean_list})
phone_mean_df = pd.DataFrame({'PHONE_MEAN': phone_mean_list})

# Concatenate all new columns together
new_columns_test = pd.concat([employed_mean_df, id_mean_df, reg_mean_df, phone_mean_df], axis=1)

# Add these new columns to the original DataFrame
app_test = pd.concat([app_test, new_columns_test], axis=1)

app_train['STATUS_CHANGE'] = app_train['PHONE_MEAN'] + app_train['REG_MEAN'] +\
app_train['EMPLOYED_MEAN'] + app_train['ID_MEAN'] 

app_test['STATUS_CHANGE'] = app_test['PHONE_MEAN'] + app_test['REG_MEAN'] +\
app_test['EMPLOYED_MEAN'] + app_test['ID_MEAN']

# We will calculate the number of active loans, the sum of credit prolongations, the number of past loans, 
# the average days overdue, the total credit limit, and the average annuity
agg_bureau = bureau.groupby('SK_ID_CURR').agg(
    num_active_loans=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
    sum_credit_prolong=('CNT_CREDIT_PROLONG', 'sum'),
    num_past_loans = ('CREDIT_ACTIVE','count'),
    avg_days_overdue = ('CREDIT_DAY_OVERDUE','mean'),
    total_credit_limit = ('AMT_CREDIT_SUM_LIMIT','sum'),
    avg_annuity = ('AMT_ANNUITY','mean')
)

# Drop the columns used to create the new columns
agg_bureau.reset_index(inplace=True)

# We will calculate the frequency of late payments, the sum of days overdue, the average installment duration,
# and the maximum installments left
agg_pos = pos.groupby('SK_ID_CURR').agg(
    late_payment_freq = ('SK_DPD', lambda x: (x > 30).sum()),
    DPD_sum = ('SK_DPD','sum'),
    avg_installment_duration = ('CNT_INSTALMENT','mean'),
    max_installment_left = ('CNT_INSTALMENT_FUTURE','max'),
)

agg_pos.reset_index(inplace=True)

# We will calculate the average balance, the percentage of ATM and POS withdrawals, the credit usage
agg_cc = credit_card.groupby('SK_ID_CURR').agg(
    avg_balance = ('AMT_BALANCE','mean'),
    atm_drawings=('AMT_DRAWINGS_ATM_CURRENT', 'sum'),
    pos_drawings =('AMT_DRAWINGS_POS_CURRENT', 'sum'),
    all_drawings = ('AMT_DRAWINGS_CURRENT','sum'),
    credit_limit=('AMT_CREDIT_LIMIT_ACTUAL','sum')
)

# We will calculate the percentage of ATM and POS withdrawals
agg_cc['atm_drawings'] = agg_cc['atm_drawings'] / agg_cc['all_drawings']+0.001 *100
agg_cc['pos_drawings'] = agg_cc['pos_drawings'] / agg_cc['all_drawings']+0.001 *100
agg_cc['credit_usage'] = agg_cc['all_drawings'] / agg_cc['credit_limit']+0.001 *100

agg_cc.reset_index(inplace=True)

# We will calculate the sum of the credit, the minimum credit requested, the down payment, the maximum days to decision
prev_app['NAME_CLIENT_TYPE']= prev_app['NAME_CLIENT_TYPE'].apply(
    lambda x: np.nan if x == 'XNA' else x)
prev_app['NAME_YIELD_GROUP'] = prev_app['NAME_YIELD_GROUP'].apply(
    lambda x: np.nan if x == 'XNA' else x)

# We will calculate the sum of the credit, the minimum credit requested, the down payment, the maximum days to decision
agg_pa = prev_app.groupby('SK_ID_CURR').agg(
    sum_credit_prev = ('AMT_CREDIT','sum'),
    app_min_credit = ('AMT_APPLICATION','sum'),
    down_pay = ('AMT_DOWN_PAYMENT','sum'),
    days_decision = ('DAYS_DECISION','max'),
    req_insurance = ('NFLAG_INSURED_ON_APPROVAL','sum')
)
# We will calculate the percentage of the down payment
agg_pa['app_min_credit'] = agg_pa['app_min_credit'] - agg_pa['sum_credit_prev']
agg_pa['downp_percent'] = agg_pa['down_pay'] / agg_pa['sum_credit_prev'] *100

agg_pa.reset_index(inplace=True)

# We will calculate the average days delay, the extra amount paid, the days delay
agg_inst = installments.groupby('SK_ID_CURR').agg(
    paid_on = ('DAYS_ENTRY_PAYMENT','mean'),
    pay_due = ('DAYS_INSTALMENT','mean'),
    amt_paid = ('AMT_PAYMENT','sum'),
    amt_due = ('AMT_INSTALMENT','sum')
)

# We will calculate the extra amount paid and the days delay
agg_inst['amt_extra'] = agg_inst['amt_paid'] - agg_inst['amt_due']
agg_inst['days_delay'] = agg_inst['paid_on'] - agg_inst['pay_due']
agg_inst.drop(columns=['amt_paid','amt_due','pay_due','paid_on'], inplace=True)

agg_pa.reset_index(inplace=True)

# Merge the aggregated data with the main data
app_train = app_train.merge(agg_bureau, on='SK_ID_CURR', how='left')
app_train = app_train.merge(agg_pos, on='SK_ID_CURR', how='left')
app_train = app_train.merge(agg_cc, on='SK_ID_CURR', how='left')
app_train = app_train.merge(agg_pa, on='SK_ID_CURR', how='left')
app_train = app_train.merge(agg_inst, on='SK_ID_CURR', how='left')

app_test = app_test.merge(agg_bureau, on='SK_ID_CURR', how='left')
app_test = app_test.merge(agg_pos, on='SK_ID_CURR', how='left')
app_test = app_test.merge(agg_cc, on='SK_ID_CURR', how='left')
app_test = app_test.merge(agg_pa, on='SK_ID_CURR', how='left')
app_test = app_test.merge(agg_inst, on='SK_ID_CURR', how='left')

# Function to segment columns into categorical, label, and one-hot encoded columns
def col_segment(df):
    cat_cols = []
    label_cols = []
    ohe_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            cat_cols.append(col)
        elif df[col].nunique() < 3:
            label_cols.append(col)
        else:
            ohe_cols.append(col)
    
    return cat_cols, label_cols, ohe_cols

cat_cols,label_cols,ohe_cols = col_segment(app_train)

# Get the dummies for the categorical columns and label encode the label columns for both train and test sets
# Dummies means that we will create a new column for each unique value in the column and assign 1 or 0 to the new column
preprocessed_train = pd.get_dummies(app_train, columns=cat_cols)
preprocessed_test = pd.get_dummies(app_test, columns=cat_cols)

label_encoder = LabelEncoder()
for col in label_cols:
    label_encoder.fit(app_train[col])
    preprocessed_train[col] = label_encoder.transform(app_train[col])
    preprocessed_test[col] = label_encoder.transform(app_test[col])

# inner join to align the columns in the train and test sets because the get_dummies function may create different columns
preprocessed_train,preprocessed_test = preprocessed_train.align(preprocessed_test, join = 'inner', axis = 1)

col_names=preprocessed_train.columns

# Replace infinite values with 1e9, 1e9 is a large number that can be used to replace infinite values
preprocessed_train = preprocessed_train.replace([np.inf, -np.inf], 1e9)
preprocessed_test = preprocessed_test.replace([np.inf, -np.inf], 1e9)

# Create new columns for 0-5y and 5-10y 10-30y, 30-40y and >40y car age groups and drop the original column
preprocessed_train['OWN_CAR_AGE_0-5'] = preprocessed_train['OWN_CAR_AGE'].apply(lambda x: 1 if x <= 5 else 0)
preprocessed_train['OWN_CAR_AGE_5-10'] = preprocessed_train['OWN_CAR_AGE'].apply(lambda x: 1 if 5 < x <= 10 else 0)
preprocessed_train['OWN_CAR_AGE_10-30'] = preprocessed_train['OWN_CAR_AGE'].apply(lambda x: 1 if 10 < x <= 30 else 0)
preprocessed_train['OWN_CAR_AGE_30-40'] = preprocessed_train['OWN_CAR_AGE'].apply(lambda x: 1 if 30 < x <= 40 else 0)
preprocessed_train['OWN_CAR_AGE_>40'] = preprocessed_train['OWN_CAR_AGE'].apply(lambda x: 1 if x > 40 else 0)

preprocessed_test['OWN_CAR_AGE_0-5'] = preprocessed_test['OWN_CAR_AGE'].apply(lambda x: 1 if x <= 5 else 0)
preprocessed_test['OWN_CAR_AGE_5-10'] = preprocessed_test['OWN_CAR_AGE'].apply(lambda x: 1 if 5 < x <= 10 else 0)
preprocessed_test['OWN_CAR_AGE_10-30'] = preprocessed_test['OWN_CAR_AGE'].apply(lambda x: 1 if 10 < x <= 30 else 0)
preprocessed_test['OWN_CAR_AGE_30-40'] = preprocessed_test['OWN_CAR_AGE'].apply(lambda x: 1 if 30 < x <= 40 else 0)
preprocessed_test['OWN_CAR_AGE_>40'] = preprocessed_test['OWN_CAR_AGE'].apply(lambda x: 1 if x > 40 else 0)

preprocessed_train.drop(columns='OWN_CAR_AGE', inplace=True)
preprocessed_test.drop(columns='OWN_CAR_AGE', inplace=True)
col_names=preprocessed_train.columns

# fill the nan values in amt_req_credit_bureau, ext_source_mean, ext_source_std, dti, annuity_percent, days_employed_percent, credit_term with 0
preprocessed_train['AMT_REQ_CREDIT_BUREAU_HOUR'] = preprocessed_train['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0)
preprocessed_test['AMT_REQ_CREDIT_BUREAU_HOUR'] = preprocessed_test['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0)

preprocessed_train['AMT_REQ_CREDIT_BUREAU_DAY'] = preprocessed_train['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0)
preprocessed_test['AMT_REQ_CREDIT_BUREAU_DAY'] = preprocessed_test['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0)

preprocessed_train['AMT_REQ_CREDIT_BUREAU_WEEK'] = preprocessed_train['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0)
preprocessed_test['AMT_REQ_CREDIT_BUREAU_WEEK'] = preprocessed_test['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0)

preprocessed_train['AMT_REQ_CREDIT_BUREAU_MON'] = preprocessed_train['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0)
preprocessed_test['AMT_REQ_CREDIT_BUREAU_MON'] = preprocessed_test['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0)

preprocessed_train['AMT_REQ_CREDIT_BUREAU_QRT'] = preprocessed_train['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0)
preprocessed_test['AMT_REQ_CREDIT_BUREAU_QRT'] = preprocessed_test['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0)

preprocessed_train['AMT_REQ_CREDIT_BUREAU_YEAR'] = preprocessed_train['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)
preprocessed_test['AMT_REQ_CREDIT_BUREAU_YEAR'] = preprocessed_test['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)

preprocessed_train['EXT_SOURCE_MEAN'] = preprocessed_train['EXT_SOURCE_MEAN'].fillna(0)
preprocessed_test['EXT_SOURCE_MEAN'] = preprocessed_test['EXT_SOURCE_MEAN'].fillna(0)

preprocessed_train['EXT_SOURCE_STD'] = preprocessed_train['EXT_SOURCE_STD'].fillna(0)
preprocessed_test['EXT_SOURCE_STD'] = preprocessed_test['EXT_SOURCE_STD'].fillna(0)

preprocessed_train['DTI'] = preprocessed_train['DTI'].fillna(0)
preprocessed_test['DTI'] = preprocessed_test['DTI'].fillna(0)

preprocessed_train['ANNUITY_PERCENT'] = preprocessed_train['ANNUITY_PERCENT'].fillna(0)
preprocessed_test['ANNUITY_PERCENT'] = preprocessed_test['ANNUITY_PERCENT'].fillna(0)

preprocessed_train['DAYS_EMPLOYED_PERCENT'] = preprocessed_train['DAYS_EMPLOYED_PERCENT'].fillna(0)
preprocessed_test['DAYS_EMPLOYED_PERCENT'] = preprocessed_test['DAYS_EMPLOYED_PERCENT'].fillna(0)

preprocessed_train['CREDIT_TERM'] = preprocessed_train['CREDIT_TERM'].fillna(0)
preprocessed_test['CREDIT_TERM'] = preprocessed_test['CREDIT_TERM'].fillna(0)

preprocessed_train['STATUS_CHANGE'] = preprocessed_train['STATUS_CHANGE'].fillna(0)
preprocessed_test['STATUS_CHANGE'] = preprocessed_test['STATUS_CHANGE'].fillna(0)

print(f"Train shape: {preprocessed_train.shape}")
print(f"Test shape: {preprocessed_test.shape}")
# print the number of nan values in the train and test sets
print(f"Number of nan values in train set: {preprocessed_train.isna().sum().sum()}")
print(f"Number of nan values in test set: {preprocessed_test.isna().sum().sum()}")

imputer = SimpleImputer(strategy = 'median')
preprocessed_train = imputer.fit_transform(preprocessed_train)
preprocessed_test = imputer.transform(preprocessed_test)

# imputer = KNNImputer(n_neighbors=5)
# preprocessed_train = imputer.fit_transform(preprocessed_train)
# preprocessed_test = imputer.transform(preprocessed_test)
# save the id and target of the columns and rows to not lose them after scaling, we will not scale the target column or the id column
# save also the following rows to add them back after scaling: 

scaler = RobustScaler()
preprocessed_train = scaler.fit_transform(preprocessed_train)
preprocessed_test = scaler.transform(preprocessed_test)

# Convert the numpy arrays back to DataFrames
preprocessed_train = pd.DataFrame(preprocessed_train, columns=col_names)
preprocessed_test = pd.DataFrame(preprocessed_test, columns=col_names)

# add the id and target columns back to the preprocessed data
preprocessed_train['SK_ID_CURR'] = app_train['SK_ID_CURR']
preprocessed_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

preprocessed_train['TARGET'] = app_train['TARGET']


print(f"Train shape: {preprocessed_train.shape}")
print(f"Test shape: {preprocessed_test.shape}")
# print the number of nan values in the train and test sets
print(f"Number of nan values in train set: {preprocessed_train.isna().sum().sum()}")
print(f"Number of nan values in test set: {preprocessed_test.isna().sum().sum()}")

# Save the preprocessed data to ./results/feature_engineering.csv
preprocessed_train.to_csv("./results/feature_engineering/feature_engineering_train.csv", index=False)
preprocessed_test.to_csv("./results/feature_engineering/feature_engineering_test.csv", index=False)
print("Preprocessing done!")