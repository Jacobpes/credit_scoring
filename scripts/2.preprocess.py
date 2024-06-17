import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

app_train = pd.read_csv("./data/application_train.csv")
app_test = pd.read_csv("./data/application_test.csv")
prev_app = pd.read_csv("./data/previous_application.csv")
installments = pd.read_csv("./data/installments_payments.csv")
pos = pd.read_csv("./data/POS_CASH_balance.csv")
bureau = pd.read_csv("./data/bureau.csv")
bureau_bal = pd.read_csv("./data/bureau_balance.csv")
credit_card = pd.read_csv("./data/credit_card_balance.csv")

app_train['FLAG_OWN_CAR'] = app_train['FLAG_OWN_CAR'].apply(lambda x: 1 if x == 'Y' else 0)
app_test['FLAG_OWN_CAR'] = app_test['FLAG_OWN_CAR'].apply(lambda x: 1 if x == 'Y' else 0)
app_train['FLAG_OWN_REALTY'] = app_train['FLAG_OWN_REALTY'].apply(lambda x: 1 if x == 'Y' else 0)
app_test['FLAG_OWN_REALTY'] = app_test['FLAG_OWN_REALTY'].apply(lambda x: 1 if x == 'Y' else 0)
app_train['CODE_GENDER'] = app_train['CODE_GENDER'].apply(lambda x:np.nan if x == 'XNA' else x)
app_test['CODE_GENDER'] = app_test['CODE_GENDER'].apply(lambda x:np.nan if x == 'XNA' else x)

float64_columns = app_train.select_dtypes(include=['float64']).columns
app_train[float64_columns] = app_train[float64_columns].astype('float32')

int64_columns = app_train.select_dtypes(include=['int64']).columns
app_train[int64_columns] = app_train[int64_columns].astype('int32')

float64_columns = app_test.select_dtypes(include=['float64']).columns
app_test[float64_columns] = app_test[float64_columns].astype('float32')

int64_columns = app_test.select_dtypes(include=['int64']).columns
app_test[int64_columns] = app_test[int64_columns].astype('int32')

grouped = app_train.groupby('TARGET')['AMT_INCOME_TOTAL'].mean()

#we will impute the too big values with np.nan
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].apply(lambda x: np.nan if x > 350000 else x)
app_test['DAYS_EMPLOYED'] = app_test['DAYS_EMPLOYED'].apply(lambda x: np.nan if x > 350000 else x)

app_train['OWN_CAR_AGE'] = app_train['OWN_CAR_AGE'].apply(lambda x: np.nan if x == 64 or x == 65 else x)
app_test['OWN_CAR_AGE'] = app_test['OWN_CAR_AGE'].apply(lambda x: np.nan if x == 64 or x == 65 else x)

bin_edges = [0,21,26,36,46,56,66,100]

# Define the labels for the bins
bin_labels = ['0-20', '20-25', '26-35', '36-45', '46-55', '56-65', '65+']

# Create a new column with the age bins efficiently
age_bins = pd.cut(app_train['DAYS_BIRTH']/-365, bins=bin_edges, labels=bin_labels)
age_bins_test = pd.cut(app_test['DAYS_BIRTH']/-365, bins=bin_edges, labels=bin_labels)

# Concatenate age bins columns efficiently
app_train['AGE'] = pd.concat([age_bins], axis=1).iloc[:, 0]
app_test['AGE'] = pd.concat([age_bins_test], axis=1).iloc[:, 0]


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

# Example usage of col_segment function
cat_cols, label_cols, ohe_cols = col_segment(app_train)

# Continue with your preprocessing steps...

# Efficiently add calculated columns
annuity_percent_train = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL'] * 100
annuity_percent_test = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL'] * 100

# Concatenate ANNUITY_PERCENT columns efficiently
app_train['ANNUITY_PERCENT'] = pd.concat([annuity_percent_train], axis=1).iloc[:, 0]
app_test['ANNUITY_PERCENT'] = pd.concat([annuity_percent_test], axis=1).iloc[:, 0]

# Repeat similar steps for DTI and CTA calculations

# Remember to create copies of DataFrames after significant operations to avoid fragmentation
app_train_copy = app_train.copy()
app_test_copy = app_test.copy()


# CTA is credit to annuity ratio which means how many times the annuity can cover the credit. The higher the better.
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

# The low number of some income type entries here might give us a wrong indication about the behaviour of 
# the category becuase we don't have enough entries as we have <20 entries in businessman, student, unemployed 
# and maternity leave. But we choose to go ahead as it's still an indication.
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

agg_bureau = bureau.groupby('SK_ID_CURR').agg(
    num_active_loans=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
    sum_credit_prolong=('CNT_CREDIT_PROLONG', 'sum'),
    num_past_loans = ('CREDIT_ACTIVE','count'),
    avg_days_overdue = ('CREDIT_DAY_OVERDUE','mean'),
    total_credit_limit = ('AMT_CREDIT_SUM_LIMIT','sum'),
    avg_annuity = ('AMT_ANNUITY','mean')
)

agg_bureau.reset_index(inplace=True)

agg_pos = pos.groupby('SK_ID_CURR').agg(
    late_payment_freq = ('SK_DPD', lambda x: (x > 30).sum()),
    DPD_sum = ('SK_DPD','sum'),
    avg_installment_duration = ('CNT_INSTALMENT','mean'),
    max_installment_left = ('CNT_INSTALMENT_FUTURE','max'),
)

agg_pos.reset_index(inplace=True)

agg_cc = credit_card.groupby('SK_ID_CURR').agg(
    avg_balance = ('AMT_BALANCE','mean'),
    atm_drawings=('AMT_DRAWINGS_ATM_CURRENT', 'sum'),
    pos_drawings =('AMT_DRAWINGS_POS_CURRENT', 'sum'),
    all_drawings = ('AMT_DRAWINGS_CURRENT','sum'),
    credit_limit=('AMT_CREDIT_LIMIT_ACTUAL','sum')
)
agg_cc['atm_drawings'] = agg_cc['atm_drawings'] / agg_cc['all_drawings']+0.001 *100
agg_cc['pos_drawings'] = agg_cc['pos_drawings'] / agg_cc['all_drawings']+0.001 *100
agg_cc['credit_usage'] = agg_cc['all_drawings'] / agg_cc['credit_limit']+0.001 *100

agg_cc.reset_index(inplace=True)

prev_app['NAME_CLIENT_TYPE']= prev_app['NAME_CLIENT_TYPE'].apply(
    lambda x: np.nan if x == 'XNA' else x)
prev_app['NAME_YIELD_GROUP'] = prev_app['NAME_YIELD_GROUP'].apply(
    lambda x: np.nan if x == 'XNA' else x)

agg_pa = prev_app.groupby('SK_ID_CURR').agg(
    sum_credit_prev = ('AMT_CREDIT','sum'),
    app_min_credit = ('AMT_APPLICATION','sum'),
    down_pay = ('AMT_DOWN_PAYMENT','sum'),
    days_decision = ('DAYS_DECISION','max'),
    req_insurance = ('NFLAG_INSURED_ON_APPROVAL','sum')
)
agg_pa['app_min_credit'] = agg_pa['app_min_credit'] - agg_pa['sum_credit_prev']
agg_pa['downp_percent'] = agg_pa['down_pay'] / agg_pa['sum_credit_prev'] *100

agg_pa.reset_index(inplace=True)

agg_inst = installments.groupby('SK_ID_CURR').agg(
    paid_on = ('DAYS_ENTRY_PAYMENT','mean'),
    pay_due = ('DAYS_INSTALMENT','mean'),
    amt_paid = ('AMT_PAYMENT','sum'),
    amt_due = ('AMT_INSTALMENT','sum')
)

agg_inst['amt_extra'] = agg_inst['amt_paid'] - agg_inst['amt_due']
agg_inst['days_delay'] = agg_inst['paid_on'] - agg_inst['pay_due']
agg_inst.drop(columns=['amt_paid','amt_due','pay_due','paid_on'], inplace=True)

agg_pa.reset_index(inplace=True)

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

cat_cols,label_cols,ohe_cols = col_segment(app_train)
eng_train = pd.get_dummies(app_train, columns=cat_cols)
eng_test = pd.get_dummies(app_test, columns=cat_cols)

label_encoder = LabelEncoder()
for col in label_cols:
    label_encoder.fit(app_train[col])
    eng_train[col] = label_encoder.transform(app_train[col])
    eng_test[col] = label_encoder.transform(app_test[col])

eng_train,eng_test = eng_train.align(eng_test, join = 'inner', axis = 1)

col_names=eng_train.columns

eng_train = eng_train.replace([np.inf, -np.inf], 1e9)
eng_test = eng_test.replace([np.inf, -np.inf], 1e9)

# Save the preprocessed data to ./results/feature_engineering.csv
eng_train.to_csv("./results/feature_engineering.csv", index=False)
eng_test.to_csv("./results/feature_engineering_test.csv", index=False)
print("Preprocessing done!")