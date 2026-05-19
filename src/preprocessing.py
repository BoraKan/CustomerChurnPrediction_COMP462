import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_excel(path)
    # Total Charges comes as string in the dataset, need to convert
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)
    return df


def engineer_features(df):
    """
    Creates domain-driven features before one-hot encoding.
    Internet Service column must still be in its original string form when this is called.
    """
    service_cols = [
        'Multiple Lines', 'Online Security', 'Online Backup',
        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]
    # Total number of optional add-on services the customer subscribes to.
    # Customers with fewer services are more likely to churn (less switching cost).
    df['services_count'] = df[service_cols].sum(axis=1)

    # Average monthly spend per month of tenure.
    # High values for short-tenure customers may indicate price sensitivity.
    df['charge_per_tenure'] = df['Monthly Charges'] / (df['Tenure Months'] + 1)

    # Customers in their first 6 months are at highest churn risk.
    df['is_new_customer'] = (df['Tenure Months'] < 6).astype(int)

    # Subscribing to both phone and internet means higher switching cost.
    # Internet Service is still a string category at this point.
    df['has_both_services'] = (
        (df['Phone Service'] == 1) & (df['Internet Service'] != 'No')
    ).astype(int)

    return df


def preprocess(df):
    # columns that are not useful for prediction
    drop_cols = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude',
        'Churn Score', 'CLTV', 'Churn Reason', 'Churn Label'
    ]
    df = df.drop(columns=drop_cols)

    # simple yes/no columns
    binary_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # these columns have a third value like "No internet service" which we treat as No
    service_cols = [
        'Multiple Lines', 'Online Security', 'Online Backup',
        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]
    for col in service_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # engineer new features before one-hot encoding (Internet Service still string here)
    df = engineer_features(df)

    # one-hot encode the remaining categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'Churn Value' in cat_cols:
        cat_cols.remove('Churn Value')
    df = pd.get_dummies(df, columns=cat_cols)

    return df


def split_and_scale(df, target='Churn Value', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler
