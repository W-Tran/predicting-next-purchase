import pandas as pd
import numpy as np
from dateutil.relativedelta import *
from lifetimes.utils import summary_data_from_transaction_data


def train_split_invoices_into_calib_holdout(observation_period_end: pd.datetime, invoices: pd.DataFrame):
    calib_period_end = observation_period_end - relativedelta(months=1)
    calib_invoices = invoices[invoices.InvoiceDate <= calib_period_end].copy()
    holdout_invoices = invoices[invoices.InvoiceDate > calib_period_end].copy()

    return calib_invoices, holdout_invoices, calib_period_end


def test_split_invoices_into_calib_holdout(observation_period_end: pd.datetime, invoices: pd.DataFrame):
    calib_period_end = observation_period_end
    calib_invoices = invoices[invoices.InvoiceDate <= calib_period_end].copy()
    holdout_invoices = invoices[invoices.InvoiceDate > calib_period_end].copy()

    return calib_invoices, holdout_invoices, calib_period_end


def get_aggregation_features(calib_invoices):
    calib_invoices = calib_invoices.copy()
    calib_invoices['InvoiceDay'] = calib_invoices['InvoiceDate'].dt.date
    calib_single_daily_invoices = calib_invoices.drop_duplicates(subset=['CustomerID', 'InvoiceDay'], keep='first')
    calib_single_daily_invoices = calib_single_daily_invoices.sort_values(by=['CustomerID', 'InvoiceDate'])

    calib_single_daily_invoices['PrevInvoiceDate'] = calib_single_daily_invoices.groupby('CustomerID')[
        'InvoiceDate'].shift(1)
    calib_single_daily_invoices['TimeBetweenInvoices'] = (
            calib_single_daily_invoices['InvoiceDate'] - calib_single_daily_invoices['PrevInvoiceDate']).dt.days
    aggregation_features = calib_single_daily_invoices.groupby('CustomerID')['TimeBetweenInvoices'].agg(
        ['mean', 'std', 'min', 'max']).reset_index()
    aggregation_features.columns = ['CustomerID', 'MeanTimeBetweenPurchase', 'StDevTimeBetweenPurchase',
                                    'MinTimeBetweenPurchase', 'MaxTimeBetweenPurchase']

    features = calib_invoices[['CustomerID']].drop_duplicates().sort_values(by='CustomerID').reset_index(
        drop=True).copy()
    features = features.merge(aggregation_features, how='left', on='CustomerID')
    features = features.dropna()  # Customers who purchased at least 3 times during calibration period

    return features


def add_monetary_agg_features(features, calib_invoices):
    features = features.copy()
    calib_invoices = calib_invoices.copy()
    calib_daily_revenues = calib_invoices.groupby(["CustomerID", calib_invoices["InvoiceDate"].dt.to_period("D")])[
        'Revenue'].sum().reset_index()
    calib_daily_revenues = calib_daily_revenues.sort_values(by=['CustomerID', 'InvoiceDate'])

    aggregation_features = calib_daily_revenues.groupby('CustomerID')['Revenue'].agg([
        'mean',
        'std',
        'min',
        'max',
        'sum',
    ]).reset_index()

    aggregation_features.columns = [
        'CustomerID',
        'MeanPurchaseValue',
        'StDevPurchaseValue',
        'MinPurchaseValue',
        'MaxPurchaseValue',
        'SumPurchaseValue',
    ]

    features = features.merge(aggregation_features, how='left', on='CustomerID')

    return features


def add_rfm_features(features, calib_invoices, period_end):
    features = features.copy()
    rfm_features = summary_data_from_transaction_data(
        transactions=calib_invoices,
        customer_id_col='CustomerID',
        datetime_col='InvoiceDate',
        monetary_value_col='Revenue',
        observation_period_end=period_end,
        freq='D'
    )
    rfm_features['T_Minus_Recency'] = rfm_features['T'] - rfm_features['recency']
    features = features.merge(rfm_features, how='left', on='CustomerID')

    return features


def add_cyclical_last_invoice_date_feature(features, calib_invoices):
    features = features.copy()
    customers_last_calib_invoices = calib_invoices.groupby("CustomerID")['InvoiceDate'].max().reset_index()
    customers_last_calib_invoices.columns = ['CustomerID', 'LastCalibPurchase']
    customers_last_calib_invoices['CurrentDayOfYear'] = customers_last_calib_invoices['LastCalibPurchase'].apply(
        lambda x: x.timetuple().tm_yday)
    customers_last_calib_invoices['CurrentDayOfYear_sin'] = np.sin(
        2 * np.pi * customers_last_calib_invoices['CurrentDayOfYear'] / 365)
    customers_last_calib_invoices['CurrentDayOfYear_cos'] = np.cos(
        2 * np.pi * customers_last_calib_invoices['CurrentDayOfYear'] / 365)
    customers_last_calib_invoices.drop(columns=['CurrentDayOfYear', 'LastCalibPurchase'], inplace=True)
    features = features.merge(customers_last_calib_invoices, how='left', on='CustomerID')

    return features


def add_uk_feature(features, calib_invoices):
    uk_or_not = calib_invoices[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID']).copy()
    uk_or_not['UK'] = uk_or_not['Country'] == 'United Kingdom'
    uk_or_not = uk_or_not.drop(columns='Country')
    features = features.merge(uk_or_not, how='left', on='CustomerID')

    return features


def get_labels(features, calib_invoices, holdout_invoices, period_end):
    customer_last_calib_purchases = calib_invoices.groupby("CustomerID")['InvoiceDate'].max().reset_index()
    customer_first_holdout_purchases = holdout_invoices.groupby("CustomerID")['InvoiceDate'].min().reset_index()
    labels = customer_last_calib_purchases.merge(customer_first_holdout_purchases, how='left', on='CustomerID')
    labels.columns = ['CustomerID', 'LastCalibPurchase', 'FirstHoldoutPurchase']
    labels['PurchaseNextMonth'] = \
        (labels['FirstHoldoutPurchase'] >= period_end) & \
        (labels['FirstHoldoutPurchase'] < period_end + relativedelta(months=1))
    labels = labels.drop(columns=['LastCalibPurchase', 'FirstHoldoutPurchase'])
    labels = labels[labels.CustomerID.isin(features.CustomerID)]

    return labels


def _get_train_test_data(invoices, observation_end_date):
    train_calib_invoices, train_holdout_invoices, train_calib_period_end = train_split_invoices_into_calib_holdout(
        observation_end_date, invoices)
    test_calib_invoices, test_holdout_invoices, test_calib_period_end = test_split_invoices_into_calib_holdout(
        observation_end_date, invoices)

    train_features = get_aggregation_features(train_calib_invoices)
    train_features = add_monetary_agg_features(train_features, train_calib_invoices)
    train_features = add_rfm_features(train_features, train_calib_invoices, train_calib_period_end)
    train_features = add_uk_feature(train_features, train_calib_invoices)
    # train_features = add_cyclical_last_invoice_date_feature(train_features, train_calib_invoices)

    test_features = get_aggregation_features(test_calib_invoices)
    test_features = add_monetary_agg_features(test_features, test_calib_invoices)
    test_features = add_rfm_features(test_features, test_calib_invoices, test_calib_period_end)
    test_features = add_uk_feature(test_features, test_calib_invoices)
    # test_features = add_cyclical_last_invoice_date_feature(test_features, test_calib_invoices)

    train_labels = get_labels(train_features, train_calib_invoices, train_holdout_invoices, train_calib_period_end)
    test_labels = get_labels(test_features, test_calib_invoices, test_holdout_invoices, test_calib_period_end)

    X_train = train_features.drop(columns='CustomerID')
    X_test = test_features.drop(columns='CustomerID')
    y_train = train_labels['PurchaseNextMonth'].copy()
    y_test = test_labels['PurchaseNextMonth'].copy()

    return X_train, y_train, X_test, y_test


def get_naive_labels(invoices, observation_end_dates):
    naive_labels = dict(
        y_train=[],
        y_test=[],
    )

    for observation_end_date in observation_end_dates:
        train_calib_invoices, _, _ = train_split_invoices_into_calib_holdout(observation_end_date, invoices)
        test_calib_invoices, _, _ = test_split_invoices_into_calib_holdout(observation_end_date, invoices)

        train_features = get_aggregation_features(train_calib_invoices)
        test_features = get_aggregation_features(test_calib_invoices)

        days_in_a_month = 30.4167
        naive_labels['y_train'].append(train_features['MeanTimeBetweenPurchase'] < days_in_a_month)
        naive_labels['y_test'].append(test_features['MeanTimeBetweenPurchase'] < days_in_a_month)

    return naive_labels
