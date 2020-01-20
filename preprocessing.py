import pandas as pd
from dateutil.relativedelta import *


def rename_columns(invoices: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    column_names = list(invoices.columns)
    if 'Customer ID' in column_names:
        invoices = invoices.rename(columns={'Customer ID': 'CustomerID'})
    if 'Invoice' in column_names:
        invoices = invoices.rename(columns={'Invoice': 'InvoiceNo'})
    if 'Price' in column_names:
        invoices = invoices.rename(columns={'Price': 'UnitPrice'})

    return invoices


def concat_invoice_dataframes(invoices1: pd.DataFrame, invoices2: pd.DataFrame) -> pd.DataFrame:
    invoices1, invoices2 = invoices1.copy(), invoices2.copy()
    end_of_overlap_period = '2010-12-10'  # Specific to the two UCI online retail datasets

    if invoices1.InvoiceDate.max() > invoices2.InvoiceDate.max():
        invoices1 = invoices1[invoices1.InvoiceDate > end_of_overlap_period]
        invoices = pd.concat([invoices2, invoices1])
    else:
        invoices2 = invoices2[invoices2.InvoiceDate > end_of_overlap_period]
        invoices = pd.concat([invoices1, invoices2])

    return invoices


def add_revenue_column(invoices: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    if "Revenue" not in list(invoices.columns):
        invoices['Revenue'] = invoices['UnitPrice'] * invoices['Quantity']

    return invoices


def drop_test_invoices(invoices: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    test_invoice_indexs = invoices[invoices['StockCode'].str.contains('TEST', case=False, na=False)].index
    invoices = invoices.drop(index=test_invoice_indexs)

    return invoices


def drop_cancellation_invoices(invoices: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    cancellation_invoice_indexs = invoices[invoices["InvoiceNo"].str.contains('c', na=False, case=False)].index
    invoices = invoices.drop(index=cancellation_invoice_indexs)

    return invoices


def drop_return_invoices(invoices: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    invoices = invoices[invoices['UnitPrice'] > 0].copy()

    return invoices


def drop_non_numeric_invoice_numbers(invoices: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    invoices = invoices[pd.to_numeric(invoices['InvoiceNo'], errors='coerce').notna()]
    return invoices


def clean_stock_codes(invoices):
    invoices_copy = invoices.copy()
    invoices_copy.drop(index=invoices_copy[invoices_copy.StockCode == 'C2'].index, inplace=True)
    invoices_copy.drop(index=invoices_copy[invoices_copy.StockCode == 'C3'].index, inplace=True)
    invoices_copy['StockCode'] = invoices_copy['StockCode'].str.replace("^\D+$", "Not an Item")
    invoices_copy["StockCode"] = invoices_copy["StockCode"].str.replace("gift.*", "Not an Item")
    invoices_copy.drop(index=invoices_copy[invoices_copy.StockCode == 'Not an Item'].index, inplace=True)
    invoices_copy['StockCode'] = invoices_copy['StockCode'].str.replace("\D+$", "")

    return invoices_copy


def get_observation_end_dates(invoices: pd.DataFrame) -> list:
    first_invoice_date = invoices.InvoiceDate.min().date()
    last_invoice_date = invoices.InvoiceDate.max().date()
    time_difference = relativedelta(last_invoice_date, first_invoice_date)
    num_months = time_difference.years * 12 + time_difference.months
    observation_end_dates = [first_invoice_date + relativedelta(months=month_num) for month_num in
                             range(0, num_months + 1)]
    observation_end_dates = observation_end_dates[2:-1]

    return observation_end_dates
