import numpy as np
import pandas as pd
from functools import reduce
import re


def agg_subgroup(df, group_key, agg_dict, prefix):
    out = df.groupby(group_key).agg(agg_dict)
    out.columns = [f"{prefix}_{c[0]}_{c[1]}" for c in out.columns]
    return out.reset_index()

def add_app_features(df):
    df = df.copy()
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    df['app__CREDIT_INCOME_RATIO']  = (df["AMT_CREDIT"]  / df["AMT_INCOME_TOTAL"]).replace([np.inf,-np.inf], np.nan)
    df['app__ANNUITY_INCOME_RATIO'] = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).replace([np.inf,-np.inf], np.nan)
    df['app__CREDIT_TERM']          = (df["AMT_ANNUITY"] / df["AMT_CREDIT"]).replace([np.inf,-np.inf], np.nan)
    df['app__DAYS_EMPLOYED_PERC']   = (df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]).replace([np.inf,-np.inf], np.nan)
    df['app__INCOME_PER_PERSON']    = (df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]).replace([np.inf,-np.inf], np.nan)
    df['app__INCOME_PER_P_TO_AMT']  = (df["app__INCOME_PER_PERSON"] / df["AMT_CREDIT"]).replace([np.inf,-np.inf], np.nan)
    
    ext = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]]
    df['app__EXT_SOURCE_MEAN'] = ext.mean(axis=1)
    df['app__EXT_SOURCE_STD']  = ext.std(axis=1)
    df['app__EXT_SOURCE_PROD'] = (df["EXT_SOURCE_1"]*df["EXT_SOURCE_2"]*df["EXT_SOURCE_3"]).replace([np.inf,-np.inf], np.nan)
    df['app__credit_ext_1'] = (df['AMT_CREDIT'] / df['EXT_SOURCE_1']).replace([np.inf,-np.inf], np.nan)
    df['app__credit_ext_2'] = (df['AMT_CREDIT'] / df['EXT_SOURCE_2']).replace([np.inf,-np.inf], np.nan)
    df['app__credit_ext_3'] = (df['AMT_CREDIT'] / df['EXT_SOURCE_3']).replace([np.inf,-np.inf], np.nan)
    df['app__credit_annuity_ext_1'] = (df['AMT_ANNUITY'] / df['EXT_SOURCE_1']).replace([np.inf,-np.inf], np.nan)
    df['app__credit_annuity_ext_2'] = (df['AMT_ANNUITY'] / df['EXT_SOURCE_2']).replace([np.inf,-np.inf], np.nan)
    df['app__credit_annuity_ext_3'] = (df['AMT_ANNUITY'] / df['EXT_SOURCE_3']).replace([np.inf,-np.inf], np.nan)

    #Age
    df['app__age'] = -df['DAYS_BIRTH'] / 365.0
    labels=["age_group_1", "age_group_2", "age_group_3", "age_group_4", "age_group5","age_group6", "age_group7"]
    df['app__age_group'] = pd.cut(df['app__age'], [0, 20, 30, 40, 50, 60, 70, np.inf], labels=labels)

    #doc count
    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    df['app__doc_count'] = df[doc_cols].sum(axis=1)

    # Employment ratios
    df['app__INCOME_TO_EMPLOYED_RATIO'] = (df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']).replace([np.inf,-np.inf], np.nan)
    df['app__INCOME_TO_AGE']            = (df['AMT_INCOME_TOTAL'] / df['app__age']).replace([np.inf,-np.inf], np.nan)
    df['app__AMT_TO_DAYS_EMPLOYED']     = (df["AMT_CREDIT"] / df['DAYS_EMPLOYED']).replace([np.inf,-np.inf], np.nan)

    return df

def create_count_cols_cat(df, column_to_count, new_cols=None):
    df = df.copy()
    agg_df = pd.DataFrame(columns=["SK_ID_CURR"])
    for col in new_cols:
        tmp_df = (
            df[column_to_count].eq(col)
            .groupby(df["SK_ID_CURR"])
            .sum()
            .rename(f"{col}_{column_to_count.lower()}_count")
            .reset_index()
        )
        agg_df = agg_df.merge(tmp_df, on="SK_ID_CURR", how='outer')
    return agg_df

def create_bureau_dpd_df(df:pd.DataFrame):
    df = df.copy()

    bureau_balance_df_valid = df[~(df['STATUS'].isin(["X", "C"]))]
    dpd_counts_mild = (
        bureau_balance_df_valid[
            bureau_balance_df_valid["STATUS"]
            .isin(['1', '2', '3', '4'])
        ]
        .groupby('SK_ID_BUREAU')['STATUS']
        .count()
        .reset_index()
        .rename(columns={'STATUS': 'mild_dpd_months_count'})
    )
    dpd_counts_severe = (
        bureau_balance_df_valid[bureau_balance_df_valid["STATUS"] == '5']
        .groupby('SK_ID_BUREAU')['STATUS']
        .count()
        .reset_index()
        .rename(columns={'STATUS': 'severe_dpd_months_count'})
    )

    total_months = (
    df[~df["STATUS"].isin(['X', 'C'])]
        .groupby("SK_ID_BUREAU")
        .agg({"MONTHS_BALANCE": "count"})
        .reset_index()
        .rename({"MONTHS_BALANCE": "sum_month"}, axis=1)
    )

    dpd_severe_df = dpd_counts_severe.merge(total_months, on="SK_ID_BUREAU", how="left")
    dpd_mild_df = dpd_counts_mild.merge(total_months, on="SK_ID_BUREAU", how="left")
    dpd_severe_df["severe_dpd_ratio"] = dpd_severe_df["severe_dpd_months_count"] / dpd_severe_df["sum_month"]
    dpd_mild_df["mild_dpd_ratio"] = dpd_mild_df["mild_dpd_months_count"] / dpd_mild_df["sum_month"]
    dpd_severe_df.drop(["severe_dpd_months_count", "sum_month"], axis=1, inplace=True)
    dpd_mild_df.drop(["mild_dpd_months_count", "sum_month"], axis=1, inplace=True)
    return dpd_severe_df, dpd_mild_df

def get_pos_dpd_dfs(df, n_months=(13, 25, 37)):
    agg_df_list = []

    for month_ in n_months:
        pos_agg_dpd_n = (
            df[df["MONTHS_BALANCE"] > -month_]
            .groupby("SK_ID_PREV")
            .agg({
                "SK_DPD": ["max", "mean", "sum", "std"],
                "SK_DPD_DEF": ["max", "mean", "sum", "std"],
            })
            .reset_index()
        )

        pos_agg_dpd_n.columns = [
            "_".join(col).strip("_")
            for col in pos_agg_dpd_n.columns
        ]

        pos_agg_dpd_n.columns = ["SK_ID_PREV"] + [
            f"pos_prev{month_}_dpd_{col}"
            for col in pos_agg_dpd_n.columns[1:]
        ]

        agg_df_list.append(pos_agg_dpd_n)

    return reduce(
        lambda left, right: left.merge(right, on="SK_ID_PREV", how="outer"),
        agg_df_list
    )

def get_burueau_df_features(bureau_balance_df, bureau_df):
    dpd_severe_df, dpd_mild_df = create_bureau_dpd_df(bureau_balance_df)
    bureau_preprocessed = bureau_df.merge(dpd_severe_df, on='SK_ID_BUREAU', how='left')
    bureau_preprocessed['severe_dpd_ratio'] = bureau_preprocessed['severe_dpd_ratio'].fillna(0)

    bureau_preprocessed = bureau_preprocessed.merge(dpd_mild_df, on='SK_ID_BUREAU', how='left')
    bureau_preprocessed['mild_dpd_ratio'] = bureau_preprocessed['mild_dpd_ratio'].fillna(0)

    num_cols = bureau_preprocessed.select_dtypes(include="number").drop(columns=["SK_ID_CURR", "SK_ID_BUREAU"]).columns.to_list()

    bureau_agg_df = (
        bureau_preprocessed.groupby("SK_ID_CURR").agg(
            {
                col: ["max", "sum", "mean", "std"] for col in num_cols
            }
        )
    )
    bureau_agg_df.columns = [
        f"{col}_{agg}" for col, agg in bureau_agg_df.columns
    ]
    bureau_agg_df.reset_index(inplace=True)
    bureau_agg_df
    col_list = bureau_preprocessed['CREDIT_TYPE'].value_counts().index.to_list()
    bureau_count_df = create_count_cols_cat(bureau_preprocessed,'CREDIT_TYPE', col_list)
    bureau_agg_df = bureau_agg_df.merge(bureau_count_df, on="SK_ID_CURR", how='left')

    active_counts = (
        bureau_df.groupby("SK_ID_CURR")["CREDIT_ACTIVE"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={
            "Active": "credit_active_active_count",
            "Closed": "credit_active_closed_count",
            "Sold": "credit_active_sold_count",
            "Bad debt": "credit_active_baddebt_count",
        })
        .reset_index()
    )
    bureau_agg_df = (
        bureau_agg_df
        .merge(active_counts, on='SK_ID_CURR', how='left')
    )
    bureau_agg_df["debt_to_credit_ratio"] = (
        (
            bureau_agg_df["AMT_CREDIT_SUM_DEBT_sum"] 
            / bureau_agg_df['AMT_CREDIT_SUM_sum']
        )
        .replace([np.inf, -np.inf], np.nan)
    )
    return bureau_agg_df

def get_previous_application_features(previous_application_df):
    previous_application_df["prev__application_to_credit_ratio"] = (
        (
            previous_application_df["AMT_APPLICATION"] 
            / previous_application_df["AMT_CREDIT"].fillna(0)
        ).replace([np.inf, -np.inf], np.nan)
    )
    previous_application_df["prev__annuity_to_credit_ratio"] = (
        (
            previous_application_df["AMT_ANNUITY"] 
            / previous_application_df["AMT_CREDIT"].fillna(0)
        ).replace([np.inf, -np.inf], np.nan)
    )
    previous_application_df["prev__down_payment_to_goods_price_ratio"] = (
        (
            previous_application_df["AMT_DOWN_PAYMENT"].fillna(0) 
            / previous_application_df["AMT_GOODS_PRICE"].fillna(0)
        ).replace([np.inf, -np.inf], np.nan)
    )
    previous_application_df["prev__down_payment_to_credit_ratio"] = (
        (
            previous_application_df["AMT_DOWN_PAYMENT"].fillna(0) 
            / previous_application_df["AMT_CREDIT"].fillna(0)
        ).replace([np.inf, -np.inf], np.nan)
    )
    ############################
    last_app_n = previous_application_df[
        previous_application_df['FLAG_LAST_APPL_PER_CONTRACT'] == 'N'
    ]
    last_app_y = previous_application_df[
        previous_application_df['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y'
    ]

    num_cols_y = (
        last_app_y
        .select_dtypes(include="number")
        .drop(columns=["SK_ID_CURR", "SK_ID_PREV"])
        .columns
        .to_list()
    )
    num_cols_n = (
        last_app_n
        .select_dtypes(include="number")
        .drop(columns=["SK_ID_CURR", "SK_ID_PREV"])
        .columns
        .to_list()
    )

    last_app_y_agg = (
        last_app_y.groupby("SK_ID_CURR").agg(
            {
                col: ["max", "sum", "mean", "std"] for col in num_cols_y
            }
        )
    )
    last_app_y_agg.columns = [
        f"last_app_y_{col}_{agg}" for col, agg in last_app_y_agg.columns
    ]

    last_app_n_agg = (
        last_app_n.groupby("SK_ID_CURR").agg(
            {
                col: ["max", "sum", "mean", "std"] for col in num_cols_n
            }
        )
    )
    last_app_n_agg.columns = [
        f"last_app_n_{col}_{agg}" for col, agg in last_app_n_agg.columns
    ]
    last_app_agg = last_app_n_agg.merge(
        last_app_y_agg,
        on='SK_ID_CURR',
        how='outer'
    )
    ############################
    num_cols = (
        previous_application_df
        .select_dtypes(include="number")
        .drop(columns=["SK_ID_CURR", "SK_ID_PREV"])
        .columns
        .to_list()
    )

    previous_application_agg = (
        previous_application_df.groupby("SK_ID_CURR").agg(
            {
                col: ["max", "sum", "mean", "std"] for col in num_cols
            }
        )
    )
    previous_application_agg.columns = [
        f"{col}_{agg}" for col, agg in previous_application_agg.columns
    ]
    previous_application_agg.reset_index(inplace=True)
    ###############################
    cat_cols_prev = previous_application_df.select_dtypes(exclude="number").columns
    prev_agg_df = pd.DataFrame(columns=["SK_ID_CURR"])
    for col in cat_cols_prev:
        tmp_df = create_count_cols_cat(
            previous_application_df, 
            col, 
            previous_application_df[col].value_counts().index.to_list()
        )
        prev_agg_df = prev_agg_df.merge(tmp_df, on='SK_ID_CURR', how='outer')

    previous_application_agg = (
        previous_application_agg
        .merge(prev_agg_df, on='SK_ID_CURR', how='outer')
        .merge(last_app_agg, on='SK_ID_CURR', how='outer')
    )

    ### by refused and approved aggs
    prev_aggs = {
        "AMT_APPLICATION":           ["min", "max", "mean", "sum"],
        "AMT_CREDIT":                ["min", "max", "mean", "sum"],
        "prev__application_to_credit_ratio":["mean", "max"],
        "prev__down_payment_to_credit_ratio": ["mean", 'max'],
        "prev__down_payment_to_goods_price_ratio": ["mean", 'max'],
        "prev__annuity_to_credit_ratio": ["mean", 'max'],
    }

    approved = previous_application_df[previous_application_df["NAME_CONTRACT_STATUS"] == "Approved"]
    refused  = previous_application_df[previous_application_df["NAME_CONTRACT_STATUS"] == "Refused"]

    prev_approved = agg_subgroup(approved, "SK_ID_CURR", prev_aggs, "prev_approved")
    prev_refused  = agg_subgroup(refused,  "SK_ID_CURR", prev_aggs, "prev_refused")

    previous_application_agg = previous_application_agg.merge(prev_approved, on='SK_ID_CURR', how='left')
    previous_application_agg = previous_application_agg.merge(prev_refused, on='SK_ID_CURR', how='left')

    return previous_application_agg

def get_pos_features(pos_cash_balance_df):

    pos_agg_dpd = get_pos_dpd_dfs(pos_cash_balance_df)
    pos_cash_balance_df = pos_cash_balance_df.merge(
        pos_agg_dpd, 
        on='SK_ID_PREV',
        how='left'
    )
    pos_cash_balance_df[pos_agg_dpd.columns] = (
        pos_cash_balance_df[pos_agg_dpd.columns]
        .fillna(0)
    )
    
    #Categorical Columns
    cat_cols_pos = (
        pos_cash_balance_df
        .select_dtypes(exclude="number")
        .columns
        .to_list()
    )
    pos_cat_agg_df = pd.DataFrame(columns=["SK_ID_CURR"])
    for col in cat_cols_pos:
        tmp_df = create_count_cols_cat(
            pos_cash_balance_df, 
            col,
            pos_cash_balance_df[col].value_counts().index.to_list()
        )
        pos_cat_agg_df = pos_cat_agg_df.merge(tmp_df, on='SK_ID_CURR', how='outer')
    ### groupby SK_ID_CURR and agg
    num_cols = (
        pos_cash_balance_df
        .select_dtypes(include="number")
        .drop(columns=["SK_ID_CURR", "SK_ID_PREV"])
        .columns
        .to_list()
    )

    pos_agg = (
        pos_cash_balance_df.groupby("SK_ID_CURR").agg(
            {
                col: ["max", "sum", "mean", "std"] for col in num_cols
            }
        )
    )
    pos_agg.columns = [
        f"{col}_{agg}" for col, agg in pos_agg.columns
    ]
    pos_agg.reset_index(inplace=True)
    pos_agg = (
        pos_agg.merge(pos_cat_agg_df, on='SK_ID_CURR', how='left')
    )
    return pos_agg

def get_credit_card_features(credit_card_balance_df):
    credit_card_balance_df = credit_card_balance_df.copy()
    #### Ratio features
    credit_card_balance_df["credit__Bal_creditLimit_ratio"] = (
        credit_card_balance_df["AMT_BALANCE"] 
        / credit_card_balance_df["AMT_CREDIT_LIMIT_ACTUAL"]
    ).replace([np.inf, -np.inf], np.nan)

    credit_card_balance_df["credit__payment_receivable_ratio"]  = (
        credit_card_balance_df["AMT_PAYMENT_CURRENT"]
        / credit_card_balance_df["AMT_TOTAL_RECEIVABLE"]
    ).replace([np.inf, -np.inf], np.nan)

    ### Previous n month credit card history
    prev_12_credit = credit_card_balance_df[credit_card_balance_df['MONTHS_BALANCE'] > -13]
    
    prev_12_credit['credit__pmt_recive_princ_12'] = (
        prev_12_credit['AMT_PAYMENT_CURRENT'] >= prev_12_credit['AMT_RECEIVABLE_PRINCIPAL']
    ).astype('int')
    
    prev_12_credit['credit__pmt_recive_12'] = (
        prev_12_credit['AMT_PAYMENT_CURRENT'] >= prev_12_credit['AMT_RECIVABLE']
    ).astype('int')

    prev_12_credit['credit__pmt_recive_total_12'] = (
        prev_12_credit['AMT_PAYMENT_CURRENT'] >= prev_12_credit['AMT_TOTAL_RECEIVABLE']
    ).astype('int')
    prev_12_credit_agg = prev_12_credit.groupby('SK_ID_CURR').agg(
        {
            'credit__pmt_recive_princ_12': ['mean', 'max', 'sum'],
            'credit__pmt_recive_12': ['mean', 'max', 'sum'],
            'credit__pmt_recive_total_12': ['mean', 'max', 'sum'],
        }
    )
    prev_12_credit_agg.columns = [f"{col1}_{col2}" for col1, col2 in prev_12_credit_agg.columns]
    prev_12_credit_agg

    prev_24_credit = credit_card_balance_df[credit_card_balance_df['MONTHS_BALANCE'] > -25]
    
    prev_24_credit['credit__pmt_recive_princ_24'] = (
        prev_24_credit['AMT_PAYMENT_CURRENT'] >= prev_24_credit['AMT_RECEIVABLE_PRINCIPAL']
    ).astype('int')

    prev_24_credit['credit__pmt_recive_24'] = (
        prev_24_credit['AMT_PAYMENT_CURRENT'] >= prev_24_credit['AMT_RECIVABLE']
    ).astype('int')
    
    prev_24_credit['credit__pmt_recive_total_24'] = (
        prev_24_credit['AMT_PAYMENT_CURRENT'] >= prev_24_credit['AMT_TOTAL_RECEIVABLE']
    ).astype('int')

    prev_24_credit_agg = prev_24_credit.groupby('SK_ID_CURR').agg(
        {
            'credit__pmt_recive_princ_24': ['mean', 'max', 'sum'],
            'credit__pmt_recive_24': ['mean', 'max', 'sum'],
            'credit__pmt_recive_total_24': ['mean', 'max', 'sum'],
        }
    )
    prev_24_credit_agg.columns = [f"{col1}_{col2}" for col1, col2 in prev_24_credit_agg.columns]

    ### Categorical columns
    credit_card_cols_pos = (
        credit_card_balance_df
        .select_dtypes(exclude="number")
        .columns
        .to_list()
    )
    credit_card_cat_agg_df = pd.DataFrame(columns=["SK_ID_CURR"])
    for col in credit_card_cols_pos:
        tmp_df = create_count_cols_cat(
            credit_card_balance_df,
            col,
            credit_card_balance_df[col].value_counts().index.to_list()
        )
        credit_card_cat_agg_df = (
            credit_card_cat_agg_df
            .merge(tmp_df, on='SK_ID_CURR', how='outer')
        )

    ### groupby SK_ID_CURR and agg
    num_cols = (
        credit_card_balance_df
        .select_dtypes(include="number")
        .columns
        .drop(["MONTHS_BALANCE", 'SK_ID_PREV'])
    )
    credit_card_agg = (
        credit_card_balance_df[num_cols]
        .groupby("SK_ID_CURR")
        .agg(['mean', 'max',  'sum', 'std'])
    )
    credit_card_agg.columns = ["_".join(col) for col in credit_card_agg.columns]
    credit_card_agg = (
        credit_card_agg
        .merge(credit_card_cat_agg_df, on='SK_ID_CURR', how='left')
        .merge(prev_24_credit_agg, on='SK_ID_CURR', how='left')
        .merge(prev_12_credit_agg, on='SK_ID_CURR', how='left')
    )
    return credit_card_agg

def get_installments_features(installments_payments_df):
    ### ratios
    installments_payments_df["installment__payment_ratio"] = (
        (
            installments_payments_df["AMT_PAYMENT"].fillna(0) 
            / installments_payments_df["AMT_INSTALMENT"]
        ).replace([np.inf, -np.inf], np.nan)
    )
    installments_payments_df["installment__good_pay_ind"] = (
        (
            installments_payments_df["installment__payment_ratio"] >= 1
        ).astype(int).replace([np.inf, -np.inf], np.nan)
    )
    installments_payments_df["installment__pay_to_actual_pay"] = (
        (
            installments_payments_df["DAYS_ENTRY_PAYMENT"].fillna(0) 
            / installments_payments_df["DAYS_INSTALMENT"]
        ).replace([np.inf, -np.inf], np.nan)
    )
    installments_payments_df["installment__good_pay_day_ind"] = (
        (
            installments_payments_df["installment__pay_to_actual_pay"] >= 1
        ).astype(int).replace([np.inf, -np.inf], np.nan)
    )
    ### late indicator for installments
    installments_payments_df['installments__late_ind'] = (
        installments_payments_df['DAYS_ENTRY_PAYMENT'] > installments_payments_df['DAYS_INSTALMENT']
    ).astype('int')

    ### payment behaviour in the last n months
    recency_cols, recent_behaviour_df = get_installments_last_n_month_stat(installments_payments_df, n_months=[12, 24, 36])
    
    ### Categorical Columns
    installments_cols_pos = installments_payments_df.select_dtypes(exclude="number").columns.to_list()
    installments_cat_agg_df = pd.DataFrame(columns=["SK_ID_CURR"])
    for col in installments_cols_pos:
        tmp_df = create_count_cols_cat(installments_payments_df, col, installments_payments_df[col].value_counts().index.to_list())
        installments_cat_agg_df = installments_cat_agg_df.merge(tmp_df, on='SK_ID_CURR', how='outer')
    installments_cat_agg_df

    ### groupby SK_ID_CURR and agg
    num_cols = installments_payments_df.select_dtypes(include="number").columns.drop(["SK_ID_PREV"])
    installments_payments_agg = (
        installments_payments_df[num_cols]
        .groupby("SK_ID_CURR")
        .agg(['mean', 'max', 'std', 'sum'])
    )
    installments_payments_agg.columns = ["_".join(col) for col in installments_payments_agg.columns]

    # Merge recent risk
    installments_payments_agg = installments_payments_agg.merge(
        recent_behaviour_df, 
        on='SK_ID_CURR',
        how='outer'
    )
    ### fill recency risk for IDs that have no risk (not in the recency_df) with 0
    installments_payments_agg[recency_cols] = installments_payments_agg[recency_cols].fillna(0)

    return installments_payments_agg

def get_installments_last_n_month_stat(df, n_months=(12, 24, 36)):
    df = df.copy()
    df_list = []
    new_cols = []
    for month_ in n_months:
        last_year_installments = df[df['DAYS_INSTALMENT'] > -(month_*30)]
        last_year_installments[f'installments__is_pmt_late_past_{month_}'] = (
            (last_year_installments['DAYS_ENTRY_PAYMENT'] > last_year_installments['DAYS_INSTALMENT']
            ).astype(int)
        )
        last_year_installments[f'installments__is_pmt_adequate_{month_}'] = (
            last_year_installments["installment__payment_ratio"] >= 1
        ).astype(int)
        last_year_installments[f'installments__is_pmt_good_{month_}'] = (
            (last_year_installments["installment__payment_ratio"] >= 1)
            & (last_year_installments["DAYS_ENTRY_PAYMENT"] <= last_year_installments["DAYS_INSTALMENT"])
        ).astype(int)
        clean_cols = [
            'SK_ID_CURR',
            "SK_ID_PREV",
            f'installments__is_pmt_late_past_{month_}', 
            f'installments__is_pmt_adequate_{month_}', 
            f'installments__is_pmt_good_{month_}'
        ]
        new_cols.extend(
            [
            f'installments__is_pmt_late_past_{month_}', 
            f'installments__is_pmt_adequate_{month_}', 
            f'installments__is_pmt_good_{month_}'
        ]
        )
        clean_df = last_year_installments[clean_cols]

        count_df = clean_df.groupby("SK_ID_CURR").sum()
        df_list.append(count_df)
    agg_df = reduce(
        lambda left, right: left.drop("SK_ID_PREV", axis=1).merge(right, on=('SK_ID_CURR'), how="outer"),
        df_list
    )
    agg_df.fillna(0, inplace=True)
    
    return new_cols, agg_df

def change_json_compatible_col_names(df):
    df = df.copy()
    df = df.rename(columns=lambda c: re.sub(r'[",:\[\]{}]', '', str(c)))
    return df
