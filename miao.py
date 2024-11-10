import math
import yaml
from typing import List, Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from fracdiff import fdiff
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.svar_model import SVAR

def parse_miao_db_yml(path: str):
    with open(path, 'r') as f:
        miao_db = yaml.safe_load(f)
    return miao_db

def unit_root_test(series: pd.Series):
    results = adfuller(series, regression='c')
    adf = results[0]
    pvalue = results[1]
    
    return adf, pvalue

def to_stationary_process_recursive(s: pd.Series, retry_count = 0, init_state = None):
    significance_level = 0.05
    n = 0.1 * retry_count

    adf, p = unit_root_test(s)

    if init_state is None:
        init_state = (adf, p)

    if p >= significance_level:
        s_ = fdiff(s, n)

        adf, p = unit_root_test(s_)
        if p >= significance_level:
            return to_stationary_process_recursive(s, retry_count + 1, init_state)
        else:
            return init_state[0], init_state[1], adf, p, n, s_
    
    return init_state[0], init_state[1], None , None, n, s

def split_dates(start, end, years):
    dates = []
    while start.year < end.year:
        next_year = start.year + years
        next_year = min(next_year, end.year)
        next_year = pd.Timestamp(next_year, start.month, start.day)
        dates.append({
            "start": start,
            "end": next_year
        })
        start = next_year
    
    return dates

def auto_detect_tms(db, N_MONTHS = 12, limit_y = 4):
    n_dormant_commit = 10
    anual_days = 365.25

    end_idx = None
    start_dates = []
    end_dates = []
    target_df = None

    data = [{"name": db["name"]}] + db["competitors"]

    for row in data:   
        repo = row["name"]
        df = pd.read_csv(f"./datasets/original/monthly/{repo}.csv", index_col=0)
        df.index = [pd.Timestamp(idx) for idx in df.index]

        if target_df is None:
            target_df = df

        start_date = pd.Timestamp(row["appearance_time"]) if "appearance_time" in row else df.index[0]
        start_dates.append(start_date)
        end_dates.append(df.index[-1])

    start_dates = list(sorted(start_dates, key=lambda d: d.timestamp(), reverse=True))
    
    start_date = start_dates[0]
    
    df_after_start = target_df[target_df.index > pd.Timestamp(start_date.year, start_date.month, start_date.day)]

    years = []
    n_months = df_after_start.index
    n_loop = math.ceil(len(n_months)/N_MONTHS)
    for i in range(n_loop):
        ms = n_months[i*N_MONTHS:i*N_MONTHS+N_MONTHS]
        years.append(ms[0].year)
    
    means = []
    take = N_MONTHS
    n_commits = list(df_after_start.n_commits)
    loop = math.ceil(len(n_commits)/take)
    for i in range(loop):
        sub_commits = n_commits[i*take:i*take+take]
        means.append(np.mean(sub_commits))
    
    for i, m in enumerate(means):
        if m <= n_dormant_commit:
            break

    end_idx = i
    end_year = years[end_idx]

    if start_date.year >= end_year:
        end_year = start_date.year + 1

    if end_year - start_date.year == 1:
        end_year += 1

    tm_start = pd.Timestamp(start_date.year, start_date.month, start_date.day)
    tm_end = pd.Timestamp(end_year, tm_start.month, tm_start.day)

    sub = target_df[target_df.index < tm_end]
    commits = sub.n_commits
    tail_commits = commits.tail(10).to_numpy()
    
    for i, c in enumerate(tail_commits):
        if c < n_dormant_commit:
            break

    subtract_month = 10 - (i + 1)
    tm_end -= pd.DateOffset(months=subtract_month)
    
    num_y = math.floor((tm_end - tm_start).days/anual_days)
    remaining_days = (tm_end - tm_start).days - (limit_y*anual_days)
    diff = remaining_days/anual_days
    should_split = diff >= 0.5 or num_y > limit_y

    tms = []
    if should_split:
        tms = split_dates(tm_start, tm_end, limit_y)
    else:
        y = (tm_end - tm_start).days/anual_days
        if y <= limit_y: 
            tms = [{
                "start": tm_start,
                "end": tm_end
            }]
        else:
            tm_start = tm_end - pd.DateOffset(years=limit_y)
            tms = [{
                "start": tm_start,
                "end": tm_end
            }]

    end_dates = list(sorted(end_dates, key=lambda d: d.timestamp(), reverse=False))
    min_end_date = end_dates[0]

    if len(tms) > 1:
        last_tm = tms[len(tms)-1]
        if last_tm["end"] > min_end_date:
            last_tm["end"] = min_end_date

        last_tm = tms[-1]
        y = math.floor((last_tm["end"] - last_tm["start"]).days/anual_days)
        if y < limit_y:
            last_tm["start"] = last_tm["end"] - pd.DateOffset(years=limit_y)

    return tms    

def create_Tms(start, end, interval, split_interval = False):
    current = start
    result = []
    
    while current < end:
        next_year = current + pd.DateOffset(months=interval)
        next_year_end = min(next_year, end)
        if split_interval:
            result.append([current, next_year_end - pd.DateOffset(days=1)])
        else:
            end_ = next_year_end - pd.DateOffset(days=1)

            if len(result) >= 1:
                if (end_ - result[-1][1]).days == 1:
                    break

            result.append([start, end_])
        current = next_year_end

    return result

def subframe(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, date_col: str = 'ym', expand_end_date: bool = False):
    start_idx = df[df[date_col] == start_date].index[0]

    try:
        end_idx = df[df[date_col] == end_date].index[0]
    except Exception as e:
        if expand_end_date:
            last_date = df[date_col].iloc[-1]
            diff = (end_date - last_date).days

            other_cols = df.columns.difference([date_col])

            additional_data = {}
            for col in other_cols:
                additional_data[col] = pd.NA

            additional_data[date_col] = []
            for i in range(diff):
                last_date += pd.DateOffset(days=1)
                additional_data[date_col].append(last_date)

            additional_df = pd.DataFrame(additional_data)
            df = pd.concat([df, additional_df], ignore_index=True)
            end_idx = df[df[date_col] == end_date].index[0]
        else:
            raise e

    sub_df = df.iloc[start_idx:end_idx+1].reset_index(drop=True)
    sub_df.attrs['idx'] = {
        'start': start_idx,
        'end': end_idx
    }
    sub_df.attrs['date'] = {
        'start': start_date,
        'end': end_date        
    }
    return sub_df

def create_var_x_and_find_optimal_lag(data: pd.DataFrame, maxlag: int, ic: str):
    def find_optimal_item(histories):
        min_ic = float('inf')
        optimal_item = None
        significance_level = 0.1

        for item in histories:
            ic_value, _, _, test_result, _ = item
            pvalue = test_result.pvalue

            # lag=0は使わない
            if ic_value == 0:
                continue

            if pvalue > significance_level:
                return item
            
            if ic_value < min_ic:
                min_ic = ic_value
                optimal_item = item

        return optimal_item
    
    def make_histories(model):
        ics = model.select_order(maxlag).ics[ic]
        ics_idxs = np.argsort(ics)

        histories = []
        for lag in ics_idxs:
            # 0は使わない
            if lag == 0:
                lag = 1

            r, whiteness_test_lag = whiteness_test(model, lag)
            
            ic_map = {}
            for name in model.select_order(maxlag).ics:
                ic_vals = model.select_order(maxlag).ics[name]
                ic_map[name] = ic_vals[lag]

            histories.append((ics[lag], lag, ic_map, r, whiteness_test_lag))

        return histories
    
    VAR_X = pd.DataFrame(data)
    model = VAR(VAR_X)
    histories = make_histories(model)
    _, lag, ics, test_result, whiteness_test_lag = find_optimal_item(histories)

    return VAR_X, lag, ics, test_result, whiteness_test_lag, model
    
def weak_seasonal_adjusment(series, period=7):
    decomposed = sm.tsa.seasonal_decompose(series, period=period, model='additive')
    return series - decomposed.seasonal
    
def strong_seasonal_adjusment(series, period=7):
    stl = STL(series, robust=True, period=period)
    res = stl.fit()
    return series - res.seasonal

def whiteness_test(model, lag):
    if type(model) == SVAR:
        results = model.fit(maxlags=lag)
    else:
        results = model.fit(lag)
    whiteness_lags = 10
    if lag >= whiteness_lags:
        whiteness_lags = lag*2

    r = results.test_whiteness(nlags=whiteness_lags, signif=0.1)

    return r, whiteness_lags

def prepare_var_(data: List[Tuple[str, pd.Series]], maxlag, ic="aic", verbose = True):
    var_data = {}
    for name, X in data:
        var_data[name] = X

    weak_seasonaled_data = {}
    strong_seasonaled_data = {}

    weak_adf_test_results = {}
    strong_adf_test_results = {}

    for name in var_data:
        s = var_data[name].copy()

        try:
            ws = weak_seasonal_adjusment(s, period=7)
            adf, p, fracdif_adf, fracdif_p, n, ws = to_stationary_process_recursive(ws)
            weak_adf_test_results[name] = (adf, p, fracdif_adf, fracdif_p, n)
            weak_seasonaled_data[name] = ws

            ss = strong_seasonal_adjusment(s, period=7)
            adf, p, fracdif_adf, fracdif_p, n, ss = to_stationary_process_recursive(ss)
            strong_adf_test_results[name] = (adf, p, fracdif_adf, fracdif_p, n)
            strong_seasonaled_data[name] = ss
        except Exception as e:
            print(f"Failed to adjust seasonality: {name}")
            print(s)
            raise e

    weak_var = create_var_x_and_find_optimal_lag(weak_seasonaled_data, maxlag, ic)
    weak_var_x = weak_var[0]
    weak_var_lag = weak_var[1]
    weak_var_ics = weak_var[2]
    weak_whiteness_test_result = weak_var[3]
    weak_whiteness_test_lag = weak_var[4]

    strong_var = create_var_x_and_find_optimal_lag(strong_seasonaled_data, maxlag, ic)
    strong_var_x = strong_var[0]
    strong_var_lag = strong_var[1]
    strong_var_ics = strong_var[2]
    strong_whiteness_test_result = strong_var[3]
    strong_whiteness_test_lag = strong_var[4]

    rv = [
        (weak_var_lag, weak_var_x),
        (strong_var_lag, strong_var_x)
    ]
    ics = [
        weak_var_ics,
        strong_var_ics
    ]
    test_results = [
        weak_whiteness_test_result,
        strong_whiteness_test_result
    ]
    whiteness_test_lags = [
        weak_whiteness_test_lag,
        strong_whiteness_test_lag
    ]

    idx = np.argmax([p.pvalue for p in test_results])
    max_rp = test_results[idx].pvalue

    lag = rv[idx][0]
    VAR_X = rv[idx][1]

    adf_test_results = weak_adf_test_results if idx == 0 else strong_adf_test_results
    adf_ps = [val[3] if val[3] is not None else val[1] for val in adf_test_results.values()]
    adf_p = ", ".join(['{:.3f}'.format(p) for p in adf_ps])

    if verbose:
        if max_rp > 0.1:
            print(f"Failed to reject the H0 of whiteness test: pvalue={max_rp}, lag={lag}, adf_p={adf_p}")
        else:
            print(f"Rejected the H0 of whiteness test: pvalue={max_rp}, lag={lag}, adf_p={adf_p}")

    VAR_X.attrs["whiteness_test_result"] = test_results[idx]
    VAR_X.attrs["whiteness_test_lag"] =  whiteness_test_lags[idx]
    VAR_X.attrs["ics"] = ics[idx]
    VAR_X.attrs["adf_test_results"] = adf_test_results

    return lag, VAR_X

def prepare_var(tms, As, verbose = True):
    VAR_Xs = []

    for tm in tms:
        data = []
        for name, A in As:
            start, end = tm
            
            A_ = pd.DataFrame({
                "date": [pd.Timestamp(idx) for idx in A.index],
                "n_commits": A.n_commits.values
            })
            A_ = subframe(A_, start, end, date_col="date", expand_end_date=True)
            data.append((name, A_.n_commits.values))

        lag, VAR_X = prepare_var_(data, maxlag=15, ic="aic", verbose=verbose)
        VAR_X.index = A_.date
        VAR_Xs.append((lag, VAR_X))

    return VAR_Xs
