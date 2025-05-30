import math
import yaml
from typing import List, Tuple, Dict

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

def get_unique_idx(db):
    name = db["name"]
    osss = [name] + [c["name"] for c in db["competitors"]]
    return gen_idx_(osss)

def gen_idx_(oss_names):
    return ",".join([oss.split("/")[1] for oss in oss_names])

def auto_detect_tms(db, N_MONTHS = 12, limit_y = 4, daily_dataset_path = "./datasets/original/daily", monthly_dataset_path = "./datasets/original/monthly"):
    n_dormant_commit = 10
    anual_days = 365.25

    end_idx = None
    start_dates = []
    end_dates = []
    target_df = None

    data = [{"name": db["name"]}] + db["competitors"]

    for row in data:   
        repo = row["name"]
        df_monthly = pd.read_csv(f"{monthly_dataset_path}/{repo}.csv", index_col=0)
        df_monthly.index = [pd.Timestamp(idx) for idx in df_monthly.index]

        df_daily = pd.read_csv(f"{daily_dataset_path}/{repo}.csv", index_col=0)
        df_daily.index = [pd.Timestamp(idx) for idx in df_daily.index]

        if target_df is None:
            target_df = df_monthly

        start_date = pd.Timestamp(row["appearance_time"]) if "appearance_time" in row else df_daily.index[0]
        start_dates.append(start_date)
        end_dates.append(df_monthly.index[-1])    

        start_dates = list(sorted(start_dates, key=lambda d: d.timestamp(), reverse=True))
        start_date: pd.Timestamp = start_dates[0]        

    # if db["rev"] == True:
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

    
    if "end_date" in db:
        tm_end = pd.Timestamp(db["end_date"])
    else:
        tm_end = pd.Timestamp(end_year, tm_start.month, tm_start.day)

        sub = target_df[target_df.index < tm_end]
        commits = sub.n_commits
        tail_commits = commits.tail(10).to_numpy()
        
        for i, c in enumerate(tail_commits):
            if c < n_dormant_commit:
                break

        subtract_month = 10 - (i + 1)
        tm_end -= pd.DateOffset(months=subtract_month)
    # else:
    #     tm_start = pd.Timestamp(start_date.year, start_date.month, start_date.day)
    #     tm_end = np.min(end_dates)

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

def find_optimal_lag(X: pd.DataFrame, maxlag: int, ic: str):
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
    
    model = VAR(X)
    histories = make_histories(model)
    _, lag, ics, test_result, whiteness_test_lag = find_optimal_item(histories)

    return lag, ics, test_result, whiteness_test_lag
    
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
        
    weak_var_x = pd.DataFrame(weak_seasonaled_data)
    weak_var_lag, weak_var_ics, weak_whiteness_test_result, weak_whiteness_test_lag = find_optimal_lag(weak_var_x, maxlag, ic)

    strong_var_x = pd.DataFrame(strong_seasonaled_data)
    strong_var_lag, strong_var_ics, strong_whiteness_test_result, strong_whiteness_test_lag = find_optimal_lag(strong_var_x, maxlag, ic)

    rv = [(weak_var_lag, weak_var_x), (strong_var_lag, strong_var_x)]
    ics = [weak_var_ics, strong_var_ics]
    test_results = [weak_whiteness_test_result, strong_whiteness_test_result]
    whiteness_test_lags = [weak_whiteness_test_lag, strong_whiteness_test_lag]

    idx = np.argmax([p.pvalue for p in test_results])
    lag = rv[idx][0]
    VAR_X = rv[idx][1]
    max_rp = test_results[idx].pvalue

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


def miao_score(sces: List[float]) -> float:
    """
    Calculate MS_{ij} from list of SCE
    """
    return (-1)*np.sum(sces)

def miao_phase1_with_period_shifts(A_mat: np.ndarray, shifts: List[str], Tms: List[str], group_with_sep: str,  permutation_id: int, dataset_path: str = "./", irf_k=200) -> List[Dict[str, List[float]]]:
    """
    This function performs a MIAO Phase 1 and outputs the SCE (Shock Cumulative Effect) for $A_i -> A_j$. 
    Note that PREPARE_VAR has already been completed, and the data preprocessing and optimal lag orders use pre-calculated results.
    
    In addition to the standard algorithm, period shifts are taken into consideration.
    """

    def generate_irf_pair(labels: List[str]) -> Dict[str, List[str]]:
        result = {}
    
        for i in range(len(labels)):
            for j in range(len(labels)):
                key = "{}x{}".format(i, j)
                value = [labels[j], labels[i]]
                result[key] = value
                
        return result
    
    sces_with_shifts = []
    structural_covs = []
    is_stables = []
    var_result = pd.read_csv(f"{dataset_path}/var_estimation_results/permute_{permutation_id}.csv")
    
    for shift in shifts:
        sces = {}
        for Tm in Tms:
            X = pd.read_csv(f"{dataset_path}/datasets/preprocessed/permute_{permutation_id}/{shift}/{Tm}/{group_with_sep}.csv", index_col=0)
            pair = generate_irf_pair(X.columns)

            cond = (var_result["group"] == group_with_sep) & (var_result["Tm"] == Tm) & (var_result["period_shift"] == shift)
            lag = var_result[cond].iloc[0].lag
            
            model = SVAR(X, svar_type="A", A=A_mat)
            results = model.fit(maxlags=lag)

            # Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the eigenvalues of the companion matrix must lie outside the unit circle
            is_stable = results.is_stable()
            is_stables.append(is_stable)

            # 方法2: SVAR構造ショックの共分散行列
            # Σₑ = A⁻¹BB'(A⁻¹)'
            A_inv = np.linalg.inv(results.A)
            B_mat = results.B
            structural_cov = A_inv @ B_mat @ B_mat.T @ A_inv.T
            structural_covs.append(structural_cov)
    
            irf = results.irf(irf_k)
    
            n = len(X.columns)
    
            for from_ in range(n):
                for to_ in range(n):
                    if from_ == to_:
                        continue
                    
                    left, right = pair[f"{from_}x{to_}"]                
                    label = f"{left} -> {right}"
                    if label not in sces:
                        sces[label] = []
                    
                    sce = irf.svar_cum_effects[-1, from_, to_]
                    sces[label].append(sce)
    
        sces_with_shifts.append(sces)

    return sces_with_shifts, structural_covs, is_stables

def miao_phase2_with_period_shifts(sces_with_shifts: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calculate AMS_{ij} and construct the MIAO score table. 
    """
    
    ams = {}

    for sces in sces_with_shifts:
        for key in sces:
            if key not in ams:
                ams[key] = 0
    
            ms = miao_score(sces[key])
            ams[key] += np.sum(ms)
    
    table = pd.DataFrame({
        "AMS_ij": [v/len(sces_with_shifts) for v in ams.values()]
    }, index=ams.keys())

    return table