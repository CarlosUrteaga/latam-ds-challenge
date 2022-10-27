from datetime import datetime
from typing import List
from pandas import DataFrame
import pandas as pd
from latam_challenge.helpers.constants import ( TEMPORADA_ALTA, DATE, PERIODO_DIA, TIME, NIGHT, AFTERNOON, MORNING, 
                                                DIF_MIN, ATRASO_15, FECHAI, FECHAO,  STR_WINTER_START, 
                                                STR_WINTER_END, STR_WINTER_BIS_START, STR_WINTER_BIS_END, 
                                                STR_JULY_START, STR_JULY_END, STR_SEPT_START, STR_SEPT_END, 
                                                STR_MANANA_START, STR_MANANA_END, STR_TARDE_START, STR_TARDE_END,
                                                VLOO, ORIO, DESO, EMPO, VLOI, DIANOM, OPERA, SIGLADES, COLUMN_FILTER,
                                                TIPOVUELO, ANTERIOR_DELAY, ANTERIOR_EARLY)

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

def convert_str_to_date(str_date: str) -> datetime:
    """convert string to date format '%Y-%m-%d'

    Args:
        str_date (str): string ex: '2017-01-01'

    Returns:
        datetime: string with datetime type
    """
    return datetime.strptime(str_date, '%Y-%m-%d').date()

def convert_str_to_time(str_date: str) -> datetime:
    """convert string to time format '%H:%M'

    Args:
        str_date (str): string '23:59'

    Returns:
        datetime: string with datetime type
    """
    return datetime.strptime(str_date, '%H:%M').time()

def split_date(df:DataFrame, column_name:str)-> DataFrame:
    """_summary_

    Args:
        df (DataFrame): _description_
        column_name (str): _description_

    Returns:
        DataFrame: _description_
    """
    df[column_name] = pd.to_datetime(df[column_name])
    df[DATE] = df[column_name].dt.date
    df[TIME] = df[column_name].dt.time
    return df.copy()

def add_temporada_alta_flag(df: DataFrame, dates: List) -> DataFrame:
    """Add temporada alta flag according to array of dates

    Args:
        df (DataFrame): _description_
        dates (List): _description_

    Returns:
        DataFrame: _description_
    """
    df[TEMPORADA_ALTA] = 0
    for date in dates:
        df.loc[(df[DATE] >= date[0]) & (df[DATE]<=date[1]), TEMPORADA_ALTA] = 1
    return df.copy()
def add_periodo_dia_flag(df: DataFrame, times: List) -> DataFrame:
    """Add temporada alta flag according to array of dates

    Args:
        df (DataFrame): _description_
        dates (List): _description_

    Returns:
        DataFrame: _description_
    """
    # default
    df[PERIODO_DIA] = NIGHT
    df.loc[(df[TIME] >= times[0][0]) & (df[TIME]<=times[0][1]), PERIODO_DIA] = MORNING
    df.loc[(df[TIME] >= times[1][0]) & (df[TIME]<=times[1][1]), PERIODO_DIA] = AFTERNOON
    return df.copy()

def add_dif_min_and_atraso_15(df:DataFrame) ->DataFrame:
    df[DIF_MIN] = df[FECHAO]-df[FECHAI]
    df[DIF_MIN] = df[DIF_MIN].dt.total_seconds()//60
    df[ATRASO_15] = 0
    df.loc[df[DIF_MIN]>15, ATRASO_15] = 1
    return df

def generate_temporada_alta_set()-> list:
    """
    This function generate arrays to generate temporada_alta elements, source from constant
    dates ahre the following, 
        15-Dic y 3-Mar, o 15-Jul y 31-Jul, o 11-Sep y 30-Sep,
    output: array of element, with the following 
    """
    str_winter_start = convert_str_to_date(STR_WINTER_START)
    str_winter_end = convert_str_to_date(STR_WINTER_END)
    str_winter_bis_start = convert_str_to_date(STR_WINTER_BIS_START)
    str_winter_bis_end = convert_str_to_date(STR_WINTER_BIS_END)
    str_July_start = convert_str_to_date(STR_JULY_START)
    str_July_end = convert_str_to_date(STR_JULY_END)
    str_Sept_start = convert_str_to_date(STR_SEPT_START)
    str_Sept_end = convert_str_to_date(STR_SEPT_END)

    dates = [[str_winter_start, str_winter_end],
             [str_winter_bis_start, str_winter_bis_end],
             [str_July_start, str_July_end],
             [str_Sept_start, str_Sept_end]]
    return dates

def generate_day_section():
    """
    split a day iin three part, morning, afternoon and night
    """
    str_mañana_start = convert_str_to_time(STR_MANANA_START)
    str_mañana_end = convert_str_to_time(STR_MANANA_END)
    str_tarde_start = convert_str_to_time(STR_TARDE_START)
    str_tarde_end = convert_str_to_time(STR_TARDE_END)
    times = [[str_mañana_start, str_mañana_end],
             [str_tarde_start, str_tarde_end],
            ]
    return times

# """"

def delete_operational_columns(df:DataFrame)-> DataFrame:
    del df[FECHAO]
    del df[VLOO]
    del df[ORIO]
    del df[DESO]
    del df[EMPO]
    return df

def filter_column(df:DataFrame, columns:str =  COLUMN_FILTER)->DataFrame:
    return df[columns]


def previous_flight (df: DataFrame)-> DataFrame:
    DELAY = 'delay'
    EARLY = 'early'
    df.sort_values(FECHAI, inplace = True)
    df[EARLY] = df[DIF_MIN].shift(1)
    df[ANTERIOR_EARLY] = 0
    df.loc[df[EARLY]<0, ANTERIOR_EARLY] = 1
    df.loc[df[ANTERIOR_EARLY].isna(), ANTERIOR_EARLY] = 0
    df[DELAY] = df[DIF_MIN].shift(1)
    df[ANTERIOR_DELAY] = 0
    df.loc[df[DELAY]>0, ANTERIOR_DELAY] = 1
    del df[DELAY]
    del df[EARLY]
    return df

def number_of_flights_before (df):

    df.sort_values(FECHAI, inplace = True)
    dft = df.groupby([FECHAI])[FECHAI].count().rename("col_tmp")
    dft = dft.reset_index()
    dft["number_of_flights_before"] = dft["col_tmp"].shift(1)
    del dft['col_tmp']
    df = pd.merge(df,dft)
    df.loc[df['number_of_flights_before'].isna(), 'number_of_flights_before'] = 0

    return df


def number_of_flights_same_opera(df):
    dft = df.groupby([FECHAI, 'OPERA'])[FECHAI].count().rename("number_of_flights_same_opera")
    dft = dft.reset_index()
    dft['number_of_flights_same_opera'] = dft['number_of_flights_same_opera'] -1 
    df = pd.merge(df,dft)
    df.loc[df['number_of_flights_same_opera'].isna(), 'number_of_flights_same_opera'] = 0
    return df

def number_of_flights_same_dest(df):
    dft = df.groupby([FECHAI, 'SIGLADES'])[FECHAI].count().rename("number_of_flights_same_dest")
    dft = dft.reset_index()
    dft['number_of_flights_same_dest'] = dft['number_of_flights_same_dest'] -1 
    df = pd.merge(df,dft)
    df.loc[df['number_of_flights_same_dest'].isna(), 'number_of_flights_same_dest'] = 0
    return df

def one_hot_encoder(df:DataFrame)-> DataFrame:
    
    df = pd.get_dummies(df, columns = [OPERA])
    df = pd.get_dummies(df, columns = [SIGLADES])
    df = pd.get_dummies(df, columns = [TEMPORADA_ALTA]) 
    df = pd.get_dummies(df, columns = [PERIODO_DIA]) 
    df = pd.get_dummies(df, columns = [VLOI])
    df = pd.get_dummies(df, columns = [DIANOM])
    return df 


def to_bool_tipo_vuelo(df:DataFrame)-> DataFrame:
    df[f"{TIPOVUELO}_"] = df[TIPOVUELO]
    df.loc[df[f"{TIPOVUELO}_"]== 'I', TIPOVUELO] = 1
    df.loc[df[f"{TIPOVUELO}_"]== 'N', TIPOVUELO] = 0
    del df[f"{TIPOVUELO}_"]
    return df

def fit_grid_model(
    rfc: RandomForestClassifier,
    param_grid,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
):  # pragma: no cover
    """GridSeach

    Args:
        rfc (RandomForestClassifier): ML Model
        param_grid (dict): JSON parameters
        X_train (pd.DataFrame): train df
        y_train (pd.DataFrame): goal df
        model_type (str): model name

    Returns:
        [type]: model trained
    """
    model = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=3,
    )
    model.fit(X_train, y_train)
    return model

def fit_compute_importance(clf, X, y, X_test, y_test, k = 4):
    kf = KFold(n_splits=4)
    kf.get_n_splits(X)
    for i in range(0,5):
        print(f"iteration: {i}")
        for train_index, test_index in kf.split(X):
            X_train_, X_test_ = X.iloc[train_index], X.iloc[test_index]
            y_train_, y_test_ = y.iloc[train_index], y.iloc[test_index]
            clf.fit(X_train_, y_train_)
            y_pred = clf.predict(X_test_)
    
        y_pred = clf.predict(X_test)
        print(f"f1 score test set: {f1_score(y_test, y_pred, zero_division=1)*100:.2f}%\n")