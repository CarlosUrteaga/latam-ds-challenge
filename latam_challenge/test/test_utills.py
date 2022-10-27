import pandas as pd
import pytest
from datetime import datetime
from latam_challenge.helpers.utils import ( convert_str_to_date, convert_str_to_time, split_date, add_temporada_alta_flag, 
                                            add_periodo_dia_flag, add_dif_min_and_atraso_15, generate_temporada_alta_set,
                                            generate_day_section
                                          )
from  latam_challenge.helpers.constants import ( FECHAO, FECHAI, DATE, TIME, STR_WINTER_START, STR_WINTER_END, TEMPORADA_ALTA, 
                                                  STR_MANANA_START, STR_MANANA_END, STR_TARDE_START, STR_TARDE_END, PERIODO_DIA,
                                                  ATRASO_15, DIF_MIN, )

data = {
    FECHAI: ["2017-12-15 23:30:00", "2017-02-02 3:35:00", "2017-03-03 12:40:00", "2017-04-04 11:45:00"],
    FECHAO: ["2017-12-15 23:31:00", "2017-02-02 4:36:00", "2017-03-03 12:47:00", "2017-04-04 11:48:00"],
}

# dataframe from dict
df = pd.DataFrame.from_dict(data)

def test_convert_str_to_date():
    str_date = '2017-05-04'
    expected_date = datetime.strptime(str_date, '%Y-%m-%d').date()
    calculated_date = convert_str_to_date(str_date)
    assert calculated_date == expected_date

def test_convert_str_to_time():
    str_date = '23:59'
    expected_date = datetime.strptime(str_date, '%H:%M').time()
    calculated_date = convert_str_to_time(str_date)
    assert calculated_date == expected_date

@pytest.mark.parametrize("df", [df])
def test_split_date(df):
    df1 = df.copy()
    df1 = split_date(df1, FECHAI)
    expected_str = "2017-12-15"
    expected_str = datetime.strptime(expected_str, '%Y-%m-%d').date()
    assert DATE in df1.columns
    assert TIME in df1.columns
    assert df1.loc[0][DATE] == expected_str


@pytest.mark.parametrize("df", [df])
def test_add_temporada_alta_flag(df):
    str_winter_start = convert_str_to_date(STR_WINTER_START)
    str_winter_end = convert_str_to_date(STR_WINTER_END)
    dates = [[str_winter_start, str_winter_end]]
    df1 = df.copy()
    df1 = split_date(df1, FECHAI)
    df_output = add_temporada_alta_flag(df1, dates)
    assert TEMPORADA_ALTA in df1.columns
    assert df1.loc[0][TEMPORADA_ALTA] == 1


@pytest.mark.parametrize("df", [df])
def test_add_periodo_dia_flag(df):
    str_mañana_start = convert_str_to_time(STR_MANANA_START)
    str_mañana_end = convert_str_to_time(STR_MANANA_END)
    str_tarde_start = convert_str_to_time(STR_TARDE_START)
    str_tarde_end = convert_str_to_time(STR_TARDE_END)
    times = [[str_mañana_start, str_mañana_end],
            [str_tarde_start, str_tarde_end]]

    df1 = df.copy()
    df1 = split_date(df1, FECHAI)
    df1 = add_periodo_dia_flag(df1, times)
    assert PERIODO_DIA in df1.columns
    assert df1.loc[0][PERIODO_DIA] == 'noche'
    assert df1.loc[3][PERIODO_DIA] == 'mañana'

@pytest.mark.parametrize("df", [df])
def test_add_dif_min_and_atraso_15(df):
    df1 = df.copy()
    df1[FECHAO] = pd.to_datetime(df1[FECHAO])
    df1[FECHAI] = pd.to_datetime(df1[FECHAI])
    df1 = add_dif_min_and_atraso_15(df1)
    assert DIF_MIN in df1.columns
    assert ATRASO_15 in df1.columns
    assert df1.loc[0][DIF_MIN] == 1
    assert df1.loc[0][ATRASO_15] == 0
    assert df1.loc[1][DIF_MIN] == 61
    assert df1.loc[1][ATRASO_15] == 1

def test_generate_temporada_alta_set():
    assert 4 == len(generate_temporada_alta_set())

def test_generate_day_section():
    assert 2 == len(generate_day_section())