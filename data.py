import pandas as pd


raw_data = pd.read_csv('data/covid_19_data.csv')
raw_data['ObservationDate'] = pd.to_datetime(raw_data['ObservationDate'], format='%m/%d/%Y')
time_series_per_country = raw_data.groupby(['Country/Region', 'ObservationDate']).agg(
    {'Confirmed':sum, 'Deaths': sum, 'Recovered': sum}
)


def get_SIR_data(country_name, n):
    filtered_df = time_series_per_country.iloc[time_series_per_country.index.get_level_values('Country/Region') == country_name]
    filtered_df = filtered_df.reset_index(level=0, drop=True) # remove country's name from index
    filtered_df['R'] = (filtered_df.pop('Deaths') + filtered_df.pop('Recovered'))
    filtered_df['I'] = filtered_df.pop('Confirmed') - filtered_df['R']
    filtered_df['S'] = n - filtered_df['I'] - filtered_df['R']
    return filtered_df


def get_SIRD_data(country_name, n):
    filtered_df = time_series_per_country.iloc[time_series_per_country.index.get_level_values('Country/Region') == country_name]
    filtered_df = filtered_df.reset_index(level=0, drop=True) # remove country's name from index
    filtered_df['D'] = filtered_df.pop('Deaths')
    filtered_df['R'] = filtered_df.pop('Recovered')
    filtered_df['I'] = filtered_df.pop('Confirmed') - filtered_df['R'] - filtered_df['D']
    filtered_df['S'] = n - filtered_df['I'] - filtered_df['R'] - filtered_df['D']
    return filtered_df
