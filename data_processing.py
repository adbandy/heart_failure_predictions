# data available from
# https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download

"""
Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression] 'The ST segment is a portion of the electrocardiogram (ECG) waveform that represents the time between ventricular depolarization (contraction) and repolarization (relaxation).'
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease: output class [1: heart disease, 0: Normal]
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import seaborn as sns # aiaiaiaaiiiaiai sir adeian
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')
def process(df):
    # convert male/female to 1/0:
    male_female_binary_dict = {i:j for i,j in zip(['M','F'],[0,1])}
    df['Sex'] = df['Sex'].apply(lambda x : male_female_binary_dict[x])
    # convert ExcerciseAngina Y/N to 1/0:
    excercise_angina_binary_dict = {i:j for i,j in zip(['Y','N'],[0,1])}
    df['ExerciseAngina'] = df['ExerciseAngina'].apply(lambda x : excercise_angina_binary_dict[x])
    return df

def print_df_statistics(df):
    print(f'df shape:{df.shape}\n')
    print(f'df describe:\n{df.describe()}\n')
    print(f'df null value count:\n{df.isna().sum().sum()}\n')
    print(f'df info:\n{df.info()}\n')
    for col in df.columns:
        print(f'\ncol {col} has {df[col].nunique()}_unique_values')
        print(f'Top 5 values of :{df[col].value_counts().iloc[:5]}')



def plot_pearson_correlation_heatmap(df, save_fig=False, save_fig_path=''):
    fig, ax = plt.subplots(figsize = (10,10))
    numeric_df = df.select_dtypes(include=np.number)
    g= sns.heatmap(numeric_df.corr(), ax=ax, cmap = 'viridis', annot=True, fmt='.2g')
    g.set_yticklabels(g.get_yticklabels(), rotation=45)
    plt.show()
    if save_fig:
        plt.savefig(save_fig_path)

def plot_relationships_and_distribution(df):
    fig, ax = plt.subplots(figsize=(20, 20))
    g = sns.pairplot(df)
    plt.show()

def plot_categorical_correlations(df):
    fig = plt.figure(figsize=(20, 20))
    non_numeric_df = df.select_dtypes(exclude=np.number)
    non_numeric_df = pd.concat([non_numeric_df, df['HeartDisease']], axis=1)
    print(non_numeric_df.columns)
    column_len = len(non_numeric_df.columns)
    for index, col in enumerate(non_numeric_df.columns):
        cols = (column_len // 2) + 1
        rows = 2
        ax = plt.subplot(rows, cols, index+1)
        sns.countplot(non_numeric_df, x=col,
                      # y=non_numeric_df[col].value_counts().values,
                      ax=ax, hue='HeartDisease')
    plt.show()

# print_df_statistics(process(df))
# print_df_statistics(df)
# plot_pearson_correlation_heatmap(df)
'''
heart disease 
- positvely correlated with Age and Oldpeak
- negatively correlated with MaxHR 
All correlations are weak (-0.6 to +0.6)  
'''
# plot_relationships_and_distribution(df)
'''
Cholesterol column appears to have outlier values with values of zero. Suspect this is default value when Cholesterol is unknown.
OldPeak has no normal distribution with the highest frequency at zero OldPeak. Suspect this is default value when OldPeak is unknown.

Can potentially put these numeric values into bands with pd.cut
'''
plot_categorical_correlations(df)
'''
Trends. Heart Disease is prevalent in:
Sex: Male,
ChestPainType: 'ASY',
RestingECH: 'ST', 
ExcerciseAngina: 'Y',
ST_Slope; 'Flat'
'''
