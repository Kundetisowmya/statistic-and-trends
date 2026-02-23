"""
This script analyzes global education data, focusing on the share of
the population with formal education. It generates visualizations and
calculates the four major statistical moments for education rates.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_relational_plot(df):
    """
    Creates a relational scatter plot to examine the progress of
    formal education rates over the years.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Year', y='Education_Rate', alpha=0.4)
    plt.title('Global Education Progress: Formal Education Over Time')
    plt.xlabel('Year')
    plt.ylabel('Formal Education Rate (%)')
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Creates a bar plot showing average formal education rates for the
    top 10 entities (countries) in the dataset.
    """
    plt.figure(figsize=(12, 6))
    top_entities = df.groupby('Entity')['Education_Rate'].mean()
    top_entities = top_entities.sort_values(ascending=False).head(10)

    sns.barplot(
        x=top_entities.index,
        y=top_entities.values,
        hue=top_entities.index,
        palette='viridis',
        legend=False
    )
    plt.title('Top 10 Entities by Average Formal Education Rate')
    plt.xlabel('Entity (Country)')
    plt.ylabel('Average Education Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Creates a violin plot to visualize the density of education rates
    for a sample of major global entities.
    """
    plt.figure(figsize=(10, 6))
    # Using a sample of entities as 'Region' is missing in the CSV
    sample_entities = ['Albania', 'Algeria', 'Argentina', 'Brazil', 'China']
    df_subset = df[df['Entity'].isin(sample_entities)]

    sns.violinplot(data=df_subset, x='Entity', y='Education_Rate')
    plt.title('Statistical Density of Education Rates by Entity')
    plt.xlabel('Entity')
    plt.ylabel('Formal Education Rate (%)')
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Calculates Mean, Standard Deviation, Skewness, and Excess Kurtosis
    for the specified column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Cleans data, renames long columns, and provides initial overview.
    """
    # Identify the long column name from your logs and rename it
    old_name = 'Share of population with some formal education, 1820-2020'
    df = df.rename(columns={old_name: 'Education_Rate'})

    print("--- Dataset Head ---")
    print(df.head())

    # Drop missing values to ensure statistical validity
    df = df.dropna(subset=['Education_Rate', 'Entity', 'Year'])

    return df


def writing(moments, col):
    """
    Interprets the statistical moments for the education distribution.
    """
    mean, stddev, skew, excess_kurtosis = moments

    print(f'\nFor the attribute {col}:')
    print(f'Mean = {mean:.2f}, '
          f'Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, and '
          f'Excess Kurtosis = {excess_kurtosis:.2f}.')

    # Distribution interpretation logic
    skew_type = "not skewed"
    if skew > 0.5:
        skew_type = "right skewed"
    elif skew < -0.5:
        skew_type = "left skewed"

    kurt_type = "mesokurtic"
    if excess_kurtosis > 1:
        kurt_type = "leptokurtic"
    elif excess_kurtosis < -1:
        kurt_type = "platykurtic"

    print(f'The data was {skew_type} and {kurt_type}.')
    return


def main():
    """
    Main execution pipeline for education data analysis.
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'Education_Rate'

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
