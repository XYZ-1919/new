from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using enhanced scatter plots with regression lines and hexbin plots for density.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays enhanced scatter plots with regression lines and hexbin plots for density showing the relationship between the two features.
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Enhanced scatter plot with regression line
        sns.regplot(data=df, x=feature1, y=feature2, ax=ax1,
                   scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
        ax1.set_title(f'Relationship between {feature1} and {feature2}')
        
        # Calculate correlation and add it to plot
        corr = df[feature1].corr(df[feature2])
        ax1.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax1.transAxes, bbox=dict(facecolor='white'))
        
        # Enhanced hexbin plot for density
        hb = ax2.hexbin(df[feature1], df[feature2], gridsize=20, cmap='YlOrRd')
        ax2.set_title('Density of Observations')
        plt.colorbar(hb, ax=ax2)
        
        plt.tight_layout()
        plt.show()

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using violin plots with box plots inside and provides statistical insights.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays violin plots with box plots inside and prints statistical insights.
        """
        # Create violin plot with box plot inside
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x=feature1, y=feature2, inner='box')
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {feature2} across {feature1}')
        
        # Add statistical insights
        print(f"\nStatistical Summary for {feature2} by {feature1}:")
        stats = df.groupby(feature1)[feature2].agg(['mean', 'median', 'std']).round(2)
        print(stats)
        
        # Perform ANOVA test if applicable
        from scipy import stats as scipy_stats
        groups = [group for _, group in df.groupby(feature1)[feature2]]
        f_stat, p_val = scipy_stats.f_oneway(*groups)
        print(f"\nANOVA Test Results:")
        print(f"F-statistic: {f_stat:.2f}")
        print(f"p-value: {p_val:.4f}")
        
        plt.show()


class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature1, feature2)


# Example usage
if __name__ == "__main__":
  
    pass
