from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
# This class defines a template for performing multivariate analysis.
# Subclasses can override specific steps like correlation heatmap and pair plot generation.
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a correlation heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display a pair plot of the selected features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a pair plot.
        """
        pass


# Concrete Class for Multivariate Analysis with Correlation Heatmap and Pair Plot
# -------------------------------------------------------------------------------
# This class implements the methods to generate a correlation heatmap and a pair plot.
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()


# Specialized Class for Food Delivery Multivariate Analysis
# ----------------------------------------------------------
# This class implements methods tailored for analyzing food delivery datasets.
class FoodDeliveryMultivariateAnalysis(MultivariateAnalysisTemplate):
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.key_metrics = [
            'Time_taken(min)',
            'Delivery_person_Age',
            'Delivery_person_Ratings',
            'Vehicle_condition',
            'multiple_deliveries'
        ]

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data before analysis"""
        df = df.copy()
        
        # Convert numeric columns properly
        numeric_cols = {
            'Delivery_person_Age': 'float64',
            'Delivery_person_Ratings': 'float64',
            'Time_taken(min)': 'int64',
            'multiple_deliveries': 'float64',
            'Vehicle_condition': 'int64'
        }
        
        for col, dtype in numeric_cols.items():
            if col in df.columns:
                # Handle string extraction for Time_taken(min)
                if col == 'Time_taken(min)':
                    df[col] = df[col].str.extract('(\d+)').astype(dtype)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        
        return df

    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """Generate correlation heatmap for numerical features"""
        plt.figure(figsize=(10, 8))
        correlation = df[self.key_metrics].corr()
        
        mask = np.triu(correlation)
        sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Matrix of Key Delivery Metrics', fontsize=14)
        
        # Print significant insights
        print("\nKey Correlations:")
        for i in range(len(correlation.columns)):
            for j in range(i):
                corr = correlation.iloc[i, j]
                if abs(corr) > 0.2:
                    print(f"{correlation.columns[i]} vs {correlation.columns[j]}: {corr:.2f}")
        
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """Generate enhanced pairplot with key delivery metrics"""
        df = self.preprocess_data(df)
        available_features = [f for f in self.key_metrics if f in df.columns]
        
        plot_df = df[available_features].dropna()
        if len(plot_df) == 0:
            print("No valid data remaining after removing missing values")
            return
            
        g = sns.pairplot(
            plot_df,
            diag_kind="kde",
            plot_kws={'alpha': 0.6},
            height=2.5
        )
        g.fig.suptitle("Relationships between Key Delivery Metrics", y=1.02, fontsize=16)
        plt.show()

    def analyze_delivery_efficiency(self, df: pd.DataFrame):
        """Analyze delivery efficiency by traffic and vehicle condition"""
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(data=df, x='Road_traffic_density', y='Time_taken(min)',
                   hue='Vehicle_condition')
        plt.title('Delivery Time by Traffic and Vehicle Condition', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add insights
        print("\nDelivery Efficiency Insights:")
        for traffic in df['Road_traffic_density'].unique():
            mean_time = df[df['Road_traffic_density']==traffic]['Time_taken(min)'].mean()
            print(f"{traffic} traffic: Average {mean_time:.1f} minutes")
            
        plt.show()

    def analyze_delivery_patterns(self, df: pd.DataFrame):
        """Analyze delivery patterns by traffic and weather"""
        required_cols = ['Road_traffic_density', 'Time_taken(min)', 
                        'Weatherconditions', 'Type_of_vehicle', 
                        'Delivery_person_Ratings', 'multiple_deliveries']
        
        # Check if required columns exist
        if not all(col in df.columns for col in required_cols):
            print("Missing required columns for delivery pattern analysis")
            return
            
        plt.figure(figsize=(15, 6))
        
        # Time taken vs Traffic Density by Weather
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df.dropna(subset=['Road_traffic_density', 'Time_taken(min)', 
                                         'Weatherconditions']),
                   x='Road_traffic_density', y='Time_taken(min)', 
                   hue='Weatherconditions')
        plt.title('Delivery Time by Traffic and Weather')
        plt.xticks(rotation=45)
        
        # Ratings Distribution by Vehicle Type
        plt.subplot(1, 2, 2)
        sns.violinplot(data=df.dropna(subset=['Type_of_vehicle', 'Delivery_person_Ratings', 
                                            'multiple_deliveries']),
                      x='Type_of_vehicle', y='Delivery_person_Ratings',
                      hue='multiple_deliveries')
        plt.title('Ratings Distribution by Vehicle Type')
        plt.tight_layout()
        plt.show()

    def analyze_performance_metrics(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 6))
        
        # Performance analysis
        plt.subplot(1, 2, 1)
        pivot_data = df.pivot_table(
            values='Time_taken(min)',
            index='Vehicle_condition',
            columns='Road_traffic_density',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Average Delivery Time by Traffic and Vehicle Condition')
        
        # Ratings analysis
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='Type_of_vehicle', y='Delivery_person_Ratings',
                   hue='multiple_deliveries')
        plt.title('Ratings Distribution by Vehicle Type and Multiple Deliveries')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print key insights
        print("\nKey Performance Insights:")
        print(f"Average delivery time: {df['Time_taken(min)'].mean():.1f} minutes")
        print(f"Average rating: {df['Delivery_person_Ratings'].mean():.2f}")
        print("\nCorrelation between ratings and delivery time:",
              f"{df['Delivery_person_Ratings'].corr(df['Time_taken(min)']):.2f}")


# Example usage
if __name__ == "__main__":
    # Example usage with the food delivery dataset
    analyzer = FoodDeliveryMultivariateAnalysis()
    
    # Load your DataFrame here
    #df = pd.read_csv(r'C:\Users\91786\Desktop\DS\food_ds\data\extracted\train.csv')
    
    # Perform analysis
    #analyzer.analyze(df)
    #analyzer.analyze_delivery_efficiency(df)
    #analyzer.analyze_delivery_patterns(df)
