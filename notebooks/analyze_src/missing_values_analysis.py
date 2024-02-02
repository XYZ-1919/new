from abc import ABC, abstractmethod
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the count of missing values for each column.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: This method should create a visualization (e.g., heatmap) of missing values.
        """
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console.
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        cmap = mcolors.ListedColormap(['pink', 'blue'])
        fig, ax = plt.subplots(figsize=(12, 8))
        print("\nVisualizing Missing Values...")
        sns.heatmap(df.isnull(), cmap=cmap, cbar=False, ax=ax)
        plt.title("Missing Values Heatmap")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the SimpleMissingValuesAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Missing Values Analysis
    # missing_values_analyzer = SimpleMissingValuesAnalysis()
    # missing_values_analyzer.analyze(df)
    pass
