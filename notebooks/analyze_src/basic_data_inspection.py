from abc import ABC, abstractmethod

import pandas as pd


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass



class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())



class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


class SkewnessInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Calculates and prints skewness for numerical columns, with interpretation.
        It skips datetime and geospatial columns.
        """
        print("\nSkewness Analysis for Numerical Features:")
        numerical_cols = df.select_dtypes(include=['int64', 'float64'])
        datetime_cols = df.select_dtypes(include=['datetime64[ns]'])
        geo_cols = [col for col in numerical_cols if 'latitude' in col.lower() or 'longitude' in col.lower()]

        # Remove datetime and geospatial columns from numerical_cols
        numerical_cols = numerical_cols.drop(columns=datetime_cols.columns, errors='ignore')
        numerical_cols = numerical_cols.drop(columns=geo_cols, errors='ignore')

        if numerical_cols.empty:
            print("No suitable numerical columns found for skewness analysis.")
        else:
            skewness = numerical_cols.skew()
            print("\nSkewness values:")
            for col, skew_value in skewness.items():
                print(f"{col}: {skew_value:.3f}")
                if skew_value > 1:
                    print(f"  {col} is highly positively skewed.")
                elif skew_value > 0.5:
                    print(f"  {col} is moderately positively skewed.")
                elif skew_value < -1:
                    print(f"  {col} is highly negatively skewed.")
                elif skew_value < -0.5:
                    print(f"  {col} is moderately negatively skewed.")
                else:
                    print(f"  {col} is approximately symmetrically distributed.")




class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)




if __name__ == "__main__":

    pass
