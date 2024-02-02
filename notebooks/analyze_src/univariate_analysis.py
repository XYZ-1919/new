import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import List


import geopandas as gpd
import folium

from shapely.geometry import Point


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, features: List[str]):
        pass

class NumericalAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, features: List[str]):
        for feature in features:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Enhanced Box Plot with outlier percentage
            sns.boxplot(data=df, y=feature, ax=ax1, color='skyblue')
            
            # Calculate outlier percentage
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))][feature]
            outlier_pct = (len(outliers) / len(df)) * 100
            
            ax1.set_title(f'Distribution of {feature}\nOutliers: {outlier_pct:.1f}%')
            
            # Enhanced KDE Plot with mean and median lines
            sns.histplot(data=df[feature], kde=True, ax=ax2, color='purple', alpha=0.6)
            ax2.axvline(df[feature].mean(), color='red', linestyle='--', label='Mean')
            ax2.axvline(df[feature].median(), color='green', linestyle='--', label='Median')
            ax2.legend()
            
            plt.show()
            
            # Print insights
            print(f"\nInsights for {feature}:")
            print(f"Mean: {df[feature].mean():.2f}")
            print(f"Median: {df[feature].median():.2f}")
            print(f"Mode: {df[feature].mode().iloc[0]}")
            print(f"Standard Deviation: {df[feature].std():.2f}")
            print(f"Skewness: {df[feature].skew():.2f}")

class CategoricalAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, features: List[str]):
        pie_features = ['City', 'Type_of_order', 'Type_of_vehicle']
        count_features = ['Festival', 'Weatherconditions']
        stack_features = ['multiple_deliveries', 'Road_traffic_density']
        
        for feature in features:
            if feature in pie_features:
                plt.figure(figsize=(10, 8))
                df[feature].value_counts().plot.pie(autopct='%1.1f%%', 
                                                  colors=sns.color_palette('Set3'),
                                                  explode=[0.05]*len(df[feature].unique()))
                plt.title(f'Distribution of {feature}')
                plt.show()
            
            elif feature in count_features:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=df, x=feature, palette='viridis')
                plt.title(f'Count Plot of {feature}')
                plt.xticks(rotation=45)
                plt.show()
            
            elif feature in stack_features:
                plt.figure(figsize=(10, 6))
                df[feature].value_counts(normalize=True).plot(kind='bar',
                                                            stacked=True,
                                                            color=sns.color_palette('husl'))
                plt.title(f'Distribution of {feature}')
                plt.ylabel('Percentage')
                plt.show()

class TimeSeriesAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, features: List[str]):
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
        # Enhanced daily trend with rolling average
        daily_orders = df.groupby('Order_Date').size()
        plt.figure(figsize=(15, 6))
        daily_orders.plot(alpha=0.5, label='Daily Orders')
        daily_orders.rolling(7).mean().plot(color='red', linewidth=2, 
                                          label='7-day Moving Average')
        plt.title('Daily Order Trend with Weekly Moving Average', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Enhanced hourly distribution by day type
        df['Hour'] = df['Order_Date'].dt.hour
        df['Day_Type'] = df['Order_Date'].dt.dayofweek.map(
            lambda x: 'Weekend' if x >= 5 else 'Weekday')
        
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=df, x='Hour', hue='Day_Type', fill=True)
        plt.title('Hourly Order Distribution: Weekday vs Weekend', fontsize=14)
        plt.xlabel('Hour of Day')
        plt.ylabel('Density')
        plt.show()

        # Print key insights
        print("\nTime Series Insights:")
        print(f"Peak ordering hour: {df.groupby('Hour').size().idxmax()}")
        print(f"Busiest day of week: {df['Order_Date'].dt.day_name().mode()[0]}")

class AdvancedGeospatialAnalysisStrategy:
    def analyze(self, df: pd.DataFrame, features: List[str]):
        # Convert DataFrame to GeoDataFrame
        geometry_restaurant = [Point(xy) for xy in zip(df['Restaurant_longitude'], df['Restaurant_latitude'])]
        geometry_delivery = [Point(xy) for xy in zip(df['Delivery_location_longitude'], df['Delivery_location_latitude'])]
        
        gdf_restaurant = gpd.GeoDataFrame(
            df, 
            geometry=geometry_restaurant, 
            crs="EPSG:4326"
        )
        
        gdf_delivery = gpd.GeoDataFrame(
            df, 
            geometry=geometry_delivery, 
            crs="EPSG:4326"
        )
        
        # Spatial Distribution Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Restaurant Locations Plot
        gdf_restaurant.plot(
            column='City', 
            cmap='viridis', 
            legend=True,
            alpha=0.6,
            ax=ax1
        )
        ax1.set_title('Restaurant Location Distribution')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # Delivery Locations Plot
        gdf_delivery.plot(
            column='City', 
            cmap='plasma', 
            legend=True,
            alpha=0.6,
            ax=ax2
        )
        ax2.set_title('Delivery Location Distribution')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        plt.tight_layout()
        plt.show()
        
        # Interactive Map Creation
        center_lat = df['Restaurant_latitude'].mean()
        center_lon = df['Restaurant_longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add markers
        for idx, row in df.iterrows():
            if idx % 2 == 0:
                folium.CircleMarker(
                    location=[row['Restaurant_latitude'], row['Restaurant_longitude']],
                    radius=5,
                    popup=f"Restaurant in {row['City']}",
                    color='green',
                    fill=True,
                    fillColor='green'
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[row['Delivery_location_latitude'], row['Delivery_location_longitude']],
                    radius=5,
                    popup=f"Delivery in {row['City']}",
                    color='blue',
                    fill=True,
                    fillColor='blue'
                ).add_to(m)
        
        # Save the interactive map
        m.save("delivery_locations_map.html")
        
        # Print spatial statistics
        print("\nSpatial Statistics:")
        print("\nRestaurant Location Extent:")
        print(f"Latitude Range: {df['Restaurant_latitude'].min()} to {df['Restaurant_latitude'].max()}")
        print(f"Longitude Range: {df['Restaurant_longitude'].min()} to {df['Restaurant_longitude'].max()}")
        
        print("\nDelivery Location Extent:")
        print(f"Latitude Range: {df['Delivery_location_latitude'].min()} to {df['Delivery_location_latitude'].max()}")
        print(f"Longitude Range: {df['Delivery_location_longitude'].min()} to {df['Delivery_location_longitude'].max()}")
        
        # Spatial clustering
        from sklearn.cluster import DBSCAN
        
        X_restaurant = df[['Restaurant_latitude', 'Restaurant_longitude']].values
        clustering_restaurant = DBSCAN(eps=0.1, min_samples=5).fit(X_restaurant)
        df['Restaurant_Cluster'] = clustering_restaurant.labels_
        
        X_delivery = df[['Delivery_location_latitude', 'Delivery_location_longitude']].values
        clustering_delivery = DBSCAN(eps=0.1, min_samples=5).fit(X_delivery)
        df['Delivery_Cluster'] = clustering_delivery.labels_
        
        print("\nRestaurant Location Clustering:")
        print(df['Restaurant_Cluster'].value_counts())
        print("\nDelivery Location Clustering:")
        print(df['Delivery_Cluster'].value_counts())

class UnivariateAnalyzer:
    def __init__(self, df: pd.DataFrame):
        # Convert types appropriately
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd']).dt.time
        df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked']).dt.time
        df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'])
        df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'])
        df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'])
        
        # Drop ID columns
        df = df.drop(columns=['ID', 'Delivery_person_ID'], errors='ignore')
        self.df = df
        
        self.numerical_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 
                                 'Vehicle_condition', 'Time_taken(min)']
        self.categorical_features = ['City', 'Type_of_order', 'Type_of_vehicle', 
                                   'Festival', 'Weatherconditions', 'Road_traffic_density']
        
        self.strategies = {
            'numerical': NumericalAnalysisStrategy(),
            'categorical': CategoricalAnalysisStrategy(),
            'time_series': TimeSeriesAnalysisStrategy(),
            'geospatial': AdvancedGeospatialAnalysisStrategy()
        }
    
    def analyze(self):
        # Numerical features
        self.strategies['numerical'].analyze(self.df, self.numerical_features)
        
        # Categorical features
        self.strategies['categorical'].analyze(self.df, self.categorical_features)
        
        # Time series features
        self.strategies['time_series'].analyze(self.df, ['Order_Date', 'Time_Orderd'])
        
        # Geospatial features
        geospatial_features = ['Restaurant_latitude', 'Restaurant_longitude',
                              'Delivery_location_latitude', 'Delivery_location_longitude']
        self.strategies['geospatial'].analyze(self.df, geospatial_features)

# Example usage:
if __name__ == "__main__":
    # df = pd.read_csv('your_data.csv')
    # analyzer = UnivariateAnalyzer(df)
    # analyzer.analyze()
    pass