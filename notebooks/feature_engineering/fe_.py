import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import re
from geopy.distance import geodesic
from abc import ABC, abstractmethod



class FeatureTransformer(ABC):
    """Base abstract class for all feature transformers"""
    
    @abstractmethod
    def transform(self, df):
        """Transform the dataframe and return a new dataframe with additional features"""
        pass
    
    def _copy_dataframe(self, df):
        """Create a copy of the dataframe to avoid modifying the original"""
        return df.copy()


class DataCleaner(FeatureTransformer):
    """Handles basic data cleaning and type conversions"""
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Convert string columns to proper types
        df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
        df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
        
        # Extract number from 'multiple_deliveries'
        df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')
        
        # Clean Time_taken(min) - extract the numeric value
        if 'Time_taken(min)' in df.columns:
            df['Time_taken(min)'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)
        
        # Weather conditions - extract just the condition without prefix
        df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '')
        
        return df


class DistanceStrategy(ABC):
    """Abstract strategy for calculating distances"""
    @abstractmethod
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        pass

class GeodesicDistanceStrategy(DistanceStrategy):
    """Calculates geodesic distance between two points"""
    def calculate_distance(self, point1, point2):
        return geodesic(point1, point2).km

class EuclideanDistanceStrategy(DistanceStrategy):
    """Calculates straight-line Euclidean distance between two points"""
    def calculate_distance(self, point1, point2):
        return np.sqrt(
            (point1[0] - point2[0])**2 + 
            (point1[1] - point2[1])**2
        ) * 111  # Rough conversion to km (1 degree ≈ 111 km)

class GeospatialFeatureTransformer(FeatureTransformer):
    """Creates features based on geospatial data"""
    
    def __init__(self, distance_strategy: DistanceStrategy = None):
        self.distance_strategy = distance_strategy or GeodesicDistanceStrategy()
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Calculate delivery distance (km)
        df['Distance_km'] = df.apply(self._calculate_distance, axis=1)
        
        # Extract city from ID if not already present
        if 'City' not in df.columns or df['City'].isna().any():
            df['City_from_ID'] = df['Delivery_person_ID'].str.extract(r'([A-Z]+)')
        
        return df
    
    def _calculate_distance(self, row):
        """Calculate distance between restaurant and delivery location using strategy"""
        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
        return self.distance_strategy.calculate_distance(restaurant, delivery)


class TimeFeatureTransformer(FeatureTransformer):
    """Creates time-based features from datetime fields"""
    
    def __init__(self, holiday_years=None):
        self.holiday_years = holiday_years or [2021, 2022, 2023]
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Convert to datetime types
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
        
        # Process time fields
        df = self._standardize_time_fields(df)
        
        # Create datetime objects
        df = self._create_datetime_objects(df)
        
        # Extract time components
        df = self._extract_time_components(df)
        
        # Add time period categorization
        df = self._add_time_periods(df)
        
        # Add cyclic encoding
        df = self._add_cyclic_encodings(df)
        
        # Add calendar features
        df = self._add_calendar_features(df)
        
        # Add holiday features
        df = self._add_holiday_features(df)
        
        # Add interaction features
        df = self._add_interaction_features(df)
        
        return df
    
    def _standardize_time_fields(self, df):
        """Standardize time fields to consistent format"""
        for col in ['Time_Orderd', 'Time_Order_picked']:
            if col in df.columns:
                # Handle various time formats and standardize
                df[col] = df[col].astype(str).fillna('00:00')
                # Extract HH:MM portion from various formats
                df[col] = df[col].apply(lambda x: re.search(r'(\d{1,2}:\d{2})', str(x)).group(1) 
                                      if re.search(r'(\d{1,2}:\d{2})', str(x)) else '00:00')
                # Add leading zero if needed
                df[col] = df[col].apply(lambda x: f"0{x}" if len(x) < 5 else x)
        return df
    
    def _create_datetime_objects(self, df):
        """Create full datetime objects by combining date and time"""
        date_str = df['Order_Date'].dt.strftime('%Y-%m-%d')
        
        df['Order_Datetime'] = pd.to_datetime(date_str + ' ' + df['Time_Orderd'], errors='coerce')
        df['Pickup_Datetime'] = pd.to_datetime(date_str + ' ' + df['Time_Order_picked'], errors='coerce')
        
        # Handle midnight crossover
        midnight_crossover = df['Pickup_Datetime'] < df['Order_Datetime']
        df.loc[midnight_crossover, 'Pickup_Datetime'] = df.loc[midnight_crossover, 'Pickup_Datetime'] + pd.Timedelta(days=1)
        
        # Calculate preparation time in minutes
        df['Preparation_Time_Mins'] = (df['Pickup_Datetime'] - df['Order_Datetime']).dt.total_seconds() / 60
        
        return df
    
    def _extract_time_components(self, df):
        """Extract various time components as features"""
        df['Hour'] = df['Order_Datetime'].dt.hour
        df['Minute'] = df['Order_Datetime'].dt.minute
        df['Day'] = df['Order_Datetime'].dt.day
        df['Month'] = df['Order_Datetime'].dt.month
        df['Year'] = df['Order_Datetime'].dt.year
        df['DayOfWeek'] = df['Order_Datetime'].dt.dayofweek
        df['DayOfYear'] = df['Order_Datetime'].dt.dayofyear
        df['Quarter'] = df['Order_Datetime'].dt.quarter
        df['WeekOfYear'] = df['Order_Datetime'].dt.isocalendar().week
        
        return df
    
    
    def _add_time_periods(self, df):
        """Add categorical time periods and rush hour indicators. Handles NA values."""
        print("     Adding time period features...")
        
        # Time period categorization
        time_bins = [-1, 5, 9, 11, 13, 16, 19, 22, 23]
        time_labels = ['Late_Night', 'Early_Morning', 'Late_Morning', 'Lunch',
                    'Afternoon', 'Evening', 'Dinner', 'Night']
        df['Time_Period'] = pd.cut(df['Hour'], bins=time_bins, labels=time_labels)
        
        # Special time indicators - handle NA values by filling with 0 (conservative approach)
        df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).fillna(False).astype(int)
        
        # Create boolean masks and handle NAs
        lunch_rush = df['Hour'].between(11, 13, inclusive='both').fillna(False)
        dinner_rush = df['Hour'].between(18, 21, inclusive='both').fillna(False)
        breakfast_time = df['Hour'].between(7, 10, inclusive='both').fillna(False)
        late_night = ((df['Hour'] >= 22) | (df['Hour'] <= 4)).fillna(False)
        
        # Convert to integer indicators
        df['Is_Lunch_Rush'] = lunch_rush.astype(int)
        df['Is_Dinner_Rush'] = dinner_rush.astype(int)
        df['Is_Rush_Hour'] = (lunch_rush | dinner_rush).astype(int)
        df['Is_Breakfast_Order'] = breakfast_time.astype(int)
        df['Is_Late_Night_Order'] = late_night.astype(int)
        
        return df
    
    def _add_cyclic_encodings(self, df):
        """Add sine and cosine transformations for cyclical features"""
        for col, max_val in [('Hour', 24), ('DayOfWeek', 7), ('Month', 12), ('Day', 31)]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
        return df
    
    def _add_calendar_features(self, df):
        """Add calendar-based features"""
        df['Is_Month_Start'] = df['Order_Datetime'].dt.is_month_start.astype(int)
        df['Is_Month_End'] = df['Order_Datetime'].dt.is_month_end.astype(int)
        df['Is_Quarter_Start'] = df['Order_Datetime'].dt.is_quarter_start.astype(int)
        df['Is_Quarter_End'] = df['Order_Datetime'].dt.is_quarter_end.astype(int)
        df['Days_In_Month'] = df['Order_Datetime'].dt.days_in_month
        
        # Order recency (days from most recent date in the dataset)
        df['Order_Recency_Days'] = (df['Order_Date'].max() - df['Order_Date']).dt.days
        
        return df
    
    def _add_holiday_features(self, df):
        """Add holiday-related features"""
        try:
            indian_holidays = holidays.India(years=self.holiday_years)
            df['Is_Holiday'] = df['Order_Date'].dt.date.map(lambda x: 1 if x in indian_holidays else 0)
            df['Next_Day_Holiday'] = df['Order_Date'].dt.date.map(
                lambda x: 1 if (x + timedelta(days=1)) in indian_holidays else 0
            )
        except Exception as e:
            print(f"Warning: Holiday features could not be added. Error: {e}")
        
        return df
    
    def _add_interaction_features(self, df):
        """Add interaction features between time and other variables"""
        if 'Is_Holiday' in df.columns:
            df['Holiday_Rush_Hour'] = df['Is_Holiday'] * df['Is_Rush_Hour']
        
        df['Rush_Hour_Weekend'] = df['Is_Rush_Hour'] * df['Is_Weekend']
        
        return df


class DeliveryPersonFeatureTransformer(FeatureTransformer):
    """Creates features related to the delivery person"""
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Create age categories
        if 'Delivery_person_Age' in df.columns:
            df['Age_Group'] = pd.cut(
                df['Delivery_person_Age'],
                bins=[0, 25, 35, 45, 100],
                labels=['Young', 'Adult', 'Middle_Aged', 'Senior']
            )
        
        # Create rating categories
        if 'Delivery_person_Ratings' in df.columns:
            df['Rating_Category'] = pd.cut(
                df['Delivery_person_Ratings'],
                bins=[0, 3.5, 4.0, 4.5, 5.0],
                labels=['Low', 'Medium', 'High', 'Excellent']
            )
        
        # Extract ID information
        if 'Delivery_person_ID' in df.columns:
            df['City_Code'] = df['Delivery_person_ID'].str.extract(r'([A-Z]+)')
            df['Person_Number'] = df['Delivery_person_ID'].str.extract(r'(\d+)')
            df['Person_Number'] = pd.to_numeric(df['Person_Number'], errors='coerce')
        
        # Vehicle condition
        if 'Vehicle_condition' in df.columns:
            df['Good_Vehicle'] = (df['Vehicle_condition'] >= 1).astype(int)
        
        return df


class OrderFeatureTransformer(FeatureTransformer):
    """Creates features related to the order characteristics"""
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Multiple deliveries
        if 'multiple_deliveries' in df.columns:
            df['Has_Multiple_Deliveries'] = (df['multiple_deliveries'] > 0).astype(int)
        
        # Road traffic density
        if 'Road_traffic_density' in df.columns:
            traffic_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
            df['Road_traffic_density_Encoded'] = df['Road_traffic_density'].map(traffic_mapping).fillna(1)
        
        # Weather conditions
        if 'Weatherconditions' in df.columns:
            weather_dummies = pd.get_dummies(df['Weatherconditions'], prefix='Weather')
            df = pd.concat([df, weather_dummies], axis=1)
        
        # Type of order
        if 'Type_of_order' in df.columns:
            order_dummies = pd.get_dummies(df['Type_of_order'], prefix='Order')
            df = pd.concat([df, order_dummies], axis=1)
        
        # Type of vehicle
        if 'Type_of_vehicle' in df.columns:
            vehicle_dummies = pd.get_dummies(df['Type_of_vehicle'], prefix='Vehicle')
            df = pd.concat([df, vehicle_dummies], axis=1)
        
        # Festival as binary
        if 'Festival' in df.columns:
            df['Festival'] = (df['Festival'] == 'Yes').astype(int)
        
        # City one-hot encoding
        if 'City' in df.columns:
            city_dummies = pd.get_dummies(df['City'], prefix='City')
            df = pd.concat([df, city_dummies], axis=1)
        
        # Create interactions
        if 'Distance_km' in df.columns and 'multiple_deliveries' in df.columns:
            df['Multiple_Deliveries_Distance'] = df['multiple_deliveries'] * df['Distance_km']
        
        if 'Festival' in df.columns and 'Road_traffic_density_Encoded' in df.columns:
            df['Festival_Traffic'] = df['Festival'] * df['Road_traffic_density_Encoded']
        
        return df


class EngineerTimeFeatures(FeatureTransformer):
    """Engineers new time-based features"""
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Calculate time differences
        df['Preparation_Time_Mins'] = (df['Pickup_Datetime'] - df['Order_Datetime']).dt.total_seconds() / 60
        df['Day_Part'] = df['Hour'].apply(self._get_day_part)
        df['Rush_Period'] = df.apply(self._is_rush_period, axis=1)
        df['Weekend_Rush'] = (df['Is_Weekend'] * df['Rush_Period']).astype(int)
        
        return df
    
    def _get_day_part(self, hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    def _is_rush_period(self, row):
        hour = row['Hour']
        return 1 if (11 <= hour <= 13) or (19 <= hour <= 21) else 0

class EngineerDistanceFeatures(FeatureTransformer):
    """Engineers distance and location based features"""
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Calculate derived distance features
        df['Distance_Per_Delivery'] = df['Distance_km'] / (df['multiple_deliveries'] + 1)
        df['Is_Long_Distance'] = (df['Distance_km'] > df['Distance_km'].median()).astype(int)
        df['Distance_Complexity'] = self._calculate_distance_complexity(df)
        
        return df
    
    def _calculate_distance_complexity(self, df):
        """Calculate a complexity score based on distance and conditions"""
        base_complexity = df['Distance_km'] * df['Road_traffic_density_Encoded']
        if 'Weather_Cloudy' in df.columns:
            base_complexity *= (1 + 0.2 * df['Weather_Cloudy'])
        return base_complexity

class EngineerDeliveryFeatures(FeatureTransformer):
    """Engineers delivery-specific features"""
    
    def transform(self, df):
        df = self._copy_dataframe(df)
        
        # Create delivery complexity features
        df['Delivery_Complexity'] = self._calculate_delivery_complexity(df)
        df['Experience_Level'] = self._calculate_experience_level(df)
        df['Delivery_Efficiency'] = df['Distance_km'] / df['Time_taken(min)']
        
        return df
    
    def _calculate_delivery_complexity(self, df):
        complexity = df['Distance_km'] * (df['multiple_deliveries'] + 1)
        if 'Road_traffic_density_Encoded' in df.columns:
            complexity *= (1 + 0.5 * df['Road_traffic_density_Encoded'])
        return complexity
    
    def _calculate_experience_level(self, df):
        return pd.qcut(df['Person_Number'], q=4, labels=['Rookie', 'Junior', 'Senior', 'Expert'])


class FeatureEngineeringPipeline:
    """Orchestrates the complete feature engineering process"""
    
    def __init__(self, distance_strategy=None, transformers=None):
        self.transformers = transformers or [
            DataCleaner(),
            GeospatialFeatureTransformer(distance_strategy=distance_strategy),
            TimeFeatureTransformer(),
            DeliveryPersonFeatureTransformer(),
            OrderFeatureTransformer(),
            EngineerTimeFeatures(),
            EngineerDistanceFeatures(),
            EngineerDeliveryFeatures()
        ]
    
    def transform(self, df):
        """Apply all feature transformations in sequence"""
        print("Starting feature engineering pipeline...")
        result_df = df.copy()
        
        for transformer in self.transformers:
            transformer_name = transformer.__class__.__name__
            print(f"Applying {transformer_name}...")
            result_df = transformer.transform(result_df)
        
        print("Feature engineering pipeline completed!")
        return result_df


def process_dataset(input_path, output_path=None):
    """Process a dataset through the feature engineering pipeline"""
    print(f"Reading data from {input_path}")
    data = pd.read_csv(input_path)
    
    pipeline = FeatureEngineeringPipeline()
    processed_data = pipeline.transform(data)
    
    if output_path:
        print(f"Saving processed data to {output_path}")
        processed_data.to_csv(output_path, index=False)
    
    return processed_data


