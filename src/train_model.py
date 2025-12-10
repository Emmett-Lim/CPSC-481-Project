import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_models(data_path):
    """
    
    Trains the machine learning model using linear regression on the inverse-square law equation.

    Args:
        data_path (string): The path to the data set (csv file)

    Returns:
        - linreg_model (LinearRegression):
        - X_test (Pandas DataFrame): 2-dimensional data
        - y_test (Pandas Series): 1-dimensional data containing outputs for Intensity
        
    """
    
    # Read csv file to get data and store into df
    df = pd.read_csv(data_path)
    
    # Separate data into x and y variables
    X = 1 / (df[["distance_m"]] ** 2)
    X.columns = ["inv_r2"]
    y = df["intensity"]

    # Split the data into training and testing sets
    # (80% of dataset -> Training set; 20% of dataset -> Testing set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear regression model training
    linreg_model = LinearRegression()
    linreg_model.fit(X_train, y_train)

    return linreg_model, X_test, y_test