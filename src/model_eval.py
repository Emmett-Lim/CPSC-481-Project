import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

def evaluate_models(data_path, data_tuple):
    """
    Evaluates the trained machine learning model (Linear Regression) using the test dataset.
    Computes performance metrics such as R-squared, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
    Also generates plots comparing predicted values against actual values, and a residual plot.

    Args:
        data_path (string): Path to the CSV file containing the full dataset.
        
        data_tuple (_type_): Contains the following:
            - linreg_model (LinearRegression): The trained LinearRegression model.
            - X_test (Pandas DataFrame): 2-dimensional data containing feature values of distance.
            - y_test (Pandas Series): 1-dimensional data containing outputs for Intensity.
    
    Returns:
        linreg_r2 (float): R-squared score of the model.
        
        linreg_rmse (float): Root Mean Squared Error.
        
        linreg_mae (float): Mean Absolute Error.
        
    """
    
    df = pd.read_csv(data_path)
    X = df["distance_m"]
    y = df['intensity']
    
    lin_model, X_test, y_test = data_tuple
    y_pred_lin = lin_model.predict(X_test)

    # Linear Regression Metrics
    linreg_r2 = r2_score(y_test, y_pred_lin)
    linreg_rmse = root_mean_squared_error(y_test, y_pred_lin)
    linreg_mae = mean_absolute_error(y_test, y_pred_lin)
    
    # Plot 1: Graph of Intensity vs Distance
    plt.scatter(X, y, alpha=0.5)
    plt.xlabel("Distance (m)")
    plt.ylabel("Intensity")
    plt.title("Intensity vs Distance")
    plt.savefig("figures/intensity_vs_distance.png")
    plt.close()
    
    # Plot 2: Graph of Intensity vs 1/Distance-squared
    inv_r2 = 1 / (X ** 2)

    plt.scatter(inv_r2, y, alpha=0.5)
    plt.xlabel("1 / Distance²")
    plt.ylabel("Intensity")
    plt.title("Intensity vs 1/Distance²")
    plt.savefig("figures/intensity_vs_inv_distsqrd.png")
    plt.close()
    
    # Plot 3: Actual vs Predicted
    plt.scatter(y_test, y_pred_lin, alpha=0.5)
    plt.plot(
        [y.min(), y.max()],
        [y.min(), y.max()],
        color="red"
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs. Actual Values")
    plt.savefig("figures/predicted_vs_actual.png")
    plt.close()

    # Plot 4: Residual Plot
    residuals = y_test - y_pred_lin
    plt.scatter(y_pred_lin, residuals, alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig("figures/residual_plot.png")
    plt.close()
    
    return linreg_r2, linreg_rmse, linreg_mae
    