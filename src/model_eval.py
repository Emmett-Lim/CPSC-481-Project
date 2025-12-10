import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

def evaluate_models(data_path, data_tuple):
    
    df = pd.read_csv(data_path)
    y = df['intensity']
    
    lin_model, X_test, y_test = data_tuple
    y_pred_lin = lin_model.predict(X_test)
    
    # Debug
    print(y.max())
    print(y_test.max())
    print(y_pred_lin.max())

    # Metrics
    print("Linear Regression Evaluation (Test Set):")
    print("Linear R2:", r2_score(y_test, y_pred_lin))
    print("Linear RMSE:", root_mean_squared_error(y_test, y_pred_lin))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_lin))
    
    # Plot 1: Actual vs Predicted
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

    # Plot 2: Residual Plot
    residuals = y_test - y_pred_lin
    plt.scatter(y_pred_lin, residuals, alpha=0.5)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig("figures/residual_plot.png")
    plt.close()
    