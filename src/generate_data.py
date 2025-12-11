import numpy as np
import pandas as pd


def generate_light_data(n_samples=100, k=1000, noise=0.0, max_distance=10):
    """
    
    Creates synthetic data simulating the inverse-square law for intensity of light with some noise constant to be used for training the model.

    Args:
        n_samples (int [optional]): Number of samples to be generated. Default = 100.
        k (int [optional]): Proportionality constant dictating intensity of the light "source". Default = 1000.
        noise (float [optional]): Noise factor to account for real world situations like reflection/absorption of light. Default = 0.0.
        max_distance (int [optional]): The max distance (r) to compare light intensity with. Default = 10.

    Returns:
        DataFrame: 2-dimensional pandas data table containing x-variable distance and y-variable intensity.
        
    """
    
    # Distances (1 to a chosen distance, max_distance)
    distances = np.linspace(1, max_distance, n_samples)

    # Inverse-square law of light (I = k / d^2)
    intensity = k / distances**2

    # Add noise to data samples
    noise_term = intensity * noise * np.random.randn(n_samples)
    intensity_noisy = intensity + noise_term

    df = pd.DataFrame({
        "distance_m": distances,
        "intensity": intensity_noisy
    })
    
    # Small cleanup to remove negative Intensity as that is not a real physical thing
    df = df[df["intensity"] > 0]
    
    return df

if __name__ == "__main__":
    df = generate_light_data()
    df.to_csv("data/light_data_test.csv", index=False)
    print("Data generated")