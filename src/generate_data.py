import numpy as np
import pandas as pd


def generate_light_data(n_samples=200, k=1000, noise=0.05):
    """
    
    Creates synthetic data simulating the inverse-square law for intensity of light with some noise constant to be used for training the model.

    Args:
        n_samples (int [optional]): Number of samples to be generated. Default = 200.
        k (int [optional]): Proportionality constant dictating intensity of the light "source". Default = 1000.
        noise (float [optional]): Noise factor to account for real world situations like reflection/absorption of light. Default = 0.05.

    Returns:
        DataFrame: 2 dimensional pandas data table containing x-variable distance and y-variable intensity.
    """
    
    # Distances (1m to 10m)
    distances = np.linspace(1, 10, n_samples)

    # Inverse-square law of light (I = k / d^2)
    intensities = k / distances**2

    # Add noise to data samples
    noise_term = intensities * noise * np.random.randn(n_samples)
    intensities_noisy = intensities + noise_term

    df = pd.DataFrame({
        "distance_m": distances,
        "intensity": intensities_noisy
    })
    
    return df

if __name__ == "__main__":
    df = generate_light_data()
    df.to_csv("data/light_data.csv", index=False)
    print("Data generated")