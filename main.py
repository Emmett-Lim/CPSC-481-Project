from src.generate_data import generate_light_data
from src.train_model import train_models
from src.model_eval import evaluate_models

def main():
    print("Generating data...")
    generate_light_data(n_samples=200, k=1000, noise=0.15, max_distance=25).to_csv("data/light_data.csv", index=False)

    print("Training model...")
    lin_model, X_test, y_test = train_models("data/light_data.csv")
    data_tuple = (lin_model, X_test, y_test)

    print("Evaluating model...")
    evaluate_models("data/light_data.csv", data_tuple)

if __name__ == "__main__":
    main()