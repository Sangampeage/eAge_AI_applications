# main.py

from inference import CropRecommender


if __name__ == "__main__":

    recommender = CropRecommender("artifacts/rf_model.pkl")

    input_json = {
        "soil": "Loamy",
        "N": 90,
        "P": 40,
        "K": 40,
        "ph": 6.5,
        "temperature": 28.0,
        "moisture": 70.0,
        "ec": 1.2
    }

    result_json = recommender.recommend(input_json)

    print(result_json)