import numpy as np
from data_processing import load_data, split_train_test
from preprocessing import create_pipeline
from model import train_model, evaluate_model, save_model, load_model, predict_prices, calculate_rmse

housing = load_data("data.csv")
 
train_set, test_set = split_train_test(housing, test_size=0.2, random_state=42)

# Separate the features and labels
housing_labels = train_set["MEDV"].copy()
housing = train_set.drop("MEDV", axis=1)

# Preprocess using a pipeline
num_pipeline = create_pipeline()
housing_num_tr = num_pipeline.fit_transform(housing)


model = RandomForestRegressor()
train_model(model, housing_num_tr, housing_labels)

# Evaluate the model 
rmse_scores = evaluate_model(model, housing_num_tr, housing_labels)
print_scores(rmse_scores)


save_model(model, "house_price_model.joblib")
model = load_model("house_price_model.joblib")

# Preprocess the test set
x_test = test_set.drop("MEDV", axis=1)
y_test = test_set["MEDV"].copy()
x_test_prep = num_pipeline.transform(x_test)

# Predict for the test set
final_pred = predict_prices(model, x_test_prep)

# Evaluate on the test set
final_rmse = calculate_rmse(y_test, final_pred)
print("Final RMSE:", final_rmse)

# Example
features = np.array([[-0.43942006, 3.12628155, -1.12165014, -0.27288841, -1.42262747,
                     -0.23979304, -1.31238772, 2.61111401, -1.0016859, -0.5778192,
                     -0.97491834, 0.41164221, -0.86091034]])

predicted_price = predict_prices(model, features)

print("Predicted Price:", predicted_price)
