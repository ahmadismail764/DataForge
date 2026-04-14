# AGENTS.md

## Role & Objective
* You are an expert Full-Stack Data Engineer and Web Developer.
* The objective of this project is to design and implement an end-to-end Automated Machine Learning Application.
* The goal is to build an intuitive platform that allows non-technical users to upload raw data, select an ML task, and receive a fully trained and evaluated model, all without writing a single line of code.

## Tech Stack Constraints
* **Backend Framework:** You must use FastAPI or Flask to create endpoints for data upload, training, and model retrieval.
* **Frontend-Backend Communication:** The frontend must communicate with these endpoints via HTTP requests.
* **Model Serialization:** Use `pickle` or `joblib` for saving the trained pipeline.

## Technical Requirements Specification

### Frontend (User Interface)
* **File Upload:** The application must support at least `.csv` and `.xlsx` (Excel) file formats. Allow the user to upload a dataset.
* **Data Preview:** Display the first few rows of the uploaded dataset so the user can verify their upload.
* **Task Selection:** Provide a dropdown or radio button menu allowing the user to select the type of machine learning problem they want to solve: Classification, Regression, or Clustering.
* **Target Selection:** If Classification or Regression is selected, prompt the user to select the target column (label) from the uploaded dataset.
* **Results Display:** Display a clear, readable report of the model's performance metrics to the user.
* **Model Export:** Add a "Save Model" button. When clicked, this should serialize the trained model (and the preprocessing pipeline) using pickle or joblib and allow the user to download it as a `.pkl` or `joblib` file.

### Backend (Automated ML Pipeline)
* **Preprocessing:** Once the user initiates the process, your backend must automatically execute an Automated Data Preprocessing Pipeline.
* **Preprocessing Steps:** You must select one technique to implement for each preprocessing step: handle missing values, encode categorical variables, scale/normalize numerical features, and if the target column is imbalanced, perform any resampling technique.
* **Data Splitting:** Automatically split the data into training and testing sets (e.g., 80/20 split).
* **Algorithm Selection:** Implement at least two different algorithms per ML task type. Train both of them and select the best one.
* **Evaluation Metrics:** Calculate appropriate metrics based on the task:
* Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
* Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared Score.
* Clustering: Silhouette Score.