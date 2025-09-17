Diabetes Risk Predictor 🩺
This project is a web application that predicts a patient's risk of having diabetes based on their medical information. It uses a Logistic Regression model trained on the Pima Indians Diabetes Dataset.

The application is built with Streamlit, providing a simple and interactive user interface where users can input their data and receive real-time predictions.

🚀 Demo
A screenshot of the application's user interface.

✨ Features
Interactive UI: A clean and user-friendly web form for data entry.

Real-Time Predictions: The model provides an instant prediction ("High Risk" or "Low Risk").

Probability Score: Shows the model's confidence in its prediction as a percentage.

Pre-trained Model: Includes a pre-trained Logistic Regression model and the corresponding data scaler.

🛠️ Technology Stack
Python: Core programming language.

Scikit-learn: For training the Logistic Regression model and scaling data.

Pandas: For data manipulation.

Streamlit: For building and serving the interactive web application.

⚙️ Setup and Installation
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
Bash

git clone https://github.com/your-username/diabetes-predictor.git
cd diabetes-predictor
2. Create a Virtual Environment
It's highly recommended to create a virtual environment to manage project dependencies.

Bash

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
3. Install Required Libraries
The requirements.txt file contains all the necessary Python libraries.

Bash

pip install -r requirements.txt
Note: If you don't have a requirements.txt file, you can create one from your training environment by running: pip freeze > requirements.txt

4. Run the Streamlit App
Once the dependencies are installed, you can run the application.

Bash

streamlit run app.py
The application will open in your web browser, ready to make predictions!

📂 File Structure
├── app.py             # The main Streamlit application script
├── model.pkl          # The serialized trained model
├── scaler.pkl         # The serialized fitted scaler
├── requirements.txt   # Project dependencies
└── README.md          # This file
💡 Important Note on Model Versions
The model.pkl and scaler.pkl files were saved using a specific version of scikit-learn. If you encounter an InconsistentVersionWarning, the most reliable solution is to retrain the model in your current environment by running your training script. This will generate new .pkl files that are perfectly compatible with your library versions.
