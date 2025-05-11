# International Education Cost Predictor 🎓

This project is a web application designed to help prospective students estimate the total cost of pursuing international education. Users can input various preferences such as country, degree level, program duration, and estimated living expenses, and the application provides a predicted total cost using a machine learning model. The costs can also be viewed in Indian Rupees (INR).

## 🌟 Features

* **Interactive Cost Prediction:** Predicts total education costs based on user inputs.
* **Country and Degree Level Selection:** Choose from a list of available countries and degree levels.
* **Customizable Inputs:** Sliders for program duration, living cost index, monthly rent, annual insurance, and visa fees.
* **Currency Conversion:** Option to display costs in USD and Indian Rupees (INR).
* **Cost Breakdown:** Visualizes the predicted cost breakdown into Tuition, Living Expenses, and Visa Fees using a pie chart.
* **Comparative Analysis:** Displays average costs for the selected program in the chosen country and compares the prediction against this average.
* **Interactive Charts:** Utilizes Plotly for dynamic bar charts (comparing average costs across countries) and pie charts.
* **Data-Driven:** Trained on the "International Education Costs" dataset.
* **User-Friendly Interface:** Built with Streamlit for an intuitive web experience.
* **Model Persistence:** Loads a pre-trained model for quick predictions; retrains if the model is not found.

## 📊 Dataset

The model is trained on the [International Education Costs dataset](https://hebbkx1anhila5yf.public.blob.vercel-storage.com/International_Education_Costs-LeyxyBrErfLztS2WwEbWNzwX0Qvt8g.csv). This dataset compiles detailed financial information for students pursuing higher education abroad, covering multiple countries, cities, and universities, and includes tuition, living expenses, and ancillary costs.

The features used for prediction include:
* `Country`
* `Level` (Degree Level)
* `Duration_Years`
* `Living_Cost_Index`
* `Rent_USD`
* `Insurance_USD`
* `Visa_Fee_USD`

The target variable is `Total_Cost_USD`.

## 🛠️ Technology Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn (`RandomForestRegressor`, `Pipeline`, `StandardScaler`, `OneHotEncoder`)
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly, Matplotlib, Seaborn
* **Data Fetching:** Requests
* **Model Persistence:** Pickle

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepositoryName.git](https://github.com/YourUsername/YourRepositoryName.git)
    cd YourRepositoryName
    ```
    *(Replace `YourUsername/YourRepositoryName` with your actual GitHub username and repository name)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    * Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    Ensure you have a `requirements.txt` file in your repository (generate one using `pip freeze > requirements.txt` if you haven't already).
    ```bash
    pip install -r requirements.txt
    ```
    Your `requirements.txt` should ideally include:
    ```
    pandas
    numpy
    requests
    scikit-learn
    streamlit
    plotly
    matplotlib
    seaborn
    # forex-python (if you decide to use it actively instead of the hardcoded rate)
    ```

## ▶️ How to Run

Once the setup is complete, you can run the Streamlit application using the following command in your terminal (from the project's root directory):

```bash
streamlit run education_cost_predictor.py
