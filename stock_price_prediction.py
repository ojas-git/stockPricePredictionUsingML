import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk, filedialog
from tkinter import messagebox
import os
plt.style.use('dark_background')

def stock_prediction(file_path, models):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Price'] = data['Close']
    
    X = np.array(data.index).reshape(-1, 1)
    y = data['Price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    predictions = {}
    train_predictions = {}

    error_metrics = []

    for model_name in models:
        if model_name == 'Linear Regression':
            model = LinearRegression()
        elif model_name == 'Random Forest':
            model = RandomForestRegressor()
        elif model_name == 'SVR':
            model = SVR(kernel='rbf')

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_predictions[model_name] = train_pred
        predictions[model_name] = test_pred

        mae = mean_absolute_error(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred)
        rmse = np.sqrt(mse)
        error_metrics.append([model_name, mae, mse, rmse])

    avg_test_pred = np.mean(list(predictions.values()), axis=0)
    avg_mae = mean_absolute_error(y_test, avg_test_pred)
    avg_mse = mean_squared_error(y_test, avg_test_pred)
    avg_rmse = np.sqrt(avg_mse)
    error_metrics.append(['Average Prediction', avg_mae, avg_mse, avg_rmse])

    plt.figure(figsize=(10, 6))
    
    # Training Data Plot
    plt.subplot(1, 2, 1)
    plt.plot(y_train, label="Actual Training Prices", color='white')
    for model_name, train_pred in train_predictions.items():
        plt.plot(train_pred, label=f"{model_name} Training Prediction")
    plt.title('Training Data Predictions', color='white')
    plt.xlabel('Time', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()

    # Testing Data Plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label="Actual Test Prices", color='white')
    for model_name, test_pred in predictions.items():
        plt.plot(test_pred, label=f"{model_name} Test Prediction")
    plt.plot(avg_test_pred, label="Average Test Prediction", linestyle='--', color='cyan')
    plt.title('Testing Data Predictions', color='white')
    plt.xlabel('Time', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return error_metrics

class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        self.root.geometry("600x500")
        self.root.configure(bg='grey')
        
        self.stock_label = Label(root, text="Select Stock CSV:", fg='white', bg='grey')
        self.stock_label.pack(pady=10)

        self.stock_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        self.selected_stock = StringVar()
        self.stock_dropdown = ttk.Combobox(root, textvariable=self.selected_stock, values=self.stock_files)
        self.stock_dropdown.pack(pady=10)

        self.model_vars = {}
        self.models = ['Linear Regression', 'Random Forest', 'SVR']

        for model in self.models:
            var = BooleanVar()
            self.model_vars[model] = var
            checkbox = Checkbutton(root, text=model, variable=var, fg='white', bg='grey', selectcolor='gray')
            checkbox.pack(pady=5)

        self.predict_button = Button(root, text="Predict", command=self.predict_stock, fg='white', bg='grey')
        self.predict_button.pack(pady=20)

        style = ttk.Style()
        style.configure("Treeview", background="grey", foreground="black", fieldbackground="grey")
        style.configure("Treeview.Heading", background="gray", foreground="black")

        self.error_table = ttk.Treeview(root, columns=("Model", "MAE", "MSE", "RMSE"), show="headings")
        self.error_table.heading("Model", text="Model")
        self.error_table.heading("MAE", text="MAE")
        self.error_table.heading("MSE", text="MSE")
        self.error_table.heading("RMSE", text="RMSE")
        
        self.error_table.pack(pady=20, fill="x")

    def predict_stock(self):
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("No Models Selected", "Please select at least one model.")
            return
        
        stock_file = self.selected_stock.get()
        if not stock_file:
            messagebox.showwarning("No Stock Selected", "Please select a stock CSV file.")
            return

        file_path = os.path.join('data', stock_file)
        error_metrics = stock_prediction(file_path, selected_models)

        for row in self.error_table.get_children():
            self.error_table.delete(row)

        for metric in error_metrics:
            self.error_table.insert("", "end", values=metric)

if __name__ == "__main__":
    root = Tk()
    app = StockPredictorGUI(root)
    root.mainloop()
