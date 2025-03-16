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

    plt.subplot(1, 2, 1)
    plt.plot(y_train, label="Actual Training Prices", color='black')
    for model_name, train_pred in train_predictions.items():
        plt.plot(train_pred, label=f"{model_name} Training Prediction")
    plt.title('Training Data Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(y_test, label="Actual Test Prices", color='black')
    for model_name, test_pred in predictions.items():
        plt.plot(test_pred, label=f"{model_name} Test Prediction")
    plt.plot(avg_test_pred, label="Average Test Prediction", linestyle='--', color='blue')
    plt.title('Testing Data Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return error_metrics

    # Load the CSV file
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Price'] = data['Close']
    
    X = np.array(data.index).reshape(-1, 1)  
    y = data['Price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    predictions = {}
    train_predictions = {}

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

    avg_test_pred = np.mean(list(predictions.values()), axis=0)
    
    for model_name, test_pred in predictions.items():
        mae = mean_absolute_error(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred)
        rmse = np.sqrt(mse)
        print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    
    avg_mae = mean_absolute_error(y_test, avg_test_pred)
    avg_mse = mean_squared_error(y_test, avg_test_pred)
    avg_rmse = np.sqrt(avg_mse)
    print(f"Average Prediction - MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}")

    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_train, label="Actual Training Prices", color='black')
    for model_name, train_pred in train_predictions.items():
        plt.plot(train_pred, label=f"{model_name} Training Prediction")
    plt.title('Training Data Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_test, label="Actual Test Prices", color='black')
    for model_name, test_pred in predictions.items():
        plt.plot(test_pred, label=f"{model_name} Test Prediction")
    plt.plot(avg_test_pred, label="Average Test Prediction", linestyle='--', color='blue')
    plt.title('Testing Data Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return predictions

    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Price'] = data['Close']
    
    # Features and target
    X = np.array(data.index).reshape(-1, 1)
    y = data['Price'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    predictions = {}
    
    if 'Linear Regression' in models:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        predictions['Linear Regression'] = lr.predict(X_test)
    
    if 'Random Forest' in models:
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        predictions['Random Forest'] = rf.predict(X_test)
    
    if 'SVR' in models:
        svr = SVR(kernel='rbf')
        svr.fit(X_train, y_train)
        predictions['SVR'] = svr.predict(X_test)
    
    avg_prediction = np.mean(list(predictions.values()), axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Prices", color='black')
    for model, pred in predictions.items():
        plt.plot(pred, label=f"{model} Prediction")
    plt.plot(avg_prediction, label="Average Prediction", linestyle='--', color='blue')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

    return predictions

class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        self.root.geometry("600x500")
        
        self.stock_label = Label(root, text="Select Stock CSV:")
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
            checkbox = Checkbutton(root, text=model, variable=var)
            checkbox.pack(pady=5)

        self.predict_button = Button(root, text="Predict", command=self.predict_stock)
        self.predict_button.pack(pady=20)

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

# Create GUI Window
if __name__ == "__main__":
    root = Tk()
    app = StockPredictorGUI(root)
    root.mainloop()

    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        self.root.geometry("600x500")

        self.stock_label = Label(root, text="Select Stock CSV:")
        self.stock_label.pack(pady=10)

        self.stock_files = [f for f in os.listdir() if f.endswith('.csv')]
        self.selected_stock = StringVar()
        self.stock_dropdown = ttk.Combobox(root, textvariable=self.selected_stock, values=self.stock_files)
        self.stock_dropdown.pack(pady=10)

        self.model_vars = {}
        self.models = ['Linear Regression', 'Random Forest', 'SVR']

        for model in self.models:
            var = BooleanVar()
            self.model_vars[model] = var
            checkbox = Checkbutton(root, text=model, variable=var)
            checkbox.pack(pady=5)

        self.predict_button = Button(root, text="Predict", command=self.predict_stock)
        self.predict_button.pack(pady=20)

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

        error_metrics = stock_prediction(stock_file, selected_models)

        for row in self.error_table.get_children():
            self.error_table.delete(row)

        for metric in error_metrics:
            self.error_table.insert("", "end", values=metric)
