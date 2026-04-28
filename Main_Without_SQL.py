from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageTk
from tkinter import messagebox

main = tkinter.Tk()
main.configure(bg='#f0f8ff')  # Light background
main.title("Maternal Health Risk Prediction") 
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# Global variables
global filename, x_train, y_train, x_test, y_test, X, Y, le, dataset
accuracy, precision, recall, fscore = [], [], [], []
stacking_clf = None

# ------------------- Dataset Functions ------------------- #
def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset) + "\n\n")

def preprocessDataset():
    global X, y, X_resampled, y_resampled, le, dataset, unique_labels, x_train, x_test, y_train, y_test
    text.delete('1.0', END)
    print(dataset.info())
    text.insert(END, str(dataset.head()) + "\n\n")
    
    non_numeric_columns = dataset.select_dtypes(exclude=['int', 'float']).columns
    dataset.dropna(inplace=True)
    X = dataset.drop('RiskLevel', axis=1)
    y = dataset['RiskLevel']
    unique_labels = y.unique()
    
    for col in non_numeric_columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    text.insert(END, "Total records found in dataset: " + str(X_resampled.shape[0]) + "\n\n")
    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)
    text.insert(END, "Training records: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Testing records: " + str(x_test.shape[0]) + "\n\n")
    
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='RiskLevel', data=dataset, palette="Set3")
    plt.title("Count Plot")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    plt.show()

def analysis():
    global X_resampled, y_resampled
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=y_resampled, palette="Set3")
    plt.title("Count Plot")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    plt.show()

# ------------------- ML Models ------------------- #
def ExtraTreesClassifierModel():
    global x_train, y_train, unique_labels, clf
    text.delete('1.0', END)
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=0)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, f"ETC Precision: {p}\nETC Recall: {r}\nETC FMeasure: {f}\nETC Accuracy: {a}\n\n")
    report = classification_report(y_test, predict, target_names=unique_labels)
    text.insert(END, "ETC Classification Report:\n" + report)
    cm = confusion_matrix(y_test, predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('ETC Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def RandomForestclassifier():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)
    Existing = RandomForestClassifier(n_estimators=30, max_depth=3)
    Existing.fit(x_train, y_train)
    predict = Existing.predict(x_test)
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, f"RFC Precision: {p}\nRFC Recall: {r}\nRFC FMeasure: {f}\nRFC Accuracy: {a}\n\n")
    report = classification_report(y_test, predict, target_names=unique_labels)
    text.insert(END, "RFC Classification Report:\n" + report)
    cm = confusion_matrix(y_test, predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('RFC Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ------------------- Hybrid LSTM Model ------------------- #
def combinedHybridLSTM():
    global x_train, y_train, x_test, y_test, unique_labels, stacking_clf
    text.delete('1.0', END)
    
    os.makedirs("model", exist_ok=True)
    hybrid_path = os.path.join("model", "hybrid_model.pkl")
    lstm_path = os.path.join("model", "lstm_model.h5")

    # Hybrid Stacking
    if os.path.exists(hybrid_path):
        stacking_clf = joblib.load(hybrid_path)
        text.insert(END, f"Loaded saved Hybrid Stacking model from: {hybrid_path}\n")
    else:
        base_learners = [('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)),
                         ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0))]
        meta_learner = ExtraTreesClassifier(n_estimators=200, random_state=0)
        stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, passthrough=True)
        stacking_clf.fit(x_train, y_train)
        joblib.dump(stacking_clf, hybrid_path)
        text.insert(END, f"Trained and saved Hybrid Stacking model at: {hybrid_path}\n")

    hybrid_probs = stacking_clf.predict_proba(x_test)

    # LSTM
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(np.unique(y_train_enc))
    y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)

    x_train_reshaped = np.array(x_train).reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test_reshaped = np.array(x_test).reshape((x_test.shape[0], 1, x_test.shape[1]))

    if os.path.exists(lstm_path):
        lstm_model = load_model(lstm_path)
        text.insert(END, f"Loaded saved LSTM model from: {lstm_path}\n")
    else:
        lstm_model = Sequential()
        lstm_model.add(LSTM(64, return_sequences=False, input_shape=(x_train_reshaped.shape[1], x_train_reshaped.shape[2])))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(Dense(64, activation='relu'))
        lstm_model.add(Dense(num_classes, activation='softmax'))
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_model.fit(x_train_reshaped, y_train_cat, epochs=30, batch_size=8, verbose=0,
                       validation_data=(x_test_reshaped, y_test_cat))
        lstm_model.save(lstm_path)
        text.insert(END, f"Trained and saved LSTM model at: {lstm_path}\n")

    lstm_probs = lstm_model.predict(x_test_reshaped)
    combined_probs = (hybrid_probs + lstm_probs) / 2.0
    final_pred = np.argmax(combined_probs, axis=1)

    p = precision_score(y_test_enc, final_pred, average='macro') * 100
    r = recall_score(y_test_enc, final_pred, average='macro') * 100
    f = f1_score(y_test_enc, final_pred, average='macro') * 100
    a = accuracy_score(y_test_enc, final_pred) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(END, f"Stacked Hybrid-LSTM Precision: {p}\nStacked Hybrid-LSTM Recall: {r}\nStacked Hybrid-LSTM FMeasure: {f}\nStacked Hybrid-LSTM Accuracy: {a}\n\n")
    report = classification_report(y_test_enc, final_pred, target_names=unique_labels)
    text.insert(END, "Stacked Hybrid-LSTM Classification Report:\n" + report)
    cm = confusion_matrix(y_test_enc, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Stacked Hybrid-LSTM Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ------------------- Prediction Function ------------------- #
def Prediction():
    global stacking_clf, unique_labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    predict = stacking_clf.predict(test)
    text.insert(END, f'Predicted Outcomes for each row:\n')
    for index, row in test.iterrows():
        predicted_outcome = unique_labels[predict[index]] if isinstance(predict[index], int) else predict[index]
        text.insert(END, f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {predicted_outcome}\n\n')

# ------------------- Graph Function ------------------- #
def graph():
    if len(accuracy) < 3:
        text.insert(END, "Run ETC, RFC, and Hybrid models first.\n")
        return
    df = pd.DataFrame([
        ['ETC', 'Precision', precision[0]], ['ETC', 'Recall', recall[0]], ['ETC', 'F1 Score', fscore[0]], ['ETC', 'Accuracy', accuracy[0]],
        ['RFC', 'Precision', precision[1]], ['RFC', 'Recall', recall[1]], ['RFC', 'F1 Score', fscore[1]], ['RFC', 'Accuracy', accuracy[1]],
        ['Hybrid-LSTM', 'Precision', precision[2]], ['Hybrid-LSTM', 'Recall', recall[2]], ['Hybrid-LSTM', 'F1 Score', fscore[2]], ['Hybrid-LSTM', 'Accuracy', accuracy[2]],
    ], columns=['Model', 'Metric', 'Value'])
    pivot_df = df.pivot(index='Metric', columns='Model', values='Value')
    pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score (%)')
    plt.xlabel('Metrics')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


def login_admin(role):
    show_admin_buttons()

def login_user(role):
    show_user_buttons()
    
# ------------------- GUI Setup ------------------- #
def clear_buttons():
    for widget in main.place_slaves():
        if isinstance(widget, tkinter.Button):
            widget.destroy()

def show_admin_buttons():
    clear_buttons()
    font1 = ('times', 13, 'bold')
    Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=80, y=150)
    Button(main, text="Preprocess Dataset", command=preprocessDataset, font=font1).place(x=300, y=150)
    Button(main, text="Apply SMOTE & Analyze", command=analysis, font=font1).place(x=550, y=150)
    Button(main, text="Train ETC Model", command=ExtraTreesClassifierModel, font=font1).place(x=100, y=220)
    Button(main, text="Train RFC Model", command=RandomForestclassifier, font=font1).place(x=300, y=220)
    Button(main, text="Stacked Hybrid-LSTM Model", command=combinedHybridLSTM, font=font1).place(x=550, y=220)
    Button(main, text="Accuracy Comparison Graph", command=graph, font=font1).place(x=850, y=220)
    Button(main, text="User Actions", command=lambda: login_user("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=1000, y=100)

def show_user_buttons():
    clear_buttons()
    font1 = ('times', 13, 'bold')
    Button(main, text="Prediction", command=Prediction, font=font1).place(x=300, y=200)
    Button(main, text="Models Comparison", command=graph, font=font1).place(x=550, y=200)
    Button(main, text="Exit", command=main.destroy, font=font1).place(x=750, y=200)

def show_login_screen():
    font1 = ('times', 14, 'bold')
    Button(main, text="Admin", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=700, y=100)

# ------------------- Title ------------------- #
font = ('times', 16, 'bold')
title = Label(main, text="LSTM-Based Predictive Modeling for Maternal Healthcare using IoT Sensor Data",
              bg='#003366', fg='white', font=font, height=3, width=120)
title.pack(pady=10)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100, y=300)
text.config(font=font1)

# Initial login/signup buttons
font1 = ('times', 14, 'bold')
Button(main, text="Admin Actions", command=lambda: login_admin("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=700, y=100)

main.mainloop()
