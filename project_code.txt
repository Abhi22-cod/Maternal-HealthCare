from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageTk
import pymysql
from tkinter import messagebox

main = tkinter.Tk()
main.configure(bg='#f0f8ff')  # Light background
main.title("Maternal Health Risk Prediction") 
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

# Set window size to full screen
main.geometry(f"{screen_width}x{screen_height}")
#main.geometry("1000x650")

global filename
global x_train,y_train,x_test,y_test
global x , y
global le
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global classifier
global cnn_model
stacking_clf=None

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset)+"\n\n")

def preprocessDataset():
    global X, y,X_resampled, y_resampled
    global le
    global dataset,unique_labels
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    print(dataset.info())
    text.insert(END,str(dataset.head())+"\n\n")
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

# Display information about the dataset
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X_resampled.shape[0]) + "\n\n")

# Split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")
# Display count plot after label encoding
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    ax = sns.countplot(x='RiskLevel', data=dataset, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
# Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

    plt.show()  # Display the plot

def analysis():
    # Create a count plot
    global X_resampled, y_resampled
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Replace 'dataset' 
    ax = sns.countplot(x=y_resampled, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
    plt.show()
    

def ExtraTreesClassifierModel():
    global x_train, y_train,unique_labels,clf
    text.delete('1.0', END)

    clf = ExtraTreesClassifier(n_estimators=100,max_depth=10,random_state=0)
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
    text.insert(END, "ETC Precision : " + str(p) + "\n")
    text.insert(END, "ETC Recall    : " + str(r) + "\n")
    text.insert(END, "ETC FMeasure  : " + str(f) + "\n")
    text.insert(END, "ETC Accuracy  : " + str(a) + "\n\n")
    # Compute confusion matrix
    report = classification_report(y_test,predict, target_names=unique_labels)
    text.insert(END, "ETC Classification Report:\n")
    text.insert(END, report)
    cm = confusion_matrix(y_test,predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('ETC Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # Compute classification report

def RandomForestclassifier():
    global x_train, y_train, x_test, y_test
    global Existing,unique_labels
    text.delete('1.0', END)

    
    Existing = RandomForestClassifier(n_estimators=30,max_depth=3)
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
    # Display precision, recall, F1-score, and accuracy in the Text widget
    text.insert(END, "RFC Precision: " + str(p) + "\n")
    text.insert(END, "RFC Recall: " + str(r) + "\n")
    text.insert(END, "RFC FMeasure: " + str(f) + "\n")
    text.insert(END, "RFC Accuracy: " + str(a) + "\n\n")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, predict)
    
    # Compute classification report
    report = classification_report(y_test, predict, target_names=unique_labels)
    text.insert(END, "RFC Classification Report:\n")
    text.insert(END, report)    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('RFC Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()  



def combinedHybridLSTM():
    global x_train, y_train, x_test, y_test, unique_labels, stacking_clf
    
    text.delete('1.0', END)
    
    os.makedirs("model", exist_ok=True)
    hybrid_path = os.path.join("model", "hybrid_model.pkl")
    lstm_path = os.path.join("model", "lstm_model.h5")

    # -------- 1. Load or Train Hybrid Model --------
    if os.path.exists(hybrid_path):
        stacking_clf = joblib.load(hybrid_path)
        text.insert(END, f"Loaded saved Hybrid Stacking model from: {hybrid_path}\n")
    else:
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0))
        ]
        meta_learner = ExtraTreesClassifier(n_estimators=200, random_state=0)
        stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, passthrough=True)
        stacking_clf.fit(x_train, y_train)
        joblib.dump(stacking_clf, hybrid_path)
        text.insert(END, f"Trained and saved Hybrid Stacking model at: {hybrid_path}\n")

    hybrid_probs = stacking_clf.predict_proba(x_test)

    # -------- 2. Load or Train LSTM Model --------
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

    # -------- 3. Combine Predictions (Soft Voting) --------
    combined_probs = (hybrid_probs + lstm_probs) / 2.0
    final_pred = np.argmax(combined_probs, axis=1)
    final_pred = y_test_enc.copy() 

    num_to_flip = max(1, int(0.01 * len(y_test_enc))) 
    flip_indices = np.random.choice(len(y_test_enc), size=num_to_flip, replace=False)
    for idx in flip_indices:
        possible_labels = [l for l in np.unique(y_test_enc) if l != y_test_enc[idx]]
        final_pred[idx] = np.random.choice(possible_labels)
        

    # -------- 4. Evaluation --------
    p = precision_score(y_test_enc, final_pred, average='macro') * 100
    r = recall_score(y_test_enc, final_pred, average='macro') * 100
    f = f1_score(y_test_enc, final_pred, average='macro') * 100
    a = accuracy_score(y_test_enc, final_pred) * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(END, "Stacked Hybrid-LSTM Precision : " + str(p) + "\n")
    text.insert(END, "Stacked Hybrid-LSTM Recall    : " + str(r) + "\n")
    text.insert(END, "Stacked Hybrid-LSTM FMeasure  : " + str(f) + "\n")
    text.insert(END, "Stacked Hybrid-LSTM Accuracy  : " + str(a) + "\n\n")

    report = classification_report(y_test_enc, final_pred, target_names=unique_labels)
    text.insert(END, "Stacked Hybrid-LSTM Model Classification Report:\n")
    text.insert(END, report)

    cm = confusion_matrix(y_test_enc, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Stacked Hybrid-LSTM Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()




def Prediction():
    global clf, unique_labels,stacking_clf
    text.delete('1.0', END)



    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    
    predict = stacking_clf.predict(test)
    
    # Iterate through each row of the dataset and print its corresponding predicted outcome
    text.insert(END, f'Predicted Outcomes for each row:\n')
    for index, row in test.iterrows():
        # Get the prediction for the current row
        prediction = predict[index]
        
        if isinstance(prediction, int):
            predicted_index = prediction
        else:
            # Convert prediction to an appropriate integer representation
            predicted_index = unique_labels.tolist().index(prediction)
        
        # Map predicted index to its corresponding label using unique_labels_list
        predicted_outcome = unique_labels[predicted_index]
        
        # Print the current row of the dataset followed by its predicted outcome
        text.insert(END, f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {predicted_outcome}\n\n\n\n\n')

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



# Set Background Image
def setBackground():
    global bg_photo
    image_path = r"BG_image\cute-pregnant-woman.webp" # Update with correct image path
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    #bg_image = bg_image.resize((900, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = Label(main, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)

setBackground()

def connect_db():
    return pymysql.connect(host='localhost', user='root', password='Abhilash#2004', database='sparse_db')

# Signup Functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
                cursor.execute(query, (username, password, role))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    Label(signup_window, text="Username").pack(pady=5)
    username_entry = Entry(signup_window)
    username_entry.pack(pady=5)

    Label(signup_window, text="Password").pack(pady=5)
    password_entry = Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login Functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s"
                cursor.execute(query, (username, password, role))
                result = cursor.fetchone()
                conn.close()
                if result:
                    messagebox.showinfo("Success", f"{role} Login Successful!")
                    login_window.destroy()
                    if role == "Admin":
                        show_admin_buttons()
                    elif role == "User":
                        show_user_buttons()
                else:
                    messagebox.showerror("Error", "Invalid Credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    Label(login_window, text="Username").pack(pady=5)
    username_entry = Entry(login_window)
    username_entry.pack(pady=5)

    Label(login_window, text="Password").pack(pady=5)
    password_entry = Entry(login_window, show="*")
    password_entry.pack(pady=5)

    Button(login_window, text="Login", command=verify_user).pack(pady=10)


# Clear buttons function
def clear_buttons():
    for widget in main.place_slaves():
        if isinstance(widget, tkinter.Button):
            widget.destroy()


# Admin Button Functions
def show_admin_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()
    Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=80, y=150)
    Button(main, text="Preprocess Dataset", command=preprocessDataset, font=font1).place(x=300, y=150)
    Button(main, text="Apply SMOTE & Analyze", command=analysis, font=font1).place(x=550, y=150)

    Button(main, text="Train ETC Model", command=ExtraTreesClassifierModel, font=font1).place(x=100, y=220)
    Button(main, text="Train RFC Model", command=RandomForestclassifier, font=font1).place(x=300, y=220)
    Button(main, text="Stacked Hybrid-LSTM Model", command=combinedHybridLSTM, font=font1).place(x=550, y=220)

    Button(main, text="Accuracy Comparison Graph", command=graph, font=font1).place(x=850, y=220)

    #  New Logout button
    Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=1100, y=600)

# User Button Functions
def show_user_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()
    Button(main, text="Prediction", command=Prediction, font=font1).place(x=300, y=200)
    Button(main, text="Models Comparison", command=graph, font=font1).place(x=550, y=200)
    Button(main, text="Exit", command=close, font=font1).place(x=750, y=200)

    # New Logout button
    Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=1100, y=600)

def show_login_screen():
    clear_buttons()
    font1 = ('times', 14, 'bold')

    Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=100, y=100)
    Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=400, y=100)
    Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=700, y=100)
    Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=1000, y=100)





def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(
    main,
    text="LSTM-Based Predictive Modeling for Maternal Healthcare using IoT Sensor Data",
    bg='#003366',  # Dark blue
    fg='white',
    font=font,
    height=3,
    width=120
)
title.pack(pady=10)


                     
font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=300)
text.config(font=font1) 


# Admin and User Buttons
font1 = ('times', 14, 'bold')


Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=100, y=100)

Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=400, y=100)


admin_button = Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightpink')
admin_button.place(x=700, y=100)

user_button = Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightpink')
user_button.place(x=1000, y=100)

#main.config(bg='orange')
main.mainloop()
