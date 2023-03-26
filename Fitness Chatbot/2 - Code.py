# Import necessary libraries
import csv

# Define user registration form
def registration_form():
    # Ask user to provide basic details
    name = input("What is your name?")
    age = input("What is your age?")
    gender = input("What is your gender?")
    weight = input("What is your weight (in kg)?")
    height = input("What is your height (in cm)?")
    email = input("What is your email address?")

    # Save user data in a CSV file
    with open('user_data.csv', mode='w', newline='') as user_file:
        fieldnames = ['Name', 'Age', 'Gender', 'Weight', 'Height', 'Email']
        writer = csv.DictWriter(user_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Name': name, 'Age': age, 'Gender': gender, 'Weight': weight, 'Height': height, 'Email': email})

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load user data from CSV file
user_data = pd.read_csv('user_data.csv')

# Scale weight and height values
scaler = MinMaxScaler()
user_data[['Weight', 'Height']] = scaler.fit_transform(user_data[['Weight', 'Height']])

# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans

# Load user data from CSV file
user_data = pd.read_csv('user_data.csv')

# Define clustering algorithm
kmeans = KMeans(n_clusters=3)

# Fit the algorithm on the user data
kmeans.fit(user_data[['Age', 'Weight', 'Height']])

# Get the labels for each user
labels = kmeans.labels_

# Print the label for each user
for i in range(len(user_data)):
    print(f"{user_data['Name'][i]} belongs to cluster {labels[i]}")

# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load workout data from CSV file
workout_data = pd.read_csv('workout_data.csv')

# Load user data from CSV file
user_data = pd.read_csv('user_data.csv')

# Calculate the cosine similarity between user data and workout data
cosine_sim = cosine_similarity(user_data[['Age', 'Weight', 'Height']], workout_data[['Age', 'Weight', 'Height']])

# Get the top 5 recommended workouts for each user
for i in range(len(user_data)):
    sim_scores = list(enumerate(cosine_sim[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    workout_indices = [i[0] for i in sim_scores]
    print(f"Top 5 recommended workouts for {user_data['Name'][i]} are:")
    print(workout_data['Workout'][workout_indices])

from tkinter import *
from chatbot import get_response

# Create a function to send a message from the user to the chatbot
def send_message(event):
    # Get the message from the user input field
    user_message = user_input.get()
    # Add the message to the chat history
    chat_history.config(state=NORMAL)
    chat_history.insert(END, "You: " + user_message + "\n\n")
    chat_history.config(state=DISABLED)
    chat_history.yview(END)
    # Get the chatbot's response to the message
    bot_response = get_response(user_message)
    # Add the chatbot's response to the chat history
    chat_history.config(state=NORMAL)
    chat_history.insert(END, "Chatbot: " + bot_response + "\n\n")
    chat_history.config(state=DISABLED)
    chat_history.yview(END)
    # Clear the user input field
    user_input.delete(0, END)

# Create the main window for the chatbot interface
root = Tk()
root.title("Fitness Chatbot")

# Create the chat history display
chat_history = Text(root, height=20, width=50)
chat_history.config(state=DISABLED)
chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Create the user input field
user_input = Entry(root, width=50)
user_input.bind("<Return>", send_message)
user_input.grid(row=1, column=0, padx=10, pady=10)

# Create the send button
send_button = Button(root, text="Send", command=lambda: send_message(None))
send_button.grid(row=1, column=1, padx=10, pady=10)

# Start the main loop for the chatbot interface
root.mainloop()
