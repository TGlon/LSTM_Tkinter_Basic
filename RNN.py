import tensorflow as tf
import numpy as np
import collections
from keras.layers import LSTM, Dense

#Hàm đọc file:
def read_data(fname):
  with open(fname) as f:
    content = f.readlines()
  content = [x.strip() for x in content]
  words = []
  for line in content:
    words.extend(line.split())
  return np.array(words)
#Hàm tạo từ điển
def build_dataset(words):
  count = collections.Counter(words).most_common()
  word2id = {}
  for word, freq in count:
    word2id[word] = len(word2id)
  id2word = dict(zip(word2id.values(), word2id.keys()))
  return word2id, id2word

#Đọc dữ liệu 
data = read_data('D:/DeepLearning/truyen.txt') 
print(data)
w2i, i2w = build_dataset(data)
vocab_size = len(w2i)
timestep = 3

#tạo dữ liệu
encoded_data = [w2i[x] for x in data]
X = encoded_data[:-1]
Y = encoded_data[timestep:]
train_data = tf.keras.preprocessing.timeseries_dataset_from_array(X, Y, sequence_length=timestep, sampling_rate=1)

#Tạo model
model = tf.keras.Sequential()
model.add(LSTM(512, return_sequences=True,
input_shape=(timestep, 1)))
model.add(LSTM(512, return_sequences=False))
model.add(Dense(vocab_size))

#Huấn Luyện
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(train_data, epochs=500)
model.save('my_rnn_model.h5')

#Hàm đọc 10 từ tiếp theo
def generate_text(model, w2i, i2w, seed_text, next_words=10):
    for _ in range(next_words):
        token_list = [w2i[word] for word in seed_text.split()]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=timestep, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        output_word = i2w[predicted_word_index]
        seed_text += " " + output_word
    return seed_text.title()

# seed_text = "they saw that"
# generated_text = generate_text(model, w2i, i2w, seed_text, next_words=10)
# print(generated_text)

# Hàm chạy khi ấn nút Generate
def run_generator():
    generated_text.set(generate_text(model, w2i, i2w, input_text.get(), 10))

# Tạo giao diện
import tkinter as tk
root = tk.Tk()
root.title("Text Generator")

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

input_label = tk.Label(input_frame, text="Input Text:")
input_label.pack(side=tk.LEFT)

input_text = tk.Entry(input_frame, width=50)
input_text.pack(side=tk.LEFT)

generate_button = tk.Button(root, text="Generate", command=run_generator)
generate_button.pack()

output_frame = tk.Frame(root)
output_frame.pack(pady=10)

output_label = tk.Label(output_frame, text="Generated Text:")
output_label.pack(side=tk.LEFT)

generated_text = tk.StringVar()
generated_text.set("")
output_text = tk.Label(output_frame, textvariable=generated_text)
output_text.pack(side=tk.LEFT)

root.mainloop()