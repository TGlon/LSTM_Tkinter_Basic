import tensorflow as tf
import numpy as np
import collections
from keras.layers import LSTM, Dense
from keras.models import load_model

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

# Load model và đặt lại các biến cần thiết
model = load_model('my_rnn_model.h5', compile=False)
w2i, i2w = build_dataset(data)
vocab_size = len(w2i)
timestep = model.input_shape[1]

# Hàm đọc 10 từ tiếp theo
def generate_text(model, w2i, i2w, seed_text, next_words=10):
    for _ in range(next_words):
        token_list = [w2i[word] for word in seed_text.split()]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=timestep, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        output_word = i2w[predicted_word_index]
        seed_text += " " + output_word
    return seed_text.title()
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
       
