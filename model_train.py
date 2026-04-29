import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# 1 LOAD DATASET
# =========================

df = pd.read_csv(
"Sleep_health_and_lifestyle_dataset.csv"
)

# =========================
# 2 PILIH 5 INPUT SESUAI MODUL
# =========================

X = df[
[
"Sleep Duration",
"Physical Activity Level",
"Stress Level",
"Heart Rate",
"Daily Steps"
]
]

y = df["Quality of Sleep"]


# =========================
# 3 NORMALISASI
# =========================

scaler_X=StandardScaler()
scaler_y=StandardScaler()

X=scaler_X.fit_transform(X)

y=y.values.reshape(-1,1)
y=scaler_y.fit_transform(y)


# =========================
# 4 SPLIT DATA
# =========================

X_train,X_test,y_train,y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


# =========================
# 5 MODEL BACKPROPAGATION
# SESUAI MODUL
# =========================

model = Sequential([

Dense(
16,
activation='relu',
input_shape=(X_train.shape[1],)
),

Dense(
12,
activation='relu'
),

Dense(
8,
activation='relu'
),

Dense(
1,
activation='linear'
)

])


model.compile(
optimizer='adam',
loss='mean_squared_error'
)


# =========================
# EARLY STOPPING
# =========================

early_stopping=EarlyStopping(
monitor='val_loss',
patience=10,
restore_best_weights=True
)



# =========================
# 6 TRAINING
# =========================

history=model.fit(

X_train,
y_train,

epochs=200,
batch_size=5,

validation_data=(
X_test,
y_test
),

callbacks=[
early_stopping
],

verbose=1
)


# =========================
# 7 EVALUASI
# =========================

test_loss=model.evaluate(
X_test,
y_test
)

print(
"\nLoss pada data uji:",
test_loss
)


# =========================
# 8 PREDIKSI BARU
# =========================

new_data=np.array([
[
7.5,
60,
4,
72,
8000
]
])

new_data_scaled=scaler_X.transform(
new_data
)

pred_scaled=model.predict(
new_data_scaled
)

pred_actual=scaler_y.inverse_transform(
pred_scaled
)

print(
"\nPrediksi kualitas tidur selanjutnya:",
pred_actual.flatten()
)


# =========================
# 9 SAVE MODEL
# =========================

model.save(
"model_sleep.h5"
)

pickle.dump(
scaler_X,
open(
"scaler.pkl",
"wb"
)
)


# =========================
# 10 GRAFIK LOSS
# =========================

plt.plot(
history.history['loss'],
label='Training Loss'
)

plt.plot(
history.history['val_loss'],
label='Validation Loss'
)

plt.title(
'Training vs Validation Loss'
)

plt.xlabel(
'Epoch'
)

plt.ylabel(
'Loss'
)

plt.legend()

plt.show()