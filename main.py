from flask import Flask,render_template,request
import numpy as np
import pickle

from tensorflow.keras.models import load_model

app=Flask(__name__)

# load model hasil training backpropagation
model=load_model(
"model_sleep.h5"
)

# load scaler
scaler=pickle.load(
open(
"scaler.pkl",
"rb"
)
)


@app.route("/")
def home():
    return render_template(
    "index.html"
    )


@app.route(
"/predict",
methods=["POST"]
)
def predict():

    sleep_duration=float(
    request.form["sleep_duration"]
    )

    activity=float(
    request.form["activity"]
    )

    stress=float(
    request.form["stress"]
    )

    heart=float(
    request.form["heart"]
    )

    steps=float(
    request.form["steps"]
    )


    data=np.array([[
        sleep_duration,
        activity,
        stress,
        heart,
        steps
    ]])


    # preprocessing
    data=scaler.transform(
    data
    )


    # prediksi backpropagation
    pred_scaled=model.predict(
    data
    )


    prediction=float(
    pred_scaled[0][0]
    )


    # kategori output
    if prediction >=8:
        kategori="Baik"

    elif prediction >=6:
        kategori="Cukup"

    else:
        kategori="Buruk"



    return render_template(
        "result.html",
        prediction=round(
        prediction,
        2
        ),
        kategori=kategori
    )



if __name__=="__main__":
    app.run(
    debug=True
    )