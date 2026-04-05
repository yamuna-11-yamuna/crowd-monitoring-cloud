from flask import Flask, jsonify, render_template, send_file
import pandas as pd
import subprocess
import os

app = Flask(__name__)

process = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start")
def start():
    global process

    if process is None or process.poll() is not None:
        print("Starting detection...")
        process = subprocess.Popen(["python", "crowd_detection.py"])
        return "Started"
    else:
        return "Already running"


@app.route("/stop")
def stop():
    global process

    if process is not None:
        process.terminate()
        process = None
        return "Stopped"
    return "No process running"


@app.route("/data")
def data():
    if os.path.exists("smart_crowd_results.csv"):
        df = pd.read_csv("smart_crowd_results.csv")

        if len(df) > 3:
            prediction = int(df["Crowd Count"].tail(3).mean())
        else:
            prediction = 0

        return jsonify({
            "data": df.tail(10).to_dict(orient="records"),
            "prediction": prediction
        })

    return jsonify({"data": [], "prediction": 0})


# ✅ FIXED DOWNLOAD
@app.route("/download")
def download():
    return send_file("smart_crowd_results.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)