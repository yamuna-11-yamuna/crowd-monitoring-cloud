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
        process = subprocess.Popen(["python", "crowd_detection.py"])
        return "Started"

    return "Already running"

@app.route("/stop")
def stop():
    global process

    if process:
        process.terminate()   # 🔥 graceful stop
        process.wait()        # 🔥 wait until stopped
        process = None

        # 🔥 CLEAR DATA (VERY IMPORTANT)
        if os.path.exists("smart_crowd_results.csv"):
            os.remove("smart_crowd_results.csv")

        return "Stopped"

    return "Not running"


@app.route("/data")
def data():
    if os.path.exists("smart_crowd_results.csv"):
        df = pd.read_csv("smart_crowd_results.csv")

        df = df.dropna()   # 🔥 important fix

        prediction = int(df["Crowd Count"].tail(3).mean()) if len(df) > 3 else 0

        return jsonify({
            "data": df.tail(10).to_dict(orient="records"),
            "prediction": prediction
        })

    return jsonify({"data": [], "prediction": 0})


@app.route("/download")
def download():
    return send_file("smart_crowd_results.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)