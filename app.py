from flask import Flask,request,render_template,json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.components.pipeline.predict_pipeline import CustomData,PredictPipeline


app = Flask(__name__)


# Route for home page
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('writing_score'),
            writing_score=request.form.get('reading_score')

        )
        pred_df=data.get_data_as_data_frame()
        
        print(pred_df)
        pred_pipeline=PredictPipeline()
        results=pred_pipeline.predict(pred_df)
        input_data_dict = {
                "gender": data.gender,
                "race_ethnicity": data.race_ethnicity,
                "parental_level_of_education": data.parental_level_of_education,
                "lunch": data.lunch,
                "test_preparation_course": data.test_preparation_course,
                "reading_score": data.reading_score,
                "writing_score": data.writing_score
            }
            
            # Save input data as JSON
        input_data_json = json.dumps(input_data_dict, indent=4)
        with open('input_data.json', 'w') as json_file:
            json_file.write(input_data_json)
        
        return render_template("home.html",results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)




