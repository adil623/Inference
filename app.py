from flask import Flask, jsonify
import json
from flask_restful import Api, Resource, reqparse
from flask import Response, request
import character_model
import background_model
import segmentation_model
import pandas as pd

app = Flask(__name__)
api = Api(app)


class CharacterModel(Resource):
    def post(self):
        # modelInput = {
        # text: qwertyxyz.....
        # }
        modelInput = request.get_json()

        errModelInput = {
            "errText": '',
            "inferenceError": ''
        }

        if modelInput['text'] == '':
            errModelInput["errText"] = "Please Provide a Text Input"
        else:
            single_characters_output, character_pairs_output = character_model.inference(
                modelInput['text'])
            if single_characters_output or character_pairs_output:
                if single_characters_output:
                    single_characters_output = pd.DataFrame(
                        single_characters_output).to_json(orient='values')
                else:
                    single_characters_output = None
                if character_pairs_output:
                    character_pairs_output = pd.DataFrame(
                        character_pairs_output).to_json(orient='values')
                else:
                    character_pairs_output = None

                return {"single_characters_output": single_characters_output,
                        "character_pairs_output": character_pairs_output}, 200
            else:
                return {"single_characters_output": None,
                        "character_pairs_output": None}, 200

        return errModelInput, 400


class BackgroundModel(Resource):
    def post(self):
        # modelInput = {
        # text: qwertyxyz.....
        # }
        modelInput = request.get_json()

        errModelInput = {
            "errText": '',
            "inferenceError": ''
        }

        if modelInput['text'] == '':
            errModelInput["errText"] = "Please Provide a Text Input"
        else:
            background = background_model.inference(modelInput['text'])
            if background:
                background = str(background)
                return {"background": background}, 200
            else:
                errModelInput["inferenceError"] = "Inference not available"

        return errModelInput, 400


class SegmentationModel(Resource):
    def post(self):
        # modelInput = {
        # text: qwertyxyz.....
        # }
        modelInput = request.get_json()

        errModelInput = {
            "errText": '',
            "inferenceError": ''
        }

        if modelInput['text'] == '':
            errModelInput["errText"] = "Please Provide a Text Input"
        else:
            scenes = segmentation_model.inference(modelInput['text'])
            if scenes:
                return {"scenes": scenes}, 200
            else:
                errModelInput["inferenceError"] = "Inference not available"

        return errModelInput, 400


api.add_resource(CharacterModel, "/api/CharacterModel/")
api.add_resource(BackgroundModel, "/api/BackgroundModel/")
api.add_resource(SegmentationModel, "/api/SegmentationModel/")

if __name__ == "__main__":
    app.run(debug=True)
