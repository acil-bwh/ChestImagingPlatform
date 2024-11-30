import json
from cip_python.classification.localHistogramModel import LocalHistogramModel


class LocalHistogramModelMapper:
    def __init__(self):
        self.json_data = {}

    def create_config_json(self, conf_name, defined_classes, drop_list):
        json_data = {"defined_classes": defined_classes, "drop_list": drop_list}

        with open(conf_name, 'w') as fp:
            json.dump(json_data, fp, indent=4)
        self.json_data = json_data

    def read_config_json(self, file_to_read):
        with open(file_to_read, 'r') as fp:
            json_data = json.load(fp)

        self.json_data = json_data

    def map_model(self, training_df):
        drop_types = self.json_data["drop_list"]
        defined_classes = self.json_data["defined_classes"]

        # Drop types
        for drops in drop_types:
            training_df = training_df[training_df['ChestType'] != drops]

        # Map defined classes
        for initial_value, final_value in defined_classes.items():
            training_df['ChestType'] = training_df['ChestType'].replace(initial_value, final_value)

        return training_df

    @staticmethod
    def transform_model(conf_file, in_model, out_model):

        mapper = LocalHistogramModelMapper()
        mapper.read_config_json(conf_file)
        lh_model = LocalHistogramModel(in_model)
        training_df = lh_model.get_model_as_df()
        transformed_df = mapper.map_df(training_df)
        lh_model.create_hdf5_from_df(transformed_df, out_model)

    def __del__(self):
        pass

