from cip_python.dcnn.projects.low_to_high_dose_filter import LowDoseToHighDoseEngine
from cip_python.dcnn.logic import DeepLearningModelsManager
from cip_python.input_output import ImageReaderWriter


class LowToHighDoseFilterDCNN:
    def __init__(self):
        pass

    def execute(self, input_file, cnn_model_path, output_file):
        image_io = ImageReaderWriter()
        in_ct, ct_header = image_io.read_in_numpy(input_file)

        ee = LowDoseToHighDoseEngine(None)
        filtered_image = ee.predict_low_to_high_dose_from_ct(in_ct, cnn_model_path)

        image_io.write_from_numpy(filtered_image, ct_header, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CNN method (3D-to-2D) for low dose to high dose image filtering')
    parser.add_argument('--i', dest='in_ct', help="Input low dose file (.nrrd)", type=str, required=True)
    parser.add_argument("--m", dest="dcnn_model", help='Optional CNN model path (.h5) for low dose to high dose '
                                                       'image filtering.',
                        metavar='<string>', required=False)
    parser.add_argument("--o", dest='out_ct', help='Output file (.nrrd)', type=str, required=True)
    op = parser.parse_args()

    if op.dcnn_model:
        dcnn_model = op.dcnn_model
    else:
        # Load with model manager
        manager = DeepLearningModelsManager()
        dcnn_model = manager.get_model('LOW_DOSE_TO_HIGH_DOSE')

    image_filter = LowToHighDoseFilterDCNN()
    image_filter.execute(op.in_ct, dcnn_model, op.out_ct)


