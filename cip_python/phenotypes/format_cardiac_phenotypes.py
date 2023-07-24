import pandas as pd
import argparse





class FormatCardiacDensitometry:

    def __init__(self):

        self.df = pd.DataFrame()


    def format_csv(self, input_csv):
        name = input_csv.split("_totalSegmentationDensitometry")[0]
        return_name = name + "_cardiacDensitometry.csv"
       

        self.df = pd.read_csv(input_csv)


        columns_to_select = ['CID','Region','Volume', 'Mass', 'HUMean', 'HUStd', 'HUKurtosis','HUSkewness', 'HUMode', 'HUMedian', 'HUMin', 'HUMax']
        column_value = ['aorta','heart_myocardium','heart_atrium_left', 'heart_ventricle_left', 'heart_atrium_right','heart_ventricle_right', 'pulmonary_artery']
        result = self.df.loc[self.df['Region'].isin(column_value), columns_to_select]

        result.to_csv(return_name, index=False)
        
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format cardiac phenotypes.')
    parser.add_argument('-i', dest='input', type=str, metavar='input', required=True, help="Input csv file")
    

    op = parser.parse_args()

    fc = FormatCardiacDensitometry()
    print ("Formatting...")
    fc.format_csv(op.input)
    print ("DONE.")