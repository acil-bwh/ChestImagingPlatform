import numpy as np

def parenchyma_phenotypes():
    """General purpose function to generate parenchyma-based phenotypes

    Parameters
    ----------

    Returns
    -------

    """
    pass 


if __name__ == "__main__":
    desc = """Generates parenchyma phenotypes given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input CT file', dest='in_ct', metavar='<string>',
                      default=None)
    parser.add_option('--in_lm',
                      help='Input label map containing structures of interest. \
                      If a phenotype requires a segmented structure that is \
                      not present in this label map, it will not be computed',
                      dest='in_lm', metavar='<string>', default=None)

    (options, args) = parser.parse_args()
