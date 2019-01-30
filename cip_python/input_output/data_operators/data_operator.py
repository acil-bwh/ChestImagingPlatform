class DataOperatorInterface(object):
    def __init__(self):
        raise Exception("You cannot create instances of this class. You must create child classes")

    def run(self, data, generate_parameters=True):
        """
        Apply an operation to a set of data.
        :param data: data where were are applying the operation (typically a numpy array)
        :param generate_parameters: bool. Generate the parameters to apply the operation according to the policy
                                    implemented in the class. Otherwise, the class manually set params will be used
        """
        raise NotImplementedError("This method must be implemented in a child class")

    def set_operation_parameters(self):
        """
        Set externally the parameters that the operation is going to use
        :return:
        """
        raise NotImplementedError("This method must be implemented in a child class")