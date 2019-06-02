import pandas as pd
import argparse
from visualize import visualize


class Datahandler():
    """Bike Sharing Dataset datahandler class. Contains methods
       for exploring and cleaning of data, as well as for
       dealing with missing values and dropping attributes.  

       Args:
            csv_path {str} -- Path to the Bike Sharing Dataset CSV file. 
    """

    def __init__(self, csv_path):
        # A list to cover some possible missing values in data
        self.missing_values = ["n/a", "na", "--"]
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path,na_values = self.missing_values)

    def explore_data(self):
        """Displays basic information about data like:
        (1) Sample of data
        (2) Features of data
        (3) Shape of dataframe
        (4) Datatypes of attributes
        """
        print('----------------------------------------------------------------')
        print('\nSample of data')
        print(self.dataframe.sample(5))
        print('\nFeatures of data')
        print(list(self.dataframe.columns))
        print('\nShape of data')
        print(self.dataframe.shape)
        print('\nData type of attributes')
        print(self.dataframe.dtypes)
        print('\nMissing value analysis')
        print(self.dataframe.isnull().any())
        print('----------------------------------------------------------------')

    #TODO: Function to handle data coming as a stream from an online source

    def handle_missing_values(self, dataframe, threshold):
        """Function for dealing with missing values.
       Checks for missing values, if fraction of values missing
       exceeds the threshold -> add attribute to drop_attribute list.
       Else -> deal with missing values (Not Implemented because
       there are no missing values currentl).

       Args:
          dataframe: input dataframe
          threshold: threshold for fraction of missing values

       Returns:
           dataframe
       """
        return dataframe
        #raise NotImplementedError

    def clean(self, attribute, threshold=0.5, handle_missing_values=False):
        """ (1) Drops unimportant attributes as given in arguments;
        (2) Handles missing values.
        (3) Normalizes 'cnt' variable.

        Args:
            attribute: list of attributes to be dropped
            threshold: threshold for percentage of missing values

        Returns:
            smallDF: cleaned dataframe
        """
        smallDF = self.dataframe.drop(attribute,axis='columns')
        if handle_missing_values:
            smallDF = self.handle_missing_values(smallDF, threshold)

        #smallDF['cnt'] = smallDF['cnt'].transform(lambda x: math.log(x))

        return smallDF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datafile", help="path to data csv file", type=str)
    parser.add_argument("--save_plot",
                        help="path where visualizations will be saved", type=str)
    args = parser.parse_args()
    datafile = args.datafile
    datahandler = Datahandler(datafile)
    # Get to know the data
    datahandler.explore_data()
    # Generate some visualizations
    visualize(datahandler.dataframe, savedir=args.save_plot)

