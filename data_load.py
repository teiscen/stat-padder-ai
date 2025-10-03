import pandas as pd
import os

class CSV_Data:
    def __init__(self, filePath, colDropList, function=None):
        self.filePath = filePath        
        self.colDropList = colDropList
        self.function = function
        self.csv_data  = None
    
    def __del__(self):
        self.delData()

    def readFile(self):
        try:
            self.csv_data = pd.read_csv(self.filePath)
            self.csv_data = self.csv_data.drop(self.colDropList, axis=1)
        except FileNotFoundError:
            print(f"Error: File not found - {self.filePath}")
            self.csv_data = None
        except pd.errors.EmptyDataError:
            print(f"Error: No data - {self.filePath}")
            self.csv_data = None
        except KeyError as e:
            print(f"Error: Column(s) not found when dropping: {e}")
            self.csv_data = None
        except Exception as e:
            print(f"Unexpected error reading {self.filePath}: {e}")
            self.csv_data = None

        if not self.function is None:
            self.function(self.csv_data)

    def getData(self):
        if self.csv_data is None:
            print(f"Warning: No data loaded from {self.filePath}")
        return self.csv_data

    def delData(self):
        try:
            del self.csv_data
            self.csv_data = None
        except AttributeError:
            print("Error: csv_data attribute does not exist.")
        except Exception as e:
            print(f"Unexpected error deleting csv_data: {e}")

    def printDebug(self):
        # baseName -> get the last part; splitext-> name.txt into [name, txt] 
        name = os.path.splitext(os.path.basename(self.filePath))[0]
        print(f"Debug: {name}, Columns:\n{self.csv_data.columns.tolist()}")

    @staticmethod
    def merge_csv_list(csv_list, merge_list):
        merge_list = [None] + merge_list
        if len(csv_list) != len(merge_list):
            raise ValueError("Length of csv_list must be one less than length of merge_list.")

        merged_data = None
        # Skip the first CSV in csv_list; start from index 1
        for csv, merge in csv_list, merge_list:
            if csv.getData() is None: csv.readFile()
            if merged_data is None:
                merged_data = csv.readFile()
                continue
            merged_data = pd.merge(merged_data, csv.getData(), on=merge, how='inner')

        del csv_list, merge_list
        return merged_data

