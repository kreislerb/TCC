import pandas as pd
import numpy as np
from datetime import datetime

class CreateDataset:

    rows = list()
    columns = []
    index = []
    patch_out = ''
    title = ''
    date_in_title = True
    date = ''

    def __init__(self, patch_out, title):
        self.patch_out = patch_out
        self.title = title

    def insert_row(self, data_row):
        self.rows.append(data_row)

    def set_columns(self, columns):
        self.columns.clear()
        self.columns.append(columns)

    def set_index(self, index):
        self.index.clear()
        self.index.append(index)

    def set_date_in_title(self, enable):
        self.date_in_title = enable

    def save(self):
        if self.date_in_title:
            date = datetime.now()
            self.date = str(date.day) + '-' + str(date.month) + '-' + str(date.year) + '  ' + str(
                date.hour) + ' ' + str(
                date.minute) + ' ' + str(date.second)
        if len(self.index) == 0:
            self.index = None
        if len(self.columns) == 0:
            self.columns = None
        df = pd.DataFrame(np.asarray(self.rows), self.index, self.columns)
        filename = self.patch_out+self.title+' '+self.date+'.csv'
        df.to_csv(filename, sep=",", line_terminator='\n', encoding='utf-8')
        df = df.iloc[0:0]







