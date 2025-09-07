import pandas as pd
import os


class MatrixExcelSaver:
    def __init__(self):
        pass

    @staticmethod
    def save(X, fp, sheet_name='Sheet1', mode='a'):
        df = pd.DataFrame(X)

        if not os.path.exists(fp):
            df.to_excel(fp, sheet_name=sheet_name, index=False)
        elif mode == 'a':
            with pd.ExcelWriter(fp, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        elif mode == 'w':
            os.remove(fp)
            df.to_excel(fp, sheet_name=sheet_name, index=False)


class DictExcelSaver:
    def __init__(self):
        pass

    @staticmethod
    def save(d, fp, sheet_name='Sheet1'):
        df = pd.DataFrame(data=d.values(), index=d.keys())
        df = df.T

        if not os.path.exists(fp):
            df.to_excel(fp, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(fp, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:  
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                

def create_folder(path):
    path_pieces = path.split('/')
    new_path = ''

    for piece in path_pieces:
        new_path = os.path.join(new_path, piece)
        if not os.path.isdir(new_path):
            print('Creating ', new_path)
            os.mkdir(new_path)
