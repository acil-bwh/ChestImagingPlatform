import pandas as pd
import glob
import os



class LocalHistogramModel:

    def __init__(self, model_list=None):

        self.columns = dict()
        self.columns['Features'] = []
        for i in range(-1024, 1):
            self.columns['Features'].append("hu" + str(i))
        self.columns['Distance'] = ['WholeLungDistance']
        self.columns['Metadata'] = ['cid', 'patch_label', 'coordinates', 'ChestRegion', 'ChestType']

        self.training_df = pd.DataFrame()

        self.model = dict()


        self.model['Features'] = pd.DataFrame(columns=self.columns['Features'])
        self.model['Distance'] = pd.DataFrame(columns=self.columns['Distance'])
        self.model['Metadata'] = pd.DataFrame(columns=self.columns['Metadata'])

        if model_list is not None:
            self.join_models(model_list)



    def create_hdf5_from_model(self, hdf5_filename):
        hdf5_filename = hdf5_filename.split('.')[0] + '.h5'
        if os.path.exists(hdf5_filename):
            raise ValueError("File already exists")

        else:
            with pd.HDFStore(hdf5_filename, 'w') as hdf5_store:
                hdf5_store['Features'] = self.model['Features']
                hdf5_store['Distance'] = self.model['Distance']
                hdf5_store['Metadata'] = self.model['Metadata']

    def read_hdf5(self, hdf5_filename):
        hdf5_filename = hdf5_filename.split('.')[0] + '.h5'
        with pd.HDFStore(hdf5_filename, 'r') as store:
            self.model['Features'] = store['Features']
            self.model['Distance'] = store['Distance']
            self.model['Metadata'] = store['Metadata']

    def read_csv_folder(self, csv_folder, file_suffix):
        self.model['Features'] = self.csv_concat_df(csv_folder, file_suffix, self.columns['Features'])
        self.model['Distance'] = self.csv_concat_df(csv_folder, file_suffix, self.columns['Distance'])
        self.model['Metadata'] = self.csv_merge_df(csv_folder, file_suffix, self.columns['Metadata'])

    @staticmethod
    def csv_concat_df(fol, suf, col):
        fl = glob.glob(os.path.join(fol, '*' + suf + '*'))
        idf = pd.DataFrame(columns=col)
        frames = [idf]

        for i in fl:
            r = pd.read_csv(i)
            frames.append(r)

        df = pd.concat(frames, join='inner', ignore_index=True)
        return df

    @staticmethod
    def csv_merge_df(fol, suf, col):
        fl = glob.glob(os.path.join(fol, '*' + suf))
        idf = pd.DataFrame(columns=col)
        frames = [idf]

        for i in fl:
            s = pd.DataFrame(columns=['cid'])
            csv_file = i.split('/')[-1]
            cid_sp = csv_file.split('_')[0:-1]
            cid = '_'.join(cid_sp)
            r = pd.read_csv(i)
            for j in range(r.shape[0]):
                s.set_value(j, 'cid',  cid)
            result = pd.concat([s, r], axis=1)
            frames.append(result)

        df = pd.concat(frames, join='inner', ignore_index=True)
        return df

    def join_models(self, model_list):
        for key in self.model.keys():
            frames = []
            for i in model_list:
                _model = dict()
                hdf5_filename = i.split('.')[0] + '.h5'
                with pd.HDFStore(hdf5_filename, 'r') as store:
                    _model[key] = store[key]
                frames.append(_model[key])
            self.model[key] = pd.concat(frames, axis=0, join='inner', ignore_index=True)

    def get_training_df_from_model(self):
        frames = []
        for key in self.model.keys():
            frames.append(self.model[key])
        if len(frames)>1:

            self.training_df = pd.concat(frames, axis=1, join='outer')
           # self.training_df.to_csv('prueba.csv', index=False)
        else:
            self.training_df = self.model

        return self.training_df
