import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from visualizer import Visualizer

class Dataset:
    def __init__(self, data, config):
        self.data = data
        self.classes = config['transformation_class']
        self.analysis_class = config['analysis_class']
        self.components = config['no_components']
        self.method = config['method']
        self.len_embeddings = config['len_embeddings']
        self.random_seed = config['random_seed']
        self.sample_size = config['sample_size']
        self.config = config
        self.analysis_method = config['analysis']
        
    def visualize(self):
        print("Visualizing data")
        visualizer = Visualizer(self.data, self.config)
        # Call the appropriate visualization method, based on the visualization_type
        if self.analysis_method == 'd':
            visualizer.plot_dendrogram()
        elif self.analysis_method == 'n':
            visualizer.plot_neighbornet()
        elif self.analysis_method == 'f':
            visualizer.plot_family_dendrograms()
        elif self.analysis_method == 'c':
            visualizer.plot_consensus_tree()
        elif self.analysis_method == 's':
            visualizer.plot_scatter()
        elif self.analysis_method in ['a']:
            visualizer.statistical_analysis()

    def sample_dataset(self):
        if self.analysis_method == 'c':
            # keep only languages with at least sample_size samples
            self.data = self.data.groupby('language').filter(lambda x: len(x) >= 1000)
            
            # sample exactly 1000 samples from each language
            self.data = (
                self.data.groupby('language', group_keys=False)
                .apply(lambda x: x.sample(n=1000, random_state=self.random_seed))
            )
        else:
            self.data = self.data.groupby('language').filter(lambda x: len(x) >= self.sample_size)
            self.data = (
                    self.data.groupby('language', group_keys=False)
                    .apply(lambda x: x.sample(n=min(len(x), 1000), random_state=self.random_seed))
                    )
            print("Number of unique languages with minimum"+str(self.sample_size)+" samples: ", len(self.data['language'].unique()))
        return self

        
    def chunks_for_consensus_tree(self):
        dataframe_list = []
        languages = self.data['language'].unique()
        for chunk_index in range(20):
                chunk_data = pd.DataFrame()
                for language in languages:
                    language_data = self.data[self.data['language'] == language]
                    start_index = chunk_index * 50
                    end_index = start_index + 50
                    chunk_data = pd.concat([chunk_data, language_data.iloc[start_index:end_index]], ignore_index=True)
                dataframe_list.append(chunk_data)
        self.data = dataframe_list
        return self


    def add_metadata(self):
        # add coordinates to data
        # read 'data/coordinates.csv' file
        language_metadata = pd.read_csv('data/iso_coordinates.tsv', sep='\t')
        
        if 'language_family' in self.data.columns:
            print("Data already contains language families.")
            # drop language_family column
            language_metadata = language_metadata.drop(columns=['language_family'])
        else:
            print("Data does not contain language families. Adding language families.")

        if 'latitude' in self.data.columns and 'longitude' in self.data.columns:
            print("Data already contains coordinates.")
            # drop latitude and longitude columns
            language_metadata = language_metadata.drop(columns=['latitude', 'longitude'])
        else:
            print("Data does not contain coordinates. Adding coordinates.")
            

        # add coordinates to data
        self.data = self.data.merge(language_metadata, on='iso', how='left')
        
        # Reorder columns
        first_columns = ['iso', 'latitude', 'longitude', 'language', 'language_family']
        other_columns = [col for col in self.data.columns if col not in first_columns]
        self.data = self.data[first_columns + other_columns]

        # if there are any missing coordinates, print the rows and break the script
        if self.data['latitude'].isnull().any():
            print("There are missing coordinates in the data.")
            # print rows where coordinates are missing
            missing_coordinates = self.data[self.data['latitude'].isnull()]
            print("Missing coordinates for the following languages:")
            print(missing_coordinates[['language', 'iso']])
            # break the script
            raise ValueError("Missing coordinates in the data. Make sure your iso codes match Glottolog iso codes.")
        # if there are any missing language families, print the rows and break the script
        if self.data['language_family'].isnull().any():
            print("There are missing language families in the data.")

            # print unique values of languages where language_family is null
            missing_language_families = self.data[self.data['language_family'].isnull()]
            
            print("Missing language families for the following languages:")
            print(missing_language_families[['language', 'iso']].groupby('language').first())
            # break the script
            raise ValueError("Missing language families in the data. Make sure your iso codes match Glottolog iso codes.")

    def prepare_languages(self):

        # check if self.data contains columns 'latitude' and 'longitude' and/or 'language_family'
        if 'latitude' in self.data.columns and 'longitude' and 'language_family' in self.data.columns:
            print("Data contains coordinates and language families.")
        else:
            print("Data does not contain coordinates or language families. Retrieveing coordinates from Glottolog file.")
            self.data = self.add_metadata().data
            return
        

        # make all columns into strings
        self.data = self.data.map(str)
        
        # trim leading and trailing whitespaces
        self.data['language'] = self.data['language'].str.strip()
        self.data['language_family'] = self.data['language_family'].str.strip()
        # replace '-' and ' ' with '_'
        self.data['language'] = self.data['language'].str.replace('-', '_').str.replace(' ', '_')
        self.data['language_family'] = self.data['language_family'].str.replace('-', '_').str.replace(' ', '_')
        # make iso as string
        
        self.data['iso'] = self.data['iso'].astype(str)
        return self

    def scale_data(self):
        
        # Standardize data
        scaler = StandardScaler(with_std=False)
        last_emb_dim = 'D' + str(self.len_embeddings)
        self.data.loc[:, 'D1':last_emb_dim] = scaler.fit_transform(self.data.loc[:, 'D1':last_emb_dim])
        if self.analysis_method == 'r':
            self.chunks_for_robustness()
        if self.analysis_method == 'c':
            self.chunks_for_consensus_tree()
        return self
    

    def dimensionality_reduction(self, data):
        
        if self.components == 'max':
            no_components = len(data[self.classes].unique()) - 1
        else:
            no_components = int(self.components)
        X = data.loc[:, 'D1':'D'+str(self.len_embeddings)]
    
        y = data.loc[:, self.classes]
        
        if self.method == 'pca':
            pca = PCA(n_components=no_components)
            pca_components = pca.fit_transform(X)
            component_name = str(no_components)
            #drop_component = str(no_components+1)
            # drop columns from '1' until and including 'self.len_embeddings'
            start = data.columns.get_loc('D1')
            end = data.columns.get_loc('D'+str(self.len_embeddings))  # Include the end column
            data = data.drop(columns=data.columns[start:end])
            pca_columns = ['D'+str(i) for i in range(1, pca_components.shape[1] + 1)]
            pca_df = pd.DataFrame(pca_components, columns=pca_columns, index=data.index)
            # Update `self.data`
            data = pd.concat([data, pca_df], axis=1).copy()

        if self.method == 'lda':
            lda = LinearDiscriminantAnalysis(n_components=no_components)
            lda_components = lda.fit(X, y).transform(X)
            component_name = str(no_components)
            start = data.columns.get_loc('D1')
            end = data.columns.get_loc('D'+str(int(self.len_embeddings)))  # Include the end column
            data = data.drop(columns=data.columns[start:end])

            # add lda components to data as columns '1' onwards
            lda_columns = ['D'+str(i) for i in range(1, lda_components.shape[1] + 1)]
            lda_df = pd.DataFrame(lda_components, columns=lda_columns, index=data.index)

            # Update `self.data`
            data = pd.concat([data, lda_df], axis=1).copy()

        return data
        
    
    def apply_dimensionality_reduction(self):
        if isinstance(self.data, list):
            print("Applying dimensionality reduction to chunks.")
            for i in range(len(self.data)):
                self.components = len(self.data[i][self.classes].unique()) - 1
                self.data[i] = self.dimensionality_reduction(self.data[i])
        else:
            print("Applying dimensionality reduction to whole dataset.")
            self.data = self.dimensionality_reduction(self.data)
        return self
        
    
    def get_means(self):
        if isinstance(self.data, list):
            print("Calculating means for chunks")
            for i in range(len(self.data)):
                self.data[i] = self.data[i].groupby([self.analysis_class] + ['language_family', 'iso']).mean(numeric_only=True).loc[:, 'D1':].reset_index()
        else:
            print("Calculating means for whole dataset")
            self.data = (
                self.data
                .groupby([self.analysis_class] + ['language_family', 'iso'])
                .mean(numeric_only=True)
                .loc[:, 'D1':].
                reset_index()
            )
        return self
    
if __name__ == "__main__":
    pass