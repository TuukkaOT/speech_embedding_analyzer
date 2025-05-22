from adjustText import adjust_text
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform, pdist
from make_trees import single_tree, consensus_tree, neighbornet, language_family_trees

class Visualizer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.no_components = config['no_components']
        self.metric = config['distance_metric']

        if self.no_components == 'max':
            if isinstance(self.data, list):
                self.config['no_components'] = len(self.data[0][self.config['transformation_class']].unique()) - 1    
            else:
                self.config['no_components'] = len(self.data[self.config['transformation_class']].unique()) - 1
            
    def get_linkage_matrix(self):    
        # check if self.data is a list
        if isinstance(self.data, list):
            linkage_matrix = []
            for i in range(len(self.data)):
                last_component = len(self.data[i][self.config['transformation_class']].unique()) - 1
                linkage_matrix.append(linkage(self.data[i].loc[:, 'D1':'D'+str(last_component)], method=self.config['agglomethod'], metric=self.config['distance_metric']))
            
        else:
            linkage_matrix = linkage(self.data.loc[:, 'D1':'D'+str(self.config['no_components'])], method=self.config['agglomethod'], metric=self.config['distance_metric'])
        return linkage_matrix
    
    def distance_matrix(self):
        distances = pdist(self.data.loc[:, 'D1':'D'+str(self.no_components)], metric=self.metric)
        distance_matrix = squareform(distances)
        return distance_matrix

    def plot_neighbornet(self):

        labels = list(self.data[self.config['analysis_class']])
        distance_matrix = self.distance_matrix()

        neighbornet(distance_matrix, labels)

    def plot_dendrogram(self):
        linkage_matrix = self.get_linkage_matrix()
        leaf_names = list(self.data[self.config['analysis_class']])
        # combine linkage matrix and leaf names so that i can send it to MakeTrees class
        combined_data = [linkage_matrix, leaf_names]
        single_tree(combined_data)

    def plot_family_dendrograms(self):

        # group self data by language family and make a list of dataframes
        self.data = self.data.groupby('language_family')
        # make a list of dataframes from the groupby object
        self.data = [group for _, group in self.data]

        # drop dataframes with less than 5 languages
        self.data = [df for df in self.data if len(df) >= 5]
        linkage_matrices = self.get_linkage_matrix()
        leaf_names_list = []
        family_names_list = []
        for i in range(len(linkage_matrices)):
            leaf_names = list(self.data[i]['language'])
            family_names = list(self.data[i]['language_family'])
            leaf_names_list.append(leaf_names)
            family_names_list.append(family_names)

        language_family_trees(linkage_matrices, leaf_names_list, family_names_list)

    def plot_consensus_tree(self):

        linkage_matrices = self.get_linkage_matrix()
        leaf_names_list = []
        family_names_list = []
        for i in range(len(linkage_matrices)):
            leaf_names = list(self.data[i]['language'])
            family_names = list(self.data[i]['language_family'])
            leaf_names_list.append(leaf_names)
            family_names_list.append(family_names)

        consensus_tree(linkage_matrices, leaf_names_list, family_names_list)

    def add_colors(self):        
        # get unique language families
        unique_families = self.data['language_family'].unique()
        
        # map tab20 colors to language families
        tab20 = plt.get_cmap('tab20')
        colors_a = tab20.colors
        tab20_b = plt.get_cmap('tab20b')
        colors_b = tab20_b.colors

        # join the two color maps
        colors = colors_a + colors_b

        colors = colors[:len(unique_families)]
        # make colors into strings
        colors = ["".join([format(int(i*255), '02x') for i in color]) for color in colors]
        colors_dict = dict(zip(unique_families, colors))
        # add color column to data
        self.data['color'] = self.data['language_family'].map(colors_dict)
        
        self.data['color'] = '#' + self.data['color']
        # put color to index 2
        color_column = self.data.pop('color')
        self.data.insert(2, 'color', color_column)
        return self

    def plot_scatter(self):
        
        # add colors to data
        self.add_colors()

        # plot scatter plot using '1' and '2' columns
        plt.figure(figsize=(14, 10))

        sns.scatterplot(data=self.data, x='D1', y='D2', hue='color', palette=list(self.data['color'].unique()))
        plt.gca().invert_xaxis()
        texts = []
        for i, txt in enumerate(self.data[self.config['analysis_class']]):
            texts.append(plt.text(self.data['D1'].iloc[i], self.data['D2'].iloc[i], txt, fontsize=12))
        adjust_text(
            texts,
            arrowprops=dict(
                arrowstyle='-', 
                color='gray', 
                connectionstyle='arc3,rad=0'
            ))
        plt.title('')
        # no legend
        plt.legend([],[], frameon=False)
        # save plot
        plt.savefig('../plots/scatter_plot.png')
        print("Wrote scatter plot to file: ../plots/scatter_plot.png.")

    def statistical_analysis(self):
        ##########################################
        # Add here the statistical analysis code #
        ##########################################
        pass