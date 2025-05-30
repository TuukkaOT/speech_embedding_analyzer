from adjustText import adjust_text
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform, pdist
from make_trees import single_tree, consensus_tree, neighbornet, language_family_trees
import pandas as pd
import numpy as np
class Visualizer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
       
        self.metric = config['distance_metric']
        self.no_components = config['no_components']
        if self.no_components == 'max':
            if isinstance(self.data, list):
                self.no_components = len(self.data[0][self.config['transformation_class']].unique()) - 1    
            else:
                self.no_components = len(self.data[self.config['transformation_class']].unique()) - 1

    def get_linkage_matrix(self, df):
        linkage_matrix = linkage(df.loc[:, 'D1':'D'+str(self.no_components)], method=self.config['agglomethod'], metric=self.config['distance_metric'])
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
        linkage_matrix = self.get_linkage_matrix(self.data)
        leaf_names = list(self.data[self.config['analysis_class']])
        # combine linkage matrix and leaf names so that i can send it to MakeTrees class
        combined_data = [linkage_matrix, leaf_names]
        single_tree(combined_data)

    def plot_family_dendrograms(self):

        # group self data by language family and make a list of dataframes
        dataframe = self.data.groupby('language_family')
        # make a list of dataframes from the groupby object
        dataframes = [group for _, group in dataframe]

        # drop dataframes with less than 5 languages
        dataframes = [df for df in dataframe if len(df) >= 5]
        linkage_matrices = [self.get_linkage_matrix(df) for df in dataframes]
        leaf_names_list = []
        family_names_list = []
        for i in range(len(linkage_matrices)):
            leaf_names = list(dataframe[i]['language'])
            family_names = list(dataframe[i]['language_family'])
            leaf_names_list.append(leaf_names)
            family_names_list.append(family_names)

        language_family_trees(linkage_matrices, leaf_names_list, family_names_list)

    def plot_consensus_tree(self):

        linkage_matrices = [self.get_linkage_matrix(df) for df in self.data]
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
        plt.savefig('plots/scatter_plot.png')
        print("Wrote scatter plot to file: plots/scatter_plot.png.")

    def plot_distance_matrix(self):

        family_order = ['Celtic', 
                        'Isolate',
                        'Italic', 
                        'Germanic', 
                        'Balto-Slavic', 
                        'Graeco-Phrygian', 
                        'Classical Indo-European', 
                        'Uralic', 
                        'Armenic', 
                        'Kartvelian', 
                        'Abkhaz-Adyge', 
                        'Afro-Asiatic', 
                        'Turkic', 
                        'Indo-Iranian', 
                        'Dravidian', 
                        'Sino-Tibetan', 
                        'Mongolic-Khitan', 
                        'Koreanic', 
                        'Japonic', 
                        'Tai-Kadai', 
                        'Austroasiatic', 
                        'Austronesian', 
                        'Mande', 
                        'Atlantic-Congo'
                        ]
        
        family_order = ['Celtic', 
                        'Isolate',
                        'Italic',
                        'Artificial_Language',
                        'Germanic', 
                        'Balto_Slavic', 
                        'Graeco_Phrygian', 
                        'Classical_Indo_European', 
                        'Uralic', 
                        'Armenic', 
                        'Kartvelian', 
                        'Abkhaz_Adyge', 
                        'Afro_Asiatic', 
                        'Turkic', 
                        'Indo_Iranian', 
                        'Dravidian', 
                        'Sino_Tibetan', 
                        'Mongolic_Khitan', 
                        'Koreanic', 
                        'Japonic', 
                        'Tai_Kadai', 
                        'Austroasiatic', 
                        'Austronesian', 
                        'Mande', 
                        'Atlantic_Congo'
                        ]
        df = self.data.sort_values(by=['language_family', 'longitude'], ascending=True).reset_index(drop=True)
     
        df = df[~df['language_family'].isin(['Artificial_Language'])]  
        distance_matrix = squareform(pdist(df.loc[:, 'D1':'D'+str(self.no_components)], metric=self.metric))
        language_family_map = df.set_index('language')['language_family'].to_dict()
        ordered_languages = sorted(df['language'], key=lambda lang: family_order.index(language_family_map[lang]))

        confusion_df = pd.DataFrame(distance_matrix, index=df['language'], columns=df['language'])
        ordered_confusion_df = confusion_df.reindex(index=ordered_languages, columns=ordered_languages)
        
        

        fig = plt.figure(figsize=(11, 16))
        sns.heatmap(
        ordered_confusion_df, 
        annot=False, 
        fmt=".2f", 
        cmap="viridis_r", 
        cbar=True,
        cbar_kws={'shrink': 0.5, 'orientation': 'horizontal', 'pad': 0.1}
        )
        ax = plt.gca()
        # Adjust labels
        ax.set_xticks(np.arange(len(ordered_languages)))
        ax.set_xticklabels(ordered_languages, fontsize=7, rotation=90)


        for label in ax.get_xticklabels():
            label.set_horizontalalignment('left')
            label.set_position((label.get_position()[0]-50, label.get_position()[1]))
        
        ax.set_yticks(np.arange(len(ordered_languages)))
        ax.set_yticklabels(ordered_languages, fontsize=6)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('top')
            label.set_position((label.get_position()[0], label.get_position()[1] - 50.))

        # Draw vertical and horizontal lines to separate language families
        prev_family = None
        boundary_positions = []
        for idx, lang in enumerate(ordered_languages):
            current_family = language_family_map[lang]
            if prev_family is not None and current_family != prev_family:
                boundary_positions.append(idx)
            prev_family = current_family

        for pos in boundary_positions:
            ax.hlines(y=pos, xmin=0, xmax=len(ordered_languages), colors='black', linewidth=1.)
            ax.vlines(x=pos, ymin=0, ymax=len(ordered_languages), colors='black', linewidth=1.)

        ax.set_xlabel("")
        ax.set_ylabel("")
        # Adjust tick label padding (closer to the plot)
        ax.tick_params(axis='x', pad=2)  # Reduce x-axis label padding
        ax.tick_params(axis='y', pad=2)  # Reduce y-axis label padding
     
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png')
        #plt.show()
       

    def plot_map(self):

        import folium
        from matplotlib import colors as mcolors
        def add_marker(lang_row, map_obj, color_map):
           
            family = lang_row['language_family']
            color = color_map.get(family, 'blue') 
            folium.CircleMarker(
                location=[lang_row['latitude'], lang_row['longitude']],
                radius=8,  # Small circle
                color=color,
                fill=True,
                opacity=1,
                fill_color=color,
                fill_opacity=1,
                popup=lang_row['language']
                #tooltip=folium.Tooltip(lang_row['iso'], permanent=True)
            ).add_to(map_obj)


        tiles = ' https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg'
        attr= 'Tiles courtesy of the <a href="https://usgs.gov/">U.S. Geological Survey</a>'
        lon, lat = -38.625, -12.875
        m = folium.Map(location=[lat, lon], tiles=tiles, attr=attr, zoom_start=2)

        overlay = folium.raster_layers.ImageOverlay(
            name='darken_overlay',
            image = 'https://upload.wikimedia.org/wikipedia/commons/4/49/A_black_image.jpg',

            bounds=[[-90, -180], [90, 180]],  # covers the entire map
            opacity=0.,  # adjust this to control the level of darkness
            interactive=False,
            zindex=1000
        )

        overlay.add_to(m)
        style = folium.Element("""
        <style>
        .leaflet-tile {
            filter: contrast(75%);
        }
        </style>
        """)
        m.get_root().html.add_child(style)

        
        distance_matrix = self.distance_matrix()
        max_dist = np.quantile(distance_matrix[distance_matrix>0], .05)
        languages = self.data['language'].values
        df = self.data
        unique_families = df['language_family'].unique()
        colors = plt.cm.get_cmap('tab20', len(unique_families)) 
        family_color_map = {family: mcolors.to_hex(colors(i)[:3]) for i, family in enumerate(unique_families)} 
        nn_tracker = {lang: [] for lang in df['language'].unique()}
        mindist_tracker = {lang: [] for lang in df['language'].unique()}
        # Find nearest neighbors for each language
        for idx, lang in enumerate(languages):
            distances_from_lang = distance_matrix[idx]
            distances_from_lang[idx] = np.inf
            nearest_neighbor_idx = np.argpartition(distances_from_lang, 100)[:10]
            nearest_neighbor = languages[nearest_neighbor_idx]
            nn_tracker[lang] = nearest_neighbor
            mindist_tracker[lang] = distances_from_lang[nearest_neighbor_idx]
      
        for lang, neighbor in nn_tracker.items():
            lang_row = df[df['language'] == lang].iloc[0]
         
            for i in range(len(neighbor)):
                neighbor_row = df[df['language'] == neighbor[i]].iloc[0]
                if mindist_tracker[lang][i] > max_dist:
                    continue
                lang_coords = [lang_row['latitude'], lang_row['longitude']]
                neighbor_coords = [neighbor_row['latitude'], neighbor_row['longitude']]
           
                line_weight = (1./(mindist_tracker[lang][i]))*1.5
                line_weight = np.min([8, line_weight])
                opacity = min([1-mindist_tracker[lang][i]+0.2, 1])
                family1 = df.loc[df['language'] == lang, 'language_family'].values[0]
                family2 = df.loc[df['language'] == neighbor[i], 'language_family'].values[0]
                dash_val= 5
                color='white'
                if family1 == family2:
                    dash_val = 0
                    color =  family_color_map.get(family1)
                folium.PolyLine(
                    locations=[lang_coords, neighbor_coords],
                    color=color,
                    weight=line_weight,  # Thickness based on frequency
                    opacity=opacity,
                    dash_array=dash_val
                ).add_to(m)
                add_marker(lang_row, m, family_color_map)
                add_marker(neighbor_row, m, family_color_map)
                folium.Marker(
                    location=[lang_row['latitude'], lang_row['longitude']+1],
                    popup=folium.Popup('<i>The center of map</i>'),
                    tooltip="closest languages:<br>  "+ ", <br>".join(nn_tracker[lang]),
                    icon=folium.DivIcon(
                        html=f"""<div style="color: white; font-size: 11px; background-color: rgba(0, 0, 0, 0.1 );display: inline-block;">{lang_row['language']}</div>"""
                      
                    )).add_to(m)
            #if mindist_tracker[lang][i] < min_dist:
                    
        m.save("languages_map.html")
        input()
    def statistical_analysis(self):

        
        ##########################################
        # Add here the statistical analysis code #
        ##########################################

        pass

if __name__ == "__main__":
    print("This is a module. Please run the main script.")
    pass