import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

def style(style="whitegrid", context="notebook", palette="tab10") -> None:
    sns.set_theme(context, style, palette)

    color_maps._set_style(style, palette)

    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['axes.titlepad'] = 10
    plt.rcParams['axes.labelpad'] = 8
    plt.rcParams['figure.autolayout'] = True 

    if style in ['dark', 'darkgrid']:
        plt.rcParams['figure.facecolor'] = "#171717" 
        plt.rcParams['text.color'] = "#E1E1E1"
        plt.rcParams['axes.titlesize'] = 16  
        plt.rcParams['axes.facecolor'] =  "#272727"
        plt.rcParams['axes.edgecolor'] = "#1D1D1D"
        plt.rcParams['grid.color'] = "#1D1D1D"
        plt.rcParams['axes.labelcolor'] = "#919191"
        plt.rcParams['xtick.color'] = "#636363"
        plt.rcParams['ytick.color'] = "#636363"

    else:
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelcolor'] = "#575757"
        plt.rcParams['xtick.color'] = "#979797"
        plt.rcParams['ytick.color'] = "#979797"

    if  style == "white" or style == "ticks":
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        
class ColorMap:
    def __init__(self):
        self._isdark = False
        self._palette = None

    def _set_style(self, style, palette):
        self._isdark = style in ['dark', 'darkgrid']
        self._palette = palette

    def purples(self):
        if self._isdark:
            return LinearSegmentedColormap.from_list("darkpurple", ["#272727", "#461F5B", "#5A009B"], N=256)
        
        return LinearSegmentedColormap.from_list("lightpurple", ["#FFFFFF", "#A020F0", "#4B0082"], N=256)
    
    def categorical(self, n_colors, palette=None):
        if self._isdark:
            return sns.color_palette(palette or self._palette, n_colors)
        
        return sns.color_palette(palette or self._palette, n_colors)

color_maps = ColorMap()