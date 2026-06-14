import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

def style(style="whitegrid", context="notebook", palette="tab10") -> None:
    sns.set_theme(context, style, palette)

    Theme._set_style(style)

    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['axes.titlepad'] = 10
    plt.rcParams['axes.labelpad'] = 8
    plt.rcParams['figure.autolayout'] = True 
    plt.rcParams["legend.frameon"]   = False

    if style in ["dark", "darkgrid"]:
        plt.rcParams['figure.facecolor'] = "#171717" 
        plt.rcParams['text.color']       = "#E1E1E1"
        plt.rcParams['axes.facecolor']   = "#171717"
        plt.rcParams['axes.titlesize']   = 16  
        plt.rcParams['axes.edgecolor']   = "#1D1D1D"
        plt.rcParams['axes.labelcolor']  = "#919191"
        plt.rcParams['xtick.color']      = "#636363"
        plt.rcParams['ytick.color']      = "#636363"
        plt.rcParams['patch.edgecolor']  = "#1D1D1D"
        plt.rcParams['patch.linewidth']  = 1


    else:
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelcolor'] = "#575757"
        plt.rcParams['xtick.color'] = "#979797"
        plt.rcParams['ytick.color'] = "#979797"

    if style in ["white", "ticks", "dark"]:
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False

    if style == "darkgrid":
        plt.rcParams['axes.facecolor']   = "#272727"
        plt.rcParams['grid.color']       = "#1D1D1D"
        

class Theme:
    _isdark: bool = False

    @classmethod
    def _set_style(cls, style,):
        cls._isdark = style in ['dark', 'darkgrid']

    @classmethod
    def main(cls):
        return "#5A009B" if cls._isdark else "#420E68"

    @classmethod
    def accent(cls):
        return "#67E8F9" if cls._isdark else "#06B6D4"

    @classmethod
    def success(cls):
        return "#4ADE80" if cls._isdark else "#16A34A"

    @classmethod
    def optimization(cls):
        return "#34D399" if cls._isdark else "#059669"

    @classmethod
    def warning(cls):
        return "#FBBF24" if cls._isdark else "#D97706"

    @classmethod
    def error(cls):
        return "#F87171" if cls._isdark else "#DC2626"

    @classmethod
    def info(cls):
        return "#60A5FA" if cls._isdark else "#2563EB"
    
    @classmethod
    def categorical(cls, n_colors, palette="tab10"):
        return sns.color_palette(palette, n_colors)

    @classmethod
    def categorical_purples(cls, normed_values):
        return cls.purples()(normed_values)

    @classmethod
    def purples(cls):
        if cls._isdark:
            return LinearSegmentedColormap.from_list(
                "darkpurple",
                ["#272727", "#461F5B", "#5A009B"],
                N=256,
            )

        return LinearSegmentedColormap.from_list(
            "lightpurple",
            ["#FFFFFF", "#7832A4", "#420E68"],
            N=256,
        )

    @classmethod
    def heat(cls):
        return "viridis"

    @classmethod
    def heat_ai(cls):
        return sns.color_palette("mako", as_cmap=True)

    @classmethod
    def heat_diverging(cls):
        return sns.color_palette("coolwarm", as_cmap=True)

    @classmethod
    def right_wrong(cls):
        return sns.color_palette([cls.error(), cls.success()])

    @classmethod
    def heat_performance(cls):
        return LinearSegmentedColormap.from_list(
            "performance",
            ["#00C853", "#FFD600", "#FF6D00", "#D50000"],
            N=256,
        )