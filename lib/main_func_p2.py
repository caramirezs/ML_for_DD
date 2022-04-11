from lib.old_main_func_p1 import path


def get_properties_stats(data_df):
    import pandas as pd
    """
    Function that calculates the mean and standard deviation of physicochemical properties of a dataset.

    Input:
    Dataset containing per compound values for physicochemical properties
    HBD, HBA, MW and LogP as columns (with exactly these names).

    Output:
    Dataframe with mean and std (columns) for each physicochemical property (rows).
    """
    properties = ["HBD", "HBA", "MW", "LogP"]

    data_stats = []

    for i in properties:
        std = data_df[i].std()
        mean = data_df[i].mean()
        da = pd.DataFrame([[round(mean, 2), round(std, 2)]], index=[i], columns=["mean", "std"])
        data_stats.append(da)

    data_stats = pd.concat(data_stats)

    return data_stats


def plot_radarplot(uniprot, data_stats):
    """
    Function that plots a radar plot based on the mean and std of 4 physicochemical properties (HBD, HBA, MW and LogP).

    Input:
    Dataframe with mean and std (columns) for each physicochemical property (rows).

    Output:
    Radar plot (saved as file and shown in Jupyter notebook).
    """
    from math import pi
    # matplotlib:
    import matplotlib.pyplot as plt
    # Get data points for lines

    path_file = path(uniprot)

    std_1 = [data_stats["mean"]["HBD"] + data_stats["std"]["HBD"],
             (data_stats["mean"]["HBA"]/2) + (data_stats["std"]["HBA"]/2),
             (data_stats["mean"]["MW"]/100) + (data_stats["std"]["MW"]/100),
             data_stats["mean"]["LogP"] + data_stats["std"]["LogP"]]
    std_2 = [data_stats["mean"]["HBD"] - data_stats["std"]["HBD"],
             (data_stats["mean"]["HBA"]/2) - (data_stats["std"]["HBA"]/2),
             (data_stats["mean"]["MW"]/100) - (data_stats["std"]["MW"]/100),
             data_stats["mean"]["LogP"] - data_stats["std"]["LogP"]]
    mean_val = [data_stats["mean"]["HBD"], (data_stats["mean"]["HBA"]/2),
                (data_stats["mean"]["MW"]/100), data_stats["mean"]["LogP"]]

    # Get data points for (filled) area (rule of five)
    rule_conditions = [5, (10/2), (500/100), 5]

    # Define property names
    parameters = ['# H-bond donors', '# H-bond acceptors/2', 'Molecular weight (Da)/100', 'LogP']

    #
    N = len(rule_conditions)

    # Set font size
    fontsize = 16

    # Angles for the condition axes
    x_as = [n / float(N) * 2 * pi for n in range(N)]

    # Since our chart will be circular we need to append a copy of the first
    # Value of each list at the end of each list with data
    std_1 += std_1[:1]
    std_2 += std_2[:1]
    mean_val += mean_val[:1]
    rule_conditions += rule_conditions[:1]
    x_as += x_as[:1]

    # Set figure size
    plt.figure(figsize=(8,8))

    # Set color of axes
    plt.rc('axes', linewidth=2, edgecolor="#888888")

    # Create polar plot
    ax = plt.subplot(111, polar=True)

    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set position of y-labels
    ax.set_rlabel_position(0)

    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=2)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=2)

    # Set number of radial axes and remove labels
    plt.xticks(x_as[:-1], [])

    # Set yticks
    plt.yticks([1, 3, 5, 7], ["1", "3", "5", "7"], size=fontsize)

    # Set axes limits
    plt.ylim(0, 7)

    # Plot data
    # Mean values
    ax.plot(x_as, mean_val, 'b', linewidth=3, linestyle='solid', zorder=3)

    # Standard deviation
    ax.plot(x_as, std_1, linewidth=2, linestyle='dashed', zorder=3, color='#111111')
    ax.plot(x_as, std_2, linewidth=2, linestyle='dashed', zorder=3, color='#333333')

    # Fill area
    ax.fill(x_as, rule_conditions, "#3465a4", alpha=0.2)

    # Draw ytick labels to make sure they fit properly
    for i in range(N):
        angle_rad = i / float(N) * 2 * pi
        if angle_rad == 0:
            ha, distance_ax = "center", 1
        elif 0 < angle_rad < pi:
            ha, distance_ax = "left", 1
        elif angle_rad == pi:
            ha, distance_ax = "center", 1
        else:
            ha, distance_ax = "right", 1
        ax.text(angle_rad, 7 + distance_ax, parameters[i], size=fontsize,
                horizontalalignment=ha, verticalalignment="center")

    # Add legend relative to top-left plot
        labels = ('Mean', 'Mean + std', 'Mean - std', 'Rule of five area')
        legend = ax.legend(labels, loc=(1.1, .7),
                           labelspacing=0.3, fontsize=fontsize)
    plt.tight_layout()

    # Save plot - use bbox_inches to include text boxes:
    output_path = f'{path_file}_RadarPlot'
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)

    # Show polar plot
    plt.show()

    plt.close()