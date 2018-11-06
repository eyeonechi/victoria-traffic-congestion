import codecs as cd
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as bm
import numpy as np
import pandas as pd
import pylab as pl
import scipy.spatial.distance as ds
import seaborn as sns
import sklearn.preprocessing as sp
import sklearn.decomposition as sd
import visual as vi

# Handles initial processing of raw datasets
def processRawData():
    lanes, speed, traffic = openRawData()
    lanes = processLanes(lanes)
    speed = processSpeed(speed)
    traffic = processTraffic(traffic)
    writeProcessedData(lanes, speed, traffic)

# Extracts data from source files
def openRawData():
    with cd.open("../data/raw/lanes.csv", 'r', 'utf-8-sig') as data1:
        lanes = pd.read_csv(data1)
        data1.close()
    with cd.open("../data/raw/speed.csv", 'r', 'utf-8-sig') as data2:
        speed = pd.read_csv(data2)
        data2.close()
    with cd.open("../data/raw/traffic.csv", 'r', 'utf-8-sig') as data3:
        traffic = pd.read_csv(data3)
        data3.close()
    return lanes, speed, traffic

# Cleans data in the lanes dataset
def processLanes(data):
    data = data.rename(columns={"OBJECTID": "id", "ROAD_NUMBER": "number",
        "ROAD_NAME": "name", "DIRECTION": "direction", "ELEMENTS": "elements",
        "SURFACE_TYPE": "surface", "NUMBER_OF_TRAFFIC_LANES": "lanes",
        "LEFT_SHOULDER_WIDTH": "left_width", "TRAFFIC_WIDTH": "traffic_width",
        "RIGHT_SHOULDER_WIDTH": "right_width", "SEAL_WIDTH": "width"})
    data.name = data.name.replace(to_replace=[" FWY", " HWY", " RD", " ST"],
        value=[" FREEWAY", " HIGHWAY", " ROAD", " STREET"], regex=True)
    data.sort_values("name", inplace=True)
    return data

# Cleans data in the speed dataset
def processSpeed(data):
    data = data.rename(columns={"NB_TRAFFIC_SURVEY": "survey",
        "NB_LOCATION_TRAFFIC_SURVEY": "survey_location",
        "ID_HOMOGENEOUS_FLOW": "id", "DS_HOMOGENEOUS_FLOW": "name",
        "QT_SPEED_LIMIT": "limit", "NB_YEAR": "year", "SPEED_STAT": "stat",
        "HOUR_OF_DAY": "hour", "SPEED": "speed",
        "NB_GPS_LATITUDE": "latitude", "NB_GPS_LONGITUDE": "longitude"})
    data.name = data.name.apply(lambda x: str(x).upper().lstrip(' '))
    data.name = data.name.replace(to_replace=[" N BD", " S BD", " E BD",
        " W BD", " NE BD", " NW BD", " SE BD", " SW BD",  " NEBD ", " NWBD",
        " SEBD", " SWBD "], value=['', '', '', '', '', '', '', '', '', '',
        '', ''], regex=True)
    data.name = data.name.replace(to_replace=[" NORTH ", " SOUTH ", " EAST ",
        " WEST "], value=['', '', '', ''], regex=True)
    data.name = data.name.replace(to_replace=[" FWY ", " HWY ", " RD ",
        " ST "], value=[" FREEWAY", " HIGHWAY", " ROAD",
        " STREET"], regex=True)
    data.name = data.name.apply(lambda x: str(x).split("BTWN")[0].rstrip(' '))
    data.name = data.name.apply(lambda x: str(x).split("TO")[0].rstrip(' '))
    data.name = data.name.apply(lambda x: str(x).split("FROM")[0].rstrip(' '))
    data.sort_values("name", inplace=True)
    return data

# Cleans data in the traffic dataset
def processTraffic(data):
    data = data.rename(columns={"OBJECTID": "object",
        "LOCATION_ID": "location", "MIDPNT_LAT": "latitude",
        "MIDPNT_LON": "longitude", "HMGNS_FLOW_ID": "flow",
        "HMGNS_LNK_ID": "id", "HMGNS_LNK_DESC": "name", "FLOW": "flow",
        "ALLVEHS_MMW": "median", "ALLVEH_CALC": "vehicle_calc",
        "ALLVEHS_AADT": "vehicles", "TRUCKS_AADT": "trucks",
        "TRUCK_CALC": "truck_calc", "PER_TRUCKS": "truck_percent",
        "TWO_WAY_AADT": "two_way_count", "ALLVEH_AMPEAK": "am_peak",
        "ALLVEH_PMPEAK": "pm_peak", "GROWTH_RATE": "growth", "CI": "ci",
        "YR": "year", "LABEL": "label"})
    data.name = data.name.apply(lambda x: str(x).upper().lstrip(' ').split(
        "BTWN")[0].rstrip(' '))
    data.name = data.name.replace(to_replace=[" N BD", " S BD", " E BD",
        " W BD", " NE BD", " NW BD", " SE BD", " SW BD",  " NEBD ", " NWBD",
        " SEBD", " SWBD "], value=['', '', '', '', '', '', '', '', '', '',
        '', ''], regex=True)
    data.name = data.name.replace(to_replace=[" BVD", " FWY", " HWY", " PDE",
        " RD"], value=[" BOULEVARD", " FREEWAY", " HIGHWAY", " PARADE",
        " ROAD"], regex=True)
    data.sort_values("name", inplace=True)
    return data

# Saves processed datasets to file
def writeProcessedData(lanes, speed, traffic):
    lanes.to_csv("processed/lanes_processed.csv")
    speed.to_csv("processed/speed_processed.csv")
    traffic.to_csv("processed/traffic_processed.csv")

# Handles merging duplicates in datasets
def processProcessedData():
    lanes, speed, traffic = openProcessedData()
    lanes = mergeLanes(lanes)
    speed = mergeSpeed(speed)
    traffic = mergeTraffic(traffic)
    writeMergedData(lanes, speed, traffic)

# Parses data that has been cleaned
def openProcessedData():
    with cd.open("../data/processed/lanes_processed.csv", 'r', 'utf-8-sig') as data1:
        lanes = pd.read_csv(data1)
        data1.close()
    with cd.open("../data/processed/speed_processed.csv", 'r', 'utf-8-sig') as data2:
        speed = pd.read_csv(data2)
        data2.close()
    with cd.open("../data/processed/traffic_processed.csv", 'r', 'utf-8-sig') as data3:
        traffic = pd.read_csv(data3)
        data3.close()
    return lanes, speed, traffic

# Removes duplicates in the lanes dataset
def mergeLanes(data):
    data = data.groupby("name").mean()
    return data

# Removes duplicates in the speed dataset
def mergeSpeed(data):
    data = data.groupby("name").mean()
    return data

# Removes duplicates in the traffic dataset
def mergeTraffic(data):
    data = data.groupby("name").mean()
    return data

# Saves merged datasets to file
def writeMergedData(lanes, speed, traffic):
    lanes.to_csv("merged/lanes_merged.csv")
    speed.to_csv("merged/speed_merged.csv")
    traffic.to_csv("merged/traffic_merged.csv")

# Handles joining the datasets into one
def concatenateData():
    lanes, speed, traffic = openMergedData()
    merged = processConcatenation(lanes, speed, traffic)
    writeConcatenatedData(merged)

# Parses merged datasets
def openMergedData():
    with cd.open("../data/merged/lanes_merged.csv", 'r', 'utf-8-sig') as data1:
        lanes = pd.read_csv(data1)
        data1.close()
    with cd.open("../data/merged/speed_merged.csv", 'r', 'utf-8-sig') as data2:
        speed = pd.read_csv(data2)
        data2.close()
    with cd.open("../data/merged/traffic_merged.csv", 'r', 'utf-8-sig') as data3:
        traffic = pd.read_csv(data3)
        data3.close()
    return lanes, speed, traffic

# Combines the datasets based on matching road names
def processConcatenation(lanes, speed, traffic):
    data = pd.merge(pd.merge(lanes, traffic, on="name"), speed, on="name")
    data.drop(["left_width", "right_width", "traffic_width", "number",
        "object", "location", "flow", "survey", "survey_location", "id",
        "median"], axis=1, inplace=True)
    data.lanes = data.lanes.astype(int)
    data["type"] = data.name.apply(lambda x: roadIdentify(x))
    return data

def roadIdentify(road):
    if "ROAD" in road:
        return "Road"
    elif "STREET" in road:
        return "Street"
    elif "HIGHWAY" in road:
        return "Highway"
    elif "FREEWAY" in road:
        return "Freeway"

# Saves concatenated dataset to file
def writeConcatenatedData(data):
    data = data.rename(columns={"latitude_x": "latitude",
        "longitude_x": "longitude"})
    data.drop(["Unnamed: 0", "Unnamed: 0_x", "Unnamed: 0_y", "id_x", "id_y",
        "year_x", "year_y", "latitude_y", "longitude_y"], axis=1, inplace=True)
    data.to_csv("concatenated/concat.csv")

# Handles graph plotting and other visualisations
def visualiseData():
    data = openConcatenatedData()
    plot1(data)
    plot2(data)
    plot3(data)
    plot4(data)
    plot5(data)
    plot6(data)
    plot7(data)
    plotVAT(data)
    plotMap(data)

# Parses the concatenated dataset
def openConcatenatedData():
    with cd.open("../data/concatenated/concat.csv", 'r', 'utf-8-sig') as data:
        concat = pd.read_csv(data)
        data.close()
    concat.drop("Unnamed: 0", axis=1, inplace=True)
    return concat

# Plot of Average Speed vs Number of Traffic Lanes
def plot1(data):
    data = data.groupby("lanes", as_index=False).mean()
    ax = plt.subplot()
    ax.plot(data.lanes.tolist(), data.speed.tolist(), 'k', color="blue",
        label="Average Speed (km/h)")
    ax.plot(data.lanes.tolist(), data.limit.tolist(), 'k--', color="red",
        label="Average Speed Limit (km/h)")
    plt.xlabel("Number of Traffic Lanes")
    plt.title("Average Speed vs Number of Traffic Lanes")
    plt.legend()
    plt.show()

# Plot of Top 10 and Bottom 10 Average Annual Vehicle Counts on Roads
def plot2(data):
    data.sort_values("vehicles", ascending=False, inplace=True)
    plt.subplots()
    width = 0.4
    plt.barh(np.arange(10), data.vehicles.tolist()[:10], width,
        align="center", color="blue", label="Vehicles")
    plt.barh(np.arange(10) + width, data.trucks.tolist()[:10], width,
        align="center", color="red", label="Trucks")
    plt.yticks(np.arange(10) + width, data.name.tolist()[:10])
    plt.gca().invert_yaxis()
    plt.xlabel("Average Annual Vehicle Count")
    plt.ylabel("Road Name")
    plt.title("Top 10 Average Annual Vehicle Counts on Roads")
    plt.legend(loc="best")
    plt.show()
    plt.barh(np.arange(10), data.vehicles.iloc[::-1].tolist()[:10], width,
        align="center", color="blue", label="Vehicles")
    plt.barh(np.arange(10) + width, data.trucks.iloc[::-1].tolist()[:10],
        width, align="center", color="red", label="Trucks")
    plt.yticks(np.arange(10) + width, data.name.iloc[::-1].tolist()[:10])
    plt.gca().invert_yaxis()
    plt.xlabel("Average Annual Vehicle Count")
    plt.ylabel("Road Name")
    plt.title("Bottom 10 Average Annual Vehicle Counts on Roads")
    plt.legend(loc="best")
    plt.show()

# Plot of Traffic Density vs Speed on Roads
def plot3(data):
    density = pd.Series(data.vehicles / data.width)
    color = data.speed
    plt.scatter(data.speed, density, edgecolor='w', c=color,
        cmap="gist_heat", s=25)
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Traffic Density")
    plt.title("Traffic Density vs Speed on Roads")
    plt.show()

# Plot of Lane Width vs Speed on Victorian Roads
def plot4(data):
    ratio = pd.Series(data.width / data.lanes)
    x = data.speed.tolist()
    y = ratio.tolist()
    color = data.limit.tolist()
    area = data.lanes.astype(int).multiply(25).tolist()
    plt.scatter(x, y, alpha=0.75, edgecolor='w', c=color, cmap="jet", s=area)
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Lane Width (m)")
    plt.title("Lane Width vs Speed on Victorian Roads")
    plt.colorbar(label="Speed Limit")
    plt.show()

# Plot of Average Vehicles and Trucks on Types of Roads
def plot5(data):
    data = data.groupby("type", as_index=False).mean()
    data.plot(x="type", y=["speed", "limit"], kind="line", legend=True,
              title="Average Speed on Types of Roads")
    data.plot(x="type", y=["vehicles", "trucks"], kind="bar", legend=True,
              title="Average Vehicles and Trucks on Types of Roads")

# Plot of Average Vehicles and Trucks on Different Sized Roads
def plot6(data):
    data.lanes = data.lanes.astype(int)
    data = data.groupby("lanes", as_index=False).mean()
    data.plot(x="lanes", y=["speed", "limit"], kind="line", legend=True,
              title="Average Vehicles and Trucks on Different Sized Roads")
    data.plot(x="lanes", y=["vehicles", "trucks"], kind="bar", legend=True,
              title="Average Vehicles and Trucks on Different Sized Roads")

# Plot of Average Speed of Vehicles and Trucks on Roads
def plot7(data):
    plt.subplots()
    color = ["red", "blue", "yellow", "green"]
    plt.scatter(data.speed, data.vehicles, c=color, marker='o')
    plt.scatter(data.speed, data.trucks, c=color, marker='o')
    plt.title("Average Speed of Vehicles and Trucks on Roads")
    plt.legend()
    plt.show()

# VAT Algorithm
def plotVAT(data):
    data.drop(["name", "type"], axis=1, inplace=True)
    data.lanes = data.lanes.astype(int)
    x = data.speed.tolist()
    y = data.width.tolist()
    sklearn_pca = sd.PCA(n_components=2)
    data_std = sp.StandardScaler().fit_transform(data)
    sklearn_pca.fit_transform(data_std)
    print("Variance explained by each PC: ",
        sklearn_pca.explained_variance_ratio_)
    colors = data.speed.tolist()
    plt.scatter(x, y, s=60, c=colors)
    plt.xlabel("First principle component")
    plt.ylabel("Second principle component")
    plt.show()
    sklearn_pca = sd.PCA()
    sklearn_pca.fit(data_std)
    plt.plot(range(len(data.columns)), sklearn_pca.explained_variance_ratio_)
    plt.xticks(range(len(data.columns)))
    plt.xlabel("Principle Component")
    plt.ylabel("Percentage of Variance Explained")
    plt.show()
    sns.heatmap(data_std, cmap="magma", xticklabels=False, yticklabels=False)
    plt.title("Heatmap")
    plt.show()
    np.random.shuffle(data_std)
    sq = ds.squareform(ds.pdist(data_std))
    ax = sns.heatmap(sq, cmap="magma", xticklabels=False, yticklabels=False)
    ax.set(xlabel="Objects", ylabel="Objects", title="Dissimilarity Matrix")
    plt.show()
    RV, R, I = vi.VAT(data_std)
    ax = sns.heatmap(RV, cmap="magma", xticklabels=False, yticklabels=False)
    ax.set(xlabel="Objects", ylabel="Objects", title="VAT Visualisation")
    plt.show()

# Basemap visualisation of shapefile
def plotMap(data):
    datamap = bm.Basemap(projection="merc", lat_0=-37.4713, lon_0=144.7852,
        resolution='l', epsg=4326, area_thresh=0.1, llcrnrlat=-39.188314,
        llcrnrlon=140.711512, urcrnrlat=-33.922496, urcrnrlon=150.052936)
    datamap.arcgisimage(service="ESRI_Imagery_World_2D",
        xpixels=1024, verbose=True)
    datamap.drawrivers(color="aqua")
    categories = np.unique(data.vehicles)
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    data["color"] = data.vehicles.apply(lambda x: colordict[x])
    x, y = datamap(data.longitude, data.latitude)
    datamap.scatter(x, y, c=data["color"], cmap="rainbow", marker='o')
    plt.colorbar(label="Traffic Density")
    plt.show()

# Determine clusters present in the data
def clusterData():
    data = openConcatenatedData()
    plotCluster(data)

# Pylab KMeans clustering on traffic lanes
def plotCluster(data):
    data.drop(["name", "type"], axis=1, inplace=True)
    pca = sd.PCA(n_components=2).fit(data)
    pca2d = pca.transform(data)
    for i in range(0, pca2d.shape[0]):
        if data.lanes[i] == 1:
            c1 = pl.scatter(pca2d[i, 0], pca2d[i, 1], c="blue", marker='^')
        elif data.lanes[i] == 2:
            c2 = pl.scatter(pca2d[i, 0], pca2d[i, 1], c="cyan", marker='o')
        elif data.lanes[i] == 3:
            c3 = pl.scatter(pca2d[i, 0], pca2d[i, 1], c="yellow", marker='s')
        elif data.lanes[i] == 4:
            c4 = pl.scatter(pca2d[i, 0], pca2d[i, 1], c="black", marker='*')
        else:
            c5 = pl.scatter(pca2d[i, 0], pca2d[i, 1], c="orange", marker='v')
    pl.title("Dataset with 5 clusters")
    pl.legend([c1, c2, c3, c4, c5], ["1 Lane", "2 Lanes", "3 Lanes",
        "4 Lanes", "5 Lanes"])
    pl.show()

# Produces a pivot table of the concatenated data
def pivotData():
    data = openConcatenatedData()
    data.type = data.type.astype("category")
    data.type.cat.set_categories(["Road", "Street", "Freeway", "Highway"],
        inplace=True)
    table = pd.pivot_table(data, index=["type", "lanes"], fill_value=0)
    table.to_csv("tables/pivot1.csv")

# Handles the main process of data wrangling
def main():
    processRawData()
    processProcessedData()
    concatenateData()
    visualiseData()
    clusterData()
    pivotData()

if __name__ == "__main__":
    main()
