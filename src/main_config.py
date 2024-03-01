from sen12ms_dataLoader import Seasons

class RunMode(enumerate):
    COMPLEX = 0
    SIMPLE = 1

seasons = [Seasons.SPRING, Seasons.SUMMER, Seasons.FALL, Seasons.WINTER]
classes = {
    1   : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Needleleaf Forests
    2   : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Broadleaf Forests
    3   : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Needleleaf Forests
    4   : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Broadleaf Forests
    5   : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],  # Mixed Forests
    6   : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],  # Closed (Dense) Shrubland
    7   : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],  # Open (Sparse) Shrubland
    8   : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],  # Woody Savannas
    9   : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],  # Savannas
    10  : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],  # Grasslands
    11  : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],  # Permanent Wetlands
    12  : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],  # Croplands
    13  : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],  # Urban and Built-Up Lands
    14  : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],  # Croplands/Natural Vegetation Mosaics
    15  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],  # Permanent Snow and Ice
    16  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # Barren
    17  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]   # Water Bodies
}
colors = {
    1   : [230,25,75],      # Evergreen Needleleaf Forests
    2   : [60,180,75],      # Evergreen Broadleaf Forests
    3   : [255,225,25],     # Deciduous Needleleaf Forests
    4   : [0,130,200],      # Deciduous Broadleaf Forests
    5   : [235,130,48],     # Mixed Forests
    6   : [145,30,180],     # Closed (Dense) Shrubland
    7   : [70,240,240],     # Open (Sparse) Shrubland
    8   : [240,50,230],     # Woody Savannas
    9   : [210,245,60],     # Savannas
    10  : [250,190,212],    # Grasslands
    11  : [0,128,128],      # Permanent Wetlands
    12  : [220,190,255],    # Croplands
    13  : [170,110,40],     # Urban and Built-Up Lands
    14  : [255,250,200],    # Croplands/Natural Vegetation Mosaics
    15  : [120,0,0],        # Permanent Snow and Ice
    16  : [170,255,195],    # Barren
    17  : [128,128,0],      # Water Bodies
}
names = {
    1   : 'Evergreen Needleleaf Forests',
    2   : 'Evergreen Broadleaf Forests',
    3   : 'Deciduous Needleleaf Forests',
    4   : 'Deciduous Broadleaf Forests',
    5   : 'Mixed Forests',
    6   : 'Closed (Dense) Shrubland',
    7   : 'Open (Sparse) Shrubland',
    8   : 'Woody Savannas',
    9   : 'Savannas',
    10  : 'Grasslands',
    11  : 'Permanent Wetlands',
    12  : 'Croplands',
    13  : 'Urban and Built-Up Lands',
    14  : 'Croplands/Natural Vegetation Mosaics',
    15  : 'Permanent Snow and Ice',
    16  : 'Barren',
    17  : 'Water Bodies'
}

simple_classes_conversion = {
    1   : [1,0,0,0,0,0,0,0,0,0],  # Forest
    2   : [1,0,0,0,0,0,0,0,0,0],  # Forest
    3   : [1,0,0,0,0,0,0,0,0,0],  # Forest
    4   : [1,0,0,0,0,0,0,0,0,0],  # Forest
    5   : [1,0,0,0,0,0,0,0,0,0],  # Forest
    6   : [0,1,0,0,0,0,0,0,0,0],  # Shrubland
    7   : [0,1,0,0,0,0,0,0,0,0],  # Shrubland
    8   : [0,0,1,0,0,0,0,0,0,0],  # Savanna
    9   : [0,0,1,0,0,0,0,0,0,0],  # Savanna
    10  : [0,0,0,1,0,0,0,0,0,0],  # Grassland
    11  : [0,0,0,0,1,0,0,0,0,0],  # Wetlands
    12  : [0,0,0,0,0,1,0,0,0,0],  # Croplands
    13  : [0,0,0,0,0,0,1,0,0,0],  # Urban and Built-Up
    14  : [0,0,0,0,0,1,0,0,0,0],  # Croplands
    15  : [0,0,0,0,0,0,0,1,0,0],  # Snow and Ice
    16  : [0,0,0,0,0,0,0,0,1,0],  # Barren
    17  : [0,0,0,0,0,0,0,0,0,1]   # Water
}
simple_classes = {
    1   : [1,0,0,0,0,0,0,0,0,0],  # Forest
    2   : [0,1,0,0,0,0,0,0,0,0],  # Shrubland
    3   : [0,0,1,0,0,0,0,0,0,0],  # Savanna
    4   : [0,0,0,1,0,0,0,0,0,0],  # Grassland
    5   : [0,0,0,0,1,0,0,0,0,0],  # Wetlands
    6   : [0,0,0,0,0,1,0,0,0,0],  # Croplands
    7   : [0,0,0,0,0,0,1,0,0,0],  # Urban and Built-Up
    8   : [0,0,0,0,0,0,0,1,0,0],  # Snow and Ice
    9   : [0,0,0,0,0,0,0,0,1,0],  # Barren
    10  : [0,0,0,0,0,0,0,0,0,1]   # Water    
}
simple_colors = {
    1   : [0,153,0],  # Forest
    2   : [198,176,68],  # Shrubland
    3   : [251,255,19],  # Savanna
    4   : [182,255,5],  # Grassland
    5   : [39,255,135],  # Wetlands
    6   : [194,95,68],  # Croplands
    7   : [165,165,165],  # Urban and Built-Up
    8   : [105,255,248],  # Snow and Ice
    9   : [249,255,164],  # Barren
    10  : [28,13,255]   # Water   
}
simple_names = {
    1   : 'Forest',
    2   : 'Shrubland',
    3   : 'Savanna',
    4   : 'Grassland',
    5   : 'Wetlands',
    6   : 'Croplands',
    7   : 'Urban and Built-Up',
    8   : 'Snow and Ice',
    9   : 'Barren',
    10  : 'Water'  
}

MODE = RunMode.SIMPLE
WORKERS = 32
MULTIPROCESSING=True
BATCH_SIZE = 4
EPOCHS = 100