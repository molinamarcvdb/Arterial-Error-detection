#Common imports
import os
import slicer
import vtk
import numpy as np
import JupyterNotebooksLib as slicernb
import skimage 
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from time import time
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import distance
import networkx as nx
import nibabel as nib
from nibabel.affines import apply_affine
import openpyxl
from openpyxl import workbook
import math
import argparse


# Load graph
G = nx.read_gpickle(os.path.join(caseDir, "graph_label.pickle"))
# Get cellId to segmentId dict
cellIdToVesselType = {}
for n0, n1 in G.edges:
    cellIdToVesselType[G[n0][n1]["CellID"]] = G[n0][n1]["edgetype"]
# {Dict CellID : Edgetype}
print()
print("We can asses the vessel type of each segment in SegementsArray following:")
print(cellIdToVesselType)
# Create a dictionary with EdgeTypes
edgeTypesDict = {
    0: "other",
    1: "AA",
    2: "BT",
    3: "RCCA",
    4: "LCCA",
    5: "RSA",
    6: "LSA",
    7: "RVA",
    8: "LVA",
    9: "RICA",
    10: "LICA",
    11: "RECA",
    12: "LECA",
    13: "BA",
    14: "AA+BT",
    15: "RVA+LVA"}

# Transform IJK to RAS coordinates
segmentsArray = np.load(os.path.join(caseDir, "segmentsArray.npy"), allow_pickle=True)
segmentsArrayAff = np.ndarray([len(segmentsArray)], dtype = object)
aff = np.asarray(nib.load(os.path.join(caseDir, caseId)).affine)
for idx in range(len(segmentsArray)):
    segmentsArrayAff[idx] = np.ndarray([len(segmentsArray[idx][0]), 3])
    for idx2 in range(len(segmentsArray[idx][0])):
        segmentsArrayAff[idx][idx2] = np.matmul(aff, np.append(segmentsArray[idx][0][idx2], 1.0))[:3]

        
##########################################
### Vertebral-Basilar Arthery Detection ###
##########################################      
        
# But still a big amount of the verterbal arthery origins remains being missdetected, so that we will develop a simple algorithm 
# which detects if the L/RSA & L/RVA segments are in contact. To do this we'll need to get the closest points of both segments. 
BA =  []  # Type 13
LVA =  []  # Type 8
RVA =  []   # Type 7

for idx in range(len(cellIdToVesselType)):
    if cellIdToVesselType[idx] == 13: #BA
        BA.append(np.asanyarray(segmentsArrayAff[idx]))
    if cellIdToVesselType[idx] == 7: #RVA
        RVA.append(segmentsArrayAff[idx])
    if cellIdToVesselType[idx] == 8: #LVA
        LVA.append(segmentsArrayAff[idx])
    
# flatten - lists of lists
flat_RVA = [item for sublist in RVA for item in sublist]
flat_LVA = [item for sublist in LVA for item in sublist]
flat_BA = [item for sublist in BA for item in sublist]


# Obtain the points of the L/RVA segments with the maximum S coordinate (RAS).

if len(flat_LVA) > 0:
    LVA_height = []
    for idx in range(len(flat_LVA)): 
        LVA_height.append(flat_LVA[idx][2])
    Max_LVA_idx = np.argmax(LVA_height)
    Max_LVA_Coord = flat_LVA[Max_LVA_idx]
    # Get the distance form all points of the L/RSA to the obtained coordinate of each vertebral arthery segment.
    #Get those (2) endpoints which are closer to the Min_L/RVA_Coord
    if len(flat_BA) > 0:
        BA_height = []
        for idx in range(len(flat_BA)): 
            BA_height.append(flat_BA[idx][2])
        Min_BA_idx = np.argmin(BA_height)
        Min_BA_Coord = flat_BA[Min_BA_idx] 

        if distance.euclidean(Min_BA_Coord, Max_LVA_Coord) > 0.5:
            COORD1.append(Min_BA_Coord)
            COORD2.append(Max_LVA_Coord)
            # Add L/RVA centerpoint to the centerpoints array
            x = (Min_BA_Coord[0]+Max_LVA_Coord[0])/2
            y = (Min_BA_Coord[1]+Max_LVA_Coord[1])/2
            z = (Min_BA_Coord[2]+Max_LVA_Coord[2])/2
            Centerpoint_LVA = np.array([x,y,z])
            if len(Centerpoints) > 0:
                Centerpoints = np.append(Centerpoints, [Centerpoint_LVA], axis = 0)
            else:
                Centerpoints = np.array([Centerpoint_LVA])
            print()
            print(f"There's a dicontinuity in the high part LVA:")
            print(f"{[Min_BA_Coord, Max_LVA_Coord]}, the associated centerpoint has been appended to Centerpoints Array")
    else:
        COORD1.append(Max_LVA_Coord)
        COORD2.append(Max_LVA_Coord)
        if len(Centerpoints) > 0:
                Centerpoints = np.append(Centerpoints, [Max_LVA_Coord], axis = 0)
        else:
            Centerpoints = np.array([Max_LVA_Coord])
        print()
        print(f"There's a dicontinuity in the high part LVA: {Max_LVA_Coord}")


if len(flat_RVA) > 0:
    RVA_height = []
    for idx in range(len(flat_RVA)): 
        RVA_height.append(flat_RVA[idx][2])
    Max_RVA_idx = np.argmax(LVA_height)
    Max_RVA_Coord = flat_RVA[Max_RVA_idx]
    # Get the distance form all points of the L/RSA to the obtained coordinate of each vertebral arthery segment.
    #Get those (2) endpoints which are closer to the Min_L/RVA_Coord
    if len(flat_BA) > 0:
        BA_height = []
        for idx in range(len(flat_BA)): 
            BA_height.append(flat_BA[idx][2])
        Min_BA_idx = np.argmin(BA_height)
        Min_BA_Coord = flat_BA[Min_BA_idx] 

        if distance.euclidean(Min_BA_Coord, Max_RVA_Coord) > 0.5:
            COORD1.append(Min_BA_Coord)
            COORD2.append(Max_RVA_Coord)
            # Add L/RVA centerpoint to the centerpoints array
            x = (Min_BA_Coord[0]+Max_RVA_Coord[0])/2
            y = (Min_BA_Coord[1]+Max_RVA_Coord[1])/2
            z = (Min_BA_Coord[2]+Max_RVA_Coord[2])/2
            Centerpoint_RVA = np.array([x,y,z])
            if len(Centerpoints) > 0:
                Centerpoints = np.append(Centerpoints, [Centerpoint_RVA], axis = 0)
            else:
                Centerpoints = np.array([Centerpoint_RVA])
            print()
            print(f"There's a dicontinuity in the high part RVA:")
            print(f"{[Min_BA_Coord, Max_RVA_Coord]}, the associated centerpoint has been appended to Centerpoints Array")
    else:
            COORD1.append(Max_RVA_Coord)
            COORD2.append(Max_RVA_Coord)
            if len(Centerpoints) > 0:
                    Centerpoints = np.append(Centerpoints, [Max_RVA_Coord], axis = 0)
            else:
                Centerpoints = np.array([Max_RVA_Coord])
            print()
            print(f"There's a dicontinuity in the high part LVA: {Max_RVA_Coord}")