# Design-Analysis-Materials
Design analysis materials provided for the manuscript
[README.md](https://github.com/user-attachments/files/25231199/README.md)
# Data and Code Availability

## Overview
The underlying data and code for the relevant figures and tables are provided in the supplementary materials.

---

## 1 Dataset file name:
factor_analysis_data.npy

### Description:
Original user rating data collected for EFA, structured according to the Likert-scale.

### Underlying data for:
-Figure 1. Correlation heatmap of service need variables
-Figure 2. Principal component analysis results
-Table 3. Factor analysis questionnaire rating sheet
-Table 4. Results of the KMO test and Bartlett's test of sphericity
-Table 5. Factor loading matrix


## 2 Dataset file name:
kano_analysis_data.npy

### Description:
Original user evaluation data collected for the KANO analysis, from which the SI and DSI values were derived.

### Underlying data for:
-Figure 3. Vector diagram of service demand hierarchy classification
-Table 6. Classification of community elderly care service need hierarchy

## 3 Dataset file name:
preexisting_system_sus.csv
proposed_system_sus.csv

### Description:
Original user evaluation data collected through the System Usability Scale (SUS) questionnaire for the pre-existing/proposed system.

### Underlying data for:
SUS score calculation and usability evaluation of the pre-existing/proposed system.



## 1 Python code file name:

plot_EFA_bokeh.py

### Description:
Underlying code used to perform the EFA and generate the figures tables.

### Underlying code for:
-Figure 1. Correlation heatmap of service need variables
-Figure 2. Principal component analysis results
-Table 4. Results of the KMO test and Bartlett's test of sphericity
-Table 5. Factor loading matrix



## 2 Python code file name:
TOPSIS_rel_cal.py

### Description:
The demand priorities were analyzed using the EWM-TOPSIS combined model.

### Underlying code for:
-Figure 4. TOPSIS relative closeness scores


## 3 Python code file name:
sus_analysis.py

### Description:
Underlying code used to process the SUS questionnaire data, calculate SUS scores, and perform statistical analysis of system usability.

### Underlying code for:
SUS score calculation for the pre-existing system and the proposed system.
Comparative statistical analysis of usability between the two systems.


