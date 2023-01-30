# -*- coding: utf-8 -*-
"""
The code was used to analyse how the agriculture indicators of climate affect 
West Africa.
Created on Mon Dec  5 14:22:46 2022

@author: Omoregbe Olotu
"""


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr


#files for analysis
file1 = "AgricLandToLAndArea.xls"
file2 = "AgricValueAddedToGDP.xls"
file3 = "CO2emission.xls"
file4 = "TotalPopulation.xls"
file5 = "ForestAreaToLAndArea.xls"
file6 = "PopGrowth.xls"


def read(filename):
    """ Reads an Excel file extracting all West African
    and returns the original dataframe and its transposed version.
    
    Parameters:
        filename: The filename of the Excel file to be read.
        
    Returns:
        [DataFrame, DataFrame Transposed]: The original dataframe
        and its transposed version."""
        
    #Importing Agricultural land (% of land area) from Excel Sheet
    data = pd.read_excel(filename, skiprows=3,
                                    usecols=[0, 44, 49, 54, 59, 63])
    
    #extracting the west african states data
    data = data.loc[[18, 19, 41, 47, 83, 85, 86, 87, 131, 158,
                                      166, 173, 174, 207, 210, 232], :]
    #resettin index
    data.set_index('Country Name', inplace=True)
    
    return data, data.transpose()

#Defining the function that plots bar chart for Agric. Land(% of Land Area)
def plot_A(data, title):
    
    """Plots a grouped bar chart for West African States.

    Args:
        data: A dataframe containing the data to plot.
    """
  
    plt.figure(figsize=(6, 3))
    width = 0.07
    counter = 0
    data_length = len(data.index)

    for header_name in data.columns:
        mySeries = data[header_name].squeeze()

        # positioning the Bar elements
        if counter == 0:
            position = np.arange(len(data.index))
        else:
            position = [x + width for x in position]

        plt.bar(position, mySeries, width=width, label=header_name,)
        counter += 1
    plt.xlabel('Country', fontweight='bold')
    plt.ylabel('%', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), title='Years')
    plt.title(title, fontweight='bold')
    plt.xticks([r + width*2 for r in range(data_length)],
               list(data.index.values), rotation=90)
    plt.show()

    
def plot_B(data, title):
    """Plots a line chart for West African States.

    Args:
        data: A dataframe containing the data to plot.
    """
    plt.figure(figsize=(6, 4))
    for header_name in data.columns:
        plt.plot(data[header_name], label=header_name)
    plt.xlabel('Years', fontweight='bold')
    plt.ylabel('CO2 emissions (kt)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Country')
    plt.title(title, fontweight='bold')  
    plt.show()

#Defining a function to plot a correlation map 
def Plot_C(data, title):
    """Plots a correlation heatmap for selected West African States.

    This function calculates the correlation between the columns in the
    given data and plots a heatmap of the correlations.

    Args:
        data: A dataframe containing the data to plot.
        title: The title to use for the plot.
    """
    # Correlation.
    corr = data.corr(method='pearson')

    # Create a figure and set the figure size.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap from the correlation matrix.
    sns.heatmap(corr, xticklabels=corr.columns, 
                yticklabels=corr.columns, ax=ax, annot=True)
    ax.set_title(title, fontsize=18)
    plt.savefig(title + '.png', dpi=500, bbox_inches='tight')
    plt.show()
    

#Original and Transposed format for Agriculture Land to Area
agric_west, agric_westTs = read(file1)
df1 = agric_westTs

#Original and Transposed format for Agriculture Value Added To GDP
agricvalue, agricvalue_westTs = read(file2)
df2 = agricvalue_westTs

#Original and Transposed format for CO2emission
emission_west, emission_westTs = read(file3)
df3 = emission_westTs

#Original and Transposed format for Total Population
pop_west, pop_westTs = read(file4)
df4 = pop_westTs

#Original and Transposed format for Forest Area To Land
forest_west, forest_westTs = read(file5)
df5 = forest_westTs

#Original and Transposed format for Population Growth
growth_west, growth_westTs = read(file6)
df6 = growth_westTs

#graphplots for the various indicators used for analysis
plot_A(agric_west,' Agricultural land (% of land area)')
plot_A(agricvalue, 'Agriculture, forestry, and fishing,value added (% of GDP)')
plot_B(df3, 'Agriculture, forestry, and fishing, value added (% of GDP)')
plot_A(df4, 'Total population)')
plot_B(df5, 'Forest area (% of land area)')
plot_A(pop_west, 'Population growth (annual %))' )


#creating dataset for Sierra Leone correlation analysis
SL = {'Agric. Land(% of Total Land': df1['Sierra Leone'], 
      'Agric.(% of GDP)': df2['Sierra Leone'], 
      'CO2 emissions (kt)': df3['Sierra Leone'],
      'Total Population':df4['Sierra Leone'],
      'Forest area (% of land area)':df5['Sierra Leone'], 
      'Population growth (annual %)': df6['Sierra Leone']}

#Converting to dataframe        
df_SL = pd.DataFrame(SL)
print(df_SL)


#plots heatmap for Sierra-Leone
Plot_C(df_SL, 'SIERRA LEONE')

# P-Value test for the validity between the correlation Agric Value to GDP
# and Agric Land (% of Land) for Sierra-Leone.
  
x = df1['Sierra Leone']
y = df2['Sierra Leone']
 
# The P-Value using Scipy to 2d.p   
p = round(pearsonr(x, y)[1], 2)
print('\n P-Value: \n', p)

#creating dataset for Nigeria's correlation analysis
Ngr = {'Agric. Land(% of Total Land': df1['Nigeria'], 
                 'Agric.(% of GDP)': df2['Nigeria'],
                 'CO2 emissions (kt)': df3['Nigeria'],
                 'Total Population': df4['Nigeria'],
                 'Forest area (% of land area)':df5['Nigeria'], 
                 'Population growth (annual %)': df6['Nigeria']}

#Converting to Dataframe       
df_Ngr = pd.DataFrame(Ngr)
print(df_Ngr)

#plots heatmap for Nigeria
Plot_C(df_Ngr, 'NIGERIA')


#creating dataset for Liberia's correlation analysis
Lie = {'Agric. Land(% of Total Land': df1['Liberia'], 
                 'Agric.(% of GDP)': df2['Liberia'], 
                 'CO2 emissions (kt)': df3['Liberia'],
                 'Total Population': df4['Liberia'],
                 'Forest area (% of land area)':df5['Liberia'], 
                 'Population growth (annual %)': df6['Liberia']}

#Converting to Dataframe        
df_Lie = pd.DataFrame(Lie)
print(df_Lie)


#plots the heatmap graph
Plot_C(df_Lie, 'LIBERIA')
