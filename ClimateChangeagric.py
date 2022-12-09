# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:22:46 2022

@author: matah
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Importing Agricultural land (% of land area) from Excel Sheet
agrictolandarea = pd.read_excel("AgricLandToLAndArea.xls", skiprows=3,
                                usecols=[0, 44, 49, 54, 59, 63])
print(agrictolandarea)

#extracting West African States data
agric_west = agrictolandarea.loc[[18, 19, 41, 47, 80, 83, 85, 87, 131, 158,
                                  166, 173, 174, 207, 210, 232], :]

# rounding off numbers to 2 significant figures
pd.options.display.float_format = '{:,.2f}'.format
print('\n Agricultural land (% of land area): \n', agric_west)

# Transposing the data frame
agric_west.set_index('Country Name', inplace=True)
df1 = agric_west.transpose()


def plot_A(data):
    
    """Plots a grouped bar chart for Agricultural land (% of land Area) 
    for West African States.

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
    plt.title('Agricultural land (% of land area)', fontweight='bold')
    plt.xticks([r + width*2 for r in range(data_length)],
               list(data.index.values), rotation=90)
    plt.savefig("AgricLand.jpg", bbox_inches = 'tight', dpi = 140)
    plt.show()


plot_A(agric_west)


#Importing Agriculture, forestry, and fishing, 
#value added (% of GDP) data from Excel Sheet

agricvalue = pd.read_excel("AgricValueAddedToGDP.xls", skiprows=3,
                           usecols=[0, 44, 49, 54, 59, 63])
print(agricvalue)

#extracting West African States data
agricvalue_west = agricvalue.loc[[18, 19, 41, 47, 80, 83, 85, 87, 131, 158,
                                  166, 173,174, 207, 210, 232], :]

# rounding off numbers to 2 significant figures
pd.options.display.float_format = '{:,.2f}'.format
print('\n Agriculture, forestry, and fishing, value added (% of GDP): \n', 
      agric_west)

#Transposing the data frame
agricvalue_west.set_index('Country Name', inplace=True)
df2 = agricvalue_west.transpose()


def plot_B(data):
    """Plots a grouped bar chart for Agriculture, forestry, and fishing, 
    value added (% of GDP) data for West African States.

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
    plt.title('Agriculture, forestry, and fishing, value added (% of GDP)',
              fontweight='bold')
    plt.xticks([r + width*2 for r in range(data_length)],
               list(data.index.values), rotation=90)
    plt.savefig("AgricValue.jpg", bbox_inches = 'tight', dpi = 140)
    plt.show()


plot_B(agricvalue_west)


#importing CO2 emissions (kt) data from Excel Sheet
emission = pd.read_excel("CO2emission.xls", skiprows=3,
                         usecols=[0, 44, 49, 54, 59, 63])
emission_west = emission.loc[[18, 19, 41, 47, 80, 83, 85, 87, 131, 158, 166, 
                              173, 174, 207, 210, 232], :]
print('\n CO2 emissions (kt):\n' ,emission_west)
# Transposing the data frame
emission_west.set_index('Country Name', inplace=True)
df3 = emission_west.transpose()

# plotting 
def plot_C(data):
    """Plots a line chart for CO2 emissions (kt) data 
    for West African States.

    Args:
        data: A dataframe containing the data to plot.
    """
    plt.figure(figsize=(6, 4))
    for header_name in data.columns:
        plt.plot(data[header_name], label=header_name)
    plt.xlabel('Years', fontweight='bold')
    plt.ylabel('CO2 emissions (kt)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Country')
    plt.title(('CO2 emissions (kt)'), fontweight='bold')
    plt.savefig("emiss.jpg", bbox_inches='tight', dpi=140)
    plt.show()


plot_C(df3)

#Importing Urban population (% of total population) data from Excel Sheet
urban = pd.read_excel("UrbanPopulationToTotal.xls",
                            skiprows=3, usecols=[0, 44, 49, 54, 59, 63])
urban_west = urban.loc[[18, 19, 41, 47, 80, 83, 85, 87, 131, 158, 166, 173,
                                 174, 207, 210, 232], :]
print('\n Urban population (% of total population): \n', urban_west)
# Transposing the data frame
urban_west.set_index('Country Name', inplace=True)
df4 = urban_west.transpose()

# plotting 
def plot_D(data):
    
    """Plots a line chart for Urban population (% of total population) data 
    for West African States.

    Args:
        data: A dataframe containing the data to plot.
    """
    plt.figure(figsize=(6, 4))
    for header_name in data.columns:
        plt.plot(data[header_name], label=header_name)
    plt.xlabel('Years', fontweight='bold')
    plt.ylabel('%', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Country')
    plt.title(('Urban population (% of total population)'),
              fontweight='bold')
    plt.savefig("urb.jpg", bbox_inches='tight', dpi=140)
    plt.show()


plot_D(df4)


# Importing ForestAreaToLand Data from Excel Sheet
forestarea = pd.read_excel("ForestAreaToLAndArea.xls",
                           skiprows=3, usecols=[0, 44, 49, 54, 59, 63])
forest_west = forestarea.loc[[18, 19, 41, 47, 80, 83, 85, 87, 131, 158, 
                              166, 173, 174, 207, 210, 232], :]
print('\n Forest area (% of land area): \n' ,forest_west)
# Transposing the data frame
forest_west.set_index('Country Name', inplace=True)
df5 = forest_west.transpose()

# plotting
def plot_E(data):
    """Plots a line chart for Forest area (% of land area) data 
    for West African States.

    Args:
        data: A dataframe containing the data to plot.
    """
    plt.figure(figsize=(6, 4))
    for header_name in data.columns:
        plt.plot(data[header_name], label=header_name)
    plt.xlabel('Years', fontweight='bold' )
    plt.ylabel('%', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), title='Country')
    plt.title(('Forest area (% of land area)'), fontweight='bold')
    plt.savefig("FOREST.jpg", bbox_inches='tight', dpi=140)
    plt.show()


plot_E(df5)


#Importing Population growth (annual %) data from Excel Sheet
popgrowth = pd.read_excel("AgricLandToLAndArea.xls", skiprows=3,
                                usecols=[0, 44, 49, 54, 59, 63])
print(agrictolandarea)

# extracting West African States
pop_west = popgrowth.loc[[18, 19, 41, 47, 80, 83, 85, 87, 131, 158, 166, 173,
                                 174, 207, 210, 232], :]

# rounding off numbers to 2 significant figures
pd.options.display.float_format = '{:,.2f}'.format
print('\n Population growth (annual %): \n', agric_west)

# Transposing the data frame
pop_west.set_index('Country Name', inplace=True)
df6 = agric_west.transpose()


def plot_F(data):
    
    """Plots a grouped bar chart forPopulation growth (annual %):) data
    for West African States.

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
    plt.title('Population growth (annual %))', fontweight='bold')
    plt.xticks([r + width*2 for r in range(data_length)],
               list(data.index.values), rotation=90)
    plt.savefig("Population growth.jpg", bbox_inches = 'tight', dpi = 140)
    plt.show()


plot_F(pop_west)

#creating a dataframe of Ghana for correlation
Ghana = {'Agric. Land(% of Total Land': df1['Ghana'], 
                 'Agric.(% of GDP)': df2['Ghana'], 
                 'CO2 emissions (kt)': df3['Ghana'],
                 'Urban population (% of total population)': df4['Ghana'],
                 'Forest area (% of land area)':df5['Ghana'], 
                 'Population growth (annual %)': df6['Ghana']}

#Correlation Analysis for Ghana        
df_Ghana = pd.DataFrame(Ghana)
print(df_Ghana)
corr_ghana = df_Ghana.corr()
print(corr_ghana)

#heatmap for Ghana
def Plot_Ghana(data, title):
    """Plots a correlation heatmap for Ghana.

    This function calculates the correlation between the columns in the
    given data and plots a heatmap of the correlations.

    Args:
        data: A dataframe containing the data to plot.
        title: The title to use for the plot.
    """
    # Calculate the correlation matrix.
    corr = corr_ghana

    # Create a figure and set the figure size.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap from the correlation matrix.
    sns.heatmap(corr, xticklabels=corr.columns, 
                yticklabels=corr.columns, ax=ax, annot=True)
    ax.set_title(title, fontsize=18)
    plt.savefig(title + '.png', dpi=500, bbox_inches='tight')
    plt.show()

Plot_Ghana(corr_ghana, 'Ghana')

#creating a dataframe of Sierra Leone for correlation
SL = {'Agric. Land(% of Total Land': df1['Sierra Leone'], 
                 'Agric.(% of GDP)': df2['Sierra Leone'], 
                 'CO2 emissions (kt)': df3['Sierra Leone'],
                 'Urban population (% of total population)': df4['Sierra Leone'],
                 'Forest area (% of land area)':df5['Sierra Leone'], 
                 'Population growth (annual %)': df6['Sierra Leone']}

#Correlation Analysis for Sierra Leone        
df_SL = pd.DataFrame(SL)
print(df_SL)
corr_SL = df_SL.corr()
print(corr_SL)

#heatmap for Sierra Leone
def Plot_SL(data, title):
    """Plots a correlation heatmap for Sierra Leone.

    This function calculates the correlation between the columns in the
    given data and plots a heatmap of the correlations.

    Args:
        data: A dataframe containing the data to plot.
        title: The title to use for the plot.
    """
    # Calculate the correlation matrix.
    corr = corr_SL

    # Create a figure and set the figure size.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap from the correlation matrix.
    sns.heatmap(corr, xticklabels=corr.columns, 
                yticklabels=corr.columns, ax=ax, annot=True)
    ax.set_title(title, fontsize=18)
    plt.savefig(title + '.png', dpi=500, bbox_inches='tight')
    plt.show()

Plot_SL(corr_SL, 'SIERRA LEONE')

#creating a dataframe of Gabon for correlation
Gb = {'Agric. Land(% of Total Land': df1['Gabon'], 
                 'Agric.(% of GDP)': df2['Gabon'], 
                 'CO2 emissions (kt)': df3['Gabon'],
                 'Urban population (% of total population)': df4['Gabon'],
                 'Forest area (% of land area)':df5['Gabon'], 
                 'Population growth (annual %)': df6['Gabon']}

#Correlation Analysis for Gabon        
df_Gb = pd.DataFrame(Gb)
print(df_Gb)
corr_Gb = df_Gb.corr()
print(corr_Gb)

#heatmap for Sierra Leone
def Plot_Gb(data, title):
    """Plots a correlation heatmap for Gabon.

    This function calculates the correlation between the columns in the
    given data and plots a heatmap of the correlations.

    Args:
        data: A dataframe containing the data to plot.
        title: The title to use for the plot.
    """
    # Calculate the correlation matrix.
    corr = corr_Gb

    # Create a figure and set the figure size.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap from the correlation matrix.
    sns.heatmap(corr, xticklabels=corr.columns, 
                yticklabels=corr.columns, ax=ax, annot=True)
    ax.set_title(title, fontsize=18)
    plt.savefig(title + '.png', dpi=500, bbox_inches='tight')
    plt.show()

Plot_Gb(corr_Gb, 'GABON')

#creating a dataframe of Nigeria for correlation
Ngr = {'Agric. Land(% of Total Land': df1['Nigeria'], 
                 'Agric.(% of GDP)': df2['Nigeria'], 
                 'CO2 emissions (kt)': df3['Nigeria'],
                 'Urban population (% of total population)': df4['Nigeria'],
                 'Forest area (% of land area)':df5['Nigeria'], 
                 'Population growth (annual %)': df6['Nigeria']}

#Correlation Analysis for Nigeria        
df_Ngr = pd.DataFrame(Ngr)
print(df_Ngr)
corr_Ngr = df_Ngr.corr()
print(corr_Ngr)

#heatmap for Sierra Leone
def Plot_Ngr(data, title):
    """Plots a correlation heatmap for Nigeria.

    This function calculates the correlation between the columns in the
    given data and plots a heatmap of the correlations.

    Args:
        data: A dataframe containing the data to plot.
        title: The title to use for the plot.
    """
    # Calculate the correlation matrix.
    corr = corr_Ngr

    # Create a figure and set the figure size.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap from the correlation matrix.
    sns.heatmap(corr, xticklabels=corr.columns, 
                yticklabels=corr.columns, ax=ax, annot=True)
    ax.set_title(title, fontsize=18)
    plt.savefig(title + '.png', dpi=500, bbox_inches='tight')
    plt.show()

Plot_Ngr(corr_Ngr, 'NIGERIA')

#creating a dataframe of Liberia for correlation
Lie = {'Agric. Land(% of Total Land': df1['Liberia'], 
                 'Agric.(% of GDP)': df2['Liberia'], 
                 'CO2 emissions (kt)': df3['Liberia'],
                 'Urban population (% of total population)': df4['Liberia'],
                 'Forest area (% of land area)':df5['Liberia'], 
                 'Population growth (annual %)': df6['Liberia']}

#Correlation Analysis for Liberia        
df_Lie = pd.DataFrame(Lie)
print(df_Lie)
corr_Lie = df_Lie.corr()
print(corr_Lie)

#heatmap for Liberia
def Plot_Lie(data, title):
    """Plots a correlation heatmap for Liberia.

    This function calculates the correlation between the columns in the
    given data and plots a heatmap of the correlations.

    Args:
        data: A dataframe containing the data to plot.
        title: The title to use for the plot.
    """
    # Calculate the correlation matrix.
    corr = corr_Lie

    # Create a figure and set the figure size.
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap from the correlation matrix.
    sns.heatmap(corr, xticklabels=corr.columns, 
                yticklabels=corr.columns, ax=ax, annot=True)
    ax.set_title(title, fontsize=18)
    plt.savefig(title + '.png', dpi=500, bbox_inches='tight')
    plt.show()

Plot_Lie(corr_Lie, 'LIBERIA')