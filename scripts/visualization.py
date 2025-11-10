#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:24:15 2025

@author: avitia
"""


import matplotlib.pyplot as plt
from venny4py.venny4py import *



def venn_diagram(orthology_dict, figure_name='suppOrth_Ven'): #Plot function
    venny4py(sets=orthology_dict)
    plt.savefig(figure_name + ".svg", bbox_inches="tight")
    