# -*- coding: utf-8 -*-

"""
rudarjenje_2.py 
"""

import time
import csv

import matplotlib.pyplot as plt
import pylab

def risanje_1(csvfilename):

    t0 = time.time()
    print "risanje_1"
    
    csvfile = open(csvfilename, "rb")
    csv_rows = csv.reader(csvfile, delimiter=';', quotechar='"')
    
    podkategorije = {}
    cas = []
    head = True    
    for podkategorija, leto, mesec, classify_count, sum_p, avg_p_str, sum_m, avg_m in csv_rows:
        if head:
            head = False
            continue        
        if podkategorija not in podkategorije: 
            podkategorije[podkategorija] = []
        avg_p = float(str.replace(avg_p_str, ',', '.'))
        podkategorije[podkategorija].append(avg_p)
        letomesec = "%s-%s" %(leto, mesec)
        if letomesec not in cas:
            cas.append(letomesec)
    
    csvfile.close()
    
    X =  range(1, len(cas) + 1)
    pylab.xticks(X, cas)
    
    for podkategorija in podkategorije:
        pylab.plot(X, podkategorije[podkategorija], label=podkategorija)
          
    plt.legend()

    print "  čas = %.3f s" % (time.time() - t0)

    pylab.show()


# ===========================================================
# main
# ===========================================================    

if __name__ == "__main__":
    
    #t0_total = time.time()
    
    # ----------------------------------------
    # parametri
    # ----------------------------------------
    
    data_path  = "data\\"   
    sql_path = data_path + "sql\\" 
    
    # ----------------------------------------
    # risanje
    # ----------------------------------------

    csv_file = sql_path + "rudarjenje_2_2c.csv"
    risanje_1(csv_file)
    
    
    # ----------------------------------------

    #print "skupni čas = %.3f s" % (time.time() - t0_total)
    print "OK"
