# -*- coding: utf-8 -*-

"""
rudarjenje_1.py 
"""

import time
import os
import csv
import MySQLdb

# ===========================================================
# print log 
# ===========================================================

logs = []
def print_log(msg):
    global logs
    logs.append(msg)
    print msg

def check_create_dir(filepathname):
    (filepath, filename) = os.path.split(filepathname) 
    filepathdir = os.path.dirname(filepath + "\\")
    if not os.path.exists(filepathdir):
        os.makedirs(filepathdir)

def write_log(logfilename):
    check_create_dir(logfilename)
    logfile = open(logfilename, "wb+")
    for msg in logs:
        logfile.write(msg + "\n")
    logfile.close()

# ===========================================================
# load / save
# ===========================================================

def load_from_db(db, query):
    t0 = time.time()
    cursor = db.cursor()
    cursor.execute(query)
    
    rows = cursor.fetchall()
    
    print_log("  load from db count = %i, čas = %.3f s" % (len(rows), time.time() - t0))
    return rows

def load_from_csv(csvfilename):
    t0 = time.time()
    csvfile = open(csvfilename, "rb")
    csv_rows = csv.reader(csvfile, delimiter=';', quotechar='"')
    
    rows = []
    head = True;
    for row in csv_rows:
        # preskočimo glavo
        if head:
            head = False
            continue
        # vrstice
        rows.append(row)

    csvfile.close()
    print_log("  load from csv count = %i, čas = %.3f s" % (len(rows), time.time() - t0))
    return rows

def save_to_csv(rows, csvfilename, head): 
    t0 = time.time()
    csvfile = open(csvfilename, "wb+")
    csvwriter = csv.writer(csvfile, delimiter=';', quotechar='"')
    
    # glava
    csvwriter.writerow(head)
    # vrstice
    for row in rows:
        csvwriter.writerow(row)
        
    csvfile.close()
    print_log("  save to csv count = %i, čas = %.3f s" % (len(rows), time.time() - t0))

def save_to_tab(rows, csvfilename, head, head_c): 
    t0 = time.time()
    csvfile = open(csvfilename, "wb+")
    csvwriter = csv.writer(csvfile, delimiter='\t', quotechar='"')
    
    # glavi
    csvwriter.writerow(head)
    csvwriter.writerow(head_c)
    # vrstice
    for row in rows:
        csvwriter.writerow(row)
        
    csvfile.close()
    print_log("  save to tab count = %i, čas = %.3f s" % (len(rows), time.time() - t0))


# ===========================================================
# load db to csv(counts)
# ===========================================================

def load_from_db_to_csv_podkategorije_classify(db, csvfilename, subfilename):
    izbor = "podkategorije.classify"
    print_log(izbor)
    rows = load_from_db(db, "SELECT novica.podkategorija, klasifikacije.classify, COUNT(*) classify_count " + \
                            "FROM komentar, novica, klasifikacije " + \
                            "WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica " + \
                            "GROUP BY novica.podkategorija, klasifikacije.classify")
    save_to_csv(rows, csvfilename % (subfilename, izbor), ("podkategorija", "classify", "classify_count")) 

# -----------------------------------------------------------

def load_from_db_uporabniki_filter(db, idUporabniki_slabi): 
    print_log("uporabniki.filter")
    rows_filter = load_from_db(db, "SELECT komentar.Uporabnik_idUporabnik " + \
                                   "FROM komentar, klasifikacije " + \
                                   "WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND " + \
                                   "      komentar.Uporabnik_idUporabnik NOT IN (%s) " % idUporabniki_slabi + \
                                   "GROUP BY komentar.Uporabnik_idUporabnik " + \
                                   "HAVING COUNT(*) > 65")
    return rows_filter
    
def get_filter(rows, rows_filter):
    frows = []
    for row in rows:
        ok = False
        for row_filter in rows_filter:
            if row[0] == row_filter[0]: # 0=idUporabnik
                ok = True
                break
        if ok:                         
            frows.append(row)
    return frows

def load_from_db_to_csv_uporabniki_classify(db, csvfilename, subfilename, idUporabniki_slabi, rows_filter): 
    izbor = "uporabniki.classify"
    print_log(izbor)
    rows = load_from_db(db, "SELECT komentar.Uporabnik_idUporabnik, klasifikacije.classify, COUNT(*) classify_count " + \
                            "FROM komentar, klasifikacije " + \
                            "WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id " + \
                            "GROUP BY Uporabnik_idUporabnik, klasifikacije.classify")   
    rows = get_filter(rows, rows_filter)        
    save_to_csv(rows, csvfilename % (subfilename, izbor), ("idUporabnik", "classify", "classify_count")) 

def load_from_db_to_csv_uporabniki_plusi_minusi(db, csvfilename, subfilename, idUporabniki_slabi, rows_filter):
    izbor = "uporabniki.(plusi,minusi)" 
    print_log(izbor)
    rows = load_from_db(db, "SELECT komentar.Uporabnik_idUporabnik, sum(komentar.plusi) sum_plusi, sum(komentar.minusi) sum_minusi " + \
                            "FROM komentar, klasifikacije " + \
                            "WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id " + \
                            "GROUP BY komentar.Uporabnik_idUporabnik")  
    rows = get_filter(rows, rows_filter)        
    save_to_csv(rows, csvfilename % (subfilename, izbor), ("idUporabnik", "plusi", "minusi")) 

def load_from_db_to_csv_uporabniki_avg_plusi_minusi(db, csvfilename, subfilename, idUporabniki_slabi, rows_filter):
    izbor = "uporabniki.avg(plusi-minusi)"
    print_log(izbor)
    rows = load_from_db(db, "SELECT komentar.Uporabnik_idUporabnik, ROUND(AVG(komentar.plusi - komentar.minusi), 0) avg_razlika " + \
                            "FROM komentar, klasifikacije " + \
                            "WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id " + \
                            "GROUP BY komentar.Uporabnik_idUporabnik")  
    rows = get_filter(rows, rows_filter)        
    save_to_csv(rows, csvfilename % (subfilename, izbor), ("idUporabnik", "avg_razlika")) 

def load_from_db_to_csv_uporabniki_avg_ocena(db, csvfilename, subfilename, idUporabniki_slabi, rows_filter):
    izbor = "uporabniki.avg(ocena)"
    print_log(izbor)
    rows = load_from_db(db, "SELECT komentar.Uporabnik_idUporabnik, AVG(novica.ocena) avg_ocena " + \
                            "FROM komentar, klasifikacije, novica " + \
                            "WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica "  + \
                            "GROUP BY komentar.Uporabnik_idUporabnik")
    rows = get_filter(rows, rows_filter)        
    save_to_csv(rows, csvfilename % (subfilename, izbor), ("idUporabnik", "avg_ocena")) 

# -------------------------------------

def load_from_db_to_csv_counts(db, csvfilename, idUporabniki_slabi):
    subfilename = "counts"
    print_log(subfilename)
    
    load_from_db_to_csv_podkategorije_classify(db, csvfilename, subfilename)    

    rows_filter = load_from_db_uporabniki_filter(db, idUporabniki_slabi)
    
    load_from_db_to_csv_uporabniki_classify(        db, csvfilename, subfilename, idUporabniki_slabi, rows_filter) 
    load_from_db_to_csv_uporabniki_plusi_minusi(    db, csvfilename, subfilename, idUporabniki_slabi, rows_filter) 
    load_from_db_to_csv_uporabniki_avg_plusi_minusi(db, csvfilename, subfilename, idUporabniki_slabi, rows_filter)
    load_from_db_to_csv_uporabniki_avg_ocena(       db, csvfilename, subfilename, idUporabniki_slabi, rows_filter)


# ===========================================================
# save csv(counts) to csv(odstotek)
# ===========================================================

def save_csv_counts_to_csv_odstotek_podkategorije_classify(csvfilename, in_subfilename, out_subfilename):
    izbor = "podkategorije.classify"
    print_log(izbor)

    rows = load_from_csv(csvfilename % (in_subfilename, izbor))
    
    rows_count = {}
    for podkategorija, classify, count_str in rows:
        count = int(count_str)
        if podkategorija not in rows_count:
            rows_count[podkategorija] = count
        else:    
            rows_count[podkategorija] += count
            
    rows_odstotek = []   
    for podkategorija, classify, count_str in rows:
        count = int(count_str)
        rows_odstotek.append((podkategorija, classify, count, rows_count[podkategorija], float(count)/rows_count[podkategorija]))
        
    save_to_csv(rows_odstotek, csvfilename % (out_subfilename, izbor), ("podkategorija", "classify", "count", "count_all", "odstotek")) 

# -------------------------------------

def save_csv_counts_to_csv_odstotek(csvfilename):
    in_subfilename = "counts"
    out_subfilename = "odstotek"
    print_log(out_subfilename)
    
    save_csv_counts_to_csv_odstotek_podkategorije_classify(csvfilename, in_subfilename, out_subfilename)
    
# ===========================================================
# save csv(counts) to csv(normirano)
# ===========================================================

def save_csv_counts_to_csv_normirano_podkategorije_classify(csvfilename, in_subfilename, out_subfilename, classify_plus): 
    izbor = "podkategorije.classify"
    print_log(izbor)
    rows = load_from_csv(csvfilename % (in_subfilename, izbor))

    razlike = {}
    for podkategorija, classify, count_str in rows:
        count = int(count_str)
        predznak = 1 if classify == classify_plus else -1
        if podkategorija not in razlike:
            razlike[podkategorija] = predznak * count
        else:
            razlike[podkategorija] = predznak * (count - abs(razlike[podkategorija])) 

    max_razlika = 0
    for podkategorija in razlike:
        if max_razlika < abs(razlike[podkategorija]):
            max_razlika = abs(razlike[podkategorija])
    
            
    rows_normirano = []   
    for podkategorija, classify, count in rows:        
        rows_normirano.append((podkategorija, classify, count, razlike[podkategorija], max_razlika, float(razlike[podkategorija])/max_razlika))
        
    save_to_csv(rows_normirano, csvfilename % (out_subfilename, izbor),  ("podkategorija", "classify", "count", "razlika", "max_razlika", "normirano"))

def save_csv_counts_to_csv_normirano_uporabniki_classify(csvfilename, in_subfilename, out_subfilename, classify_plus): 
    izbor = "uporabniki.classify"
    print_log(izbor)
    rows = load_from_csv(csvfilename % (in_subfilename, izbor))

    razlike = {}
    for idUporabnik, classify, count_str in rows:
        count = int(count_str)
        predznak = 1 if classify == classify_plus else -1
        if idUporabnik not in razlike:
            razlike[idUporabnik] = predznak * count
        else:
            razlike[idUporabnik] = predznak * (count - abs(razlike[idUporabnik])) 

    max_razlika = 0
    for idUporabnik in razlike:
        if max_razlika < abs(razlike[idUporabnik]):
            max_razlika = abs(razlike[idUporabnik])
    
            
    rows_normirano = []   
    for idUporabnik, classify, count in rows:        
        rows_normirano.append((idUporabnik, classify, count, razlike[idUporabnik], max_razlika, float(razlike[idUporabnik])/max_razlika))
        
    save_to_csv(rows_normirano, csvfilename % (out_subfilename, izbor),  ("idUporabnik", "classify", "count", "razlika", "max_razlika", "normirano"))

def save_csv_counts_to_csv_normirano_uporabniki_plusi_minusi(csvfilename, in_subfilename, out_subfilename, classify_plus): 
    izbor = "uporabniki.(plusi,minusi)"
    print_log(izbor)
    rows = load_from_csv(csvfilename % (in_subfilename, izbor))

    razlike = {}
    for idUporabnik, plusi_str, minusi_str in rows:
        plusi = int(plusi_str)
        minusi = int(minusi_str)
        razlike[idUporabnik] = plusi - minusi

    max_razlika = 0
    for idUporabnik in razlike:
        if max_razlika < abs(razlike[idUporabnik]):
            max_razlika = abs(razlike[idUporabnik])
    
            
    rows_normirano = []   
    for idUporabnik, classify, count in rows:        
        rows_normirano.append((idUporabnik, count, razlike[idUporabnik], max_razlika, float(razlike[idUporabnik])/max_razlika))
        
    save_to_csv(rows_normirano, csvfilename % (out_subfilename, izbor), ("idUporabnik", "count", "razlika", "max_razlika", "normirano"))

def save_csv_counts_to_csv_normirano_uporabniki_avg_plusi_minusi(csvfilename, in_subfilename, out_subfilename, classify_plus): 
    izbor = "uporabniki.avg(plusi-minusi)"
    print_log(izbor)
    rows = load_from_csv(csvfilename % (in_subfilename, izbor))

    max_razlika = 0
    for idUporabnik, razlika_str in rows:
        razlika = int(razlika_str)
        if max_razlika < abs(razlika):
            max_razlika = abs(razlika)
            
    rows_normirano = []   
    for idUporabnik, razlika_str in rows:        
        razlika = int(razlika_str)
        rows_normirano.append((idUporabnik, razlika, max_razlika, float(razlika)/max_razlika))
    
    save_to_csv(rows_normirano, csvfilename % (out_subfilename, izbor), ("idUporabnik", "razlika", "max_razlika", "normirano"))

def save_csv_counts_to_csv_normirano_uporabniki_avg_ocena(csvfilename, in_subfilename, out_subfilename, classify_plus): 
    izbor = "uporabniki.avg(ocena)"
    print_log(izbor)
    rows = load_from_csv(csvfilename % (in_subfilename, izbor))

    max_ocena = 0
    for idUporabnik, ocena_str in rows:
        ocena = float(ocena_str)
        if max_ocena < abs(ocena):
            max_ocena = abs(ocena)
            
    rows_normirano = []   
    for idUporabnik, ocena_str in rows:        
        ocena = float(ocena_str)
        rows_normirano.append((idUporabnik, ocena, max_ocena, float(ocena)/max_ocena))
     
    save_to_csv(rows_normirano, csvfilename % (out_subfilename, izbor), ("idUporabnik", "ocena", "max_ocena", "normirano"))

# -------------------------------------

def save_csv_counts_to_csv_normirano(csvfilename, classify_plus):
    in_subfilename = "counts"
    out_subfilename = "normirano"
    print_log(out_subfilename)
    
    save_csv_counts_to_csv_normirano_podkategorije_classify(     csvfilename, in_subfilename, out_subfilename, classify_plus)
        
    save_csv_counts_to_csv_normirano_uporabniki_classify(        csvfilename, in_subfilename, out_subfilename, classify_plus)
    save_csv_counts_to_csv_normirano_uporabniki_plusi_minusi(    csvfilename, in_subfilename, out_subfilename, classify_plus)
    save_csv_counts_to_csv_normirano_uporabniki_avg_plusi_minusi(csvfilename, in_subfilename, out_subfilename, classify_plus)
    save_csv_counts_to_csv_normirano_uporabniki_avg_ocena(       csvfilename, in_subfilename, out_subfilename, classify_plus)
    
# ===========================================================
# save csv(normirano) to csv(points)
# ===========================================================

def save_csv_normirano_to_csv_points_uporabniki(csvfilename, in_subfilename, out_subfilename, tabfilename):

    # load normirano
    
    print_log(in_subfilename)

    izbor = "uporabniki.classify"
    print_log(izbor)
    rows_x_classify = load_from_csv(csvfilename % (in_subfilename, izbor))

    izbor = "uporabniki.avg(plusi-minusi)"
    print_log(izbor)
    rows_y_plusi_minusi = load_from_csv(csvfilename % (in_subfilename, izbor))

    izbor = "uporabniki.avg(ocena)"
    print_log(izbor)
    rows_z_ocena = load_from_csv(csvfilename % (in_subfilename, izbor))
    
    # create points
    
    rows = {}
    
    for idUporabnik, classify, count, razlika, max_razlika, normirano in rows_x_classify:
        x_classify = float(normirano)
        if idUporabnik not in rows:
            rows[idUporabnik] = (0, 0, 0)
        (x, y_plusi_minusi, z_ocena) = rows[idUporabnik]    
        rows[idUporabnik] = (x_classify, y_plusi_minusi, z_ocena)
    
    for idUporabnik, razlika, max_razlika, normirano in rows_y_plusi_minusi:
        y_plusi_minusi = float(normirano)
        if idUporabnik not in rows:
            rows[idUporabnik] = (0, 0, 0)
        (x_classify, y, z_ocena) = rows[idUporabnik]    
        rows[idUporabnik] = (x_classify, y_plusi_minusi, z_ocena)

    for idUporabnik, ocena, max_ocena, normirano in rows_z_ocena:
        z_ocena = float(normirano)
        if idUporabnik not in rows:
            rows[idUporabnik] = (0, 0, 0)
        (x_classify, y_plusi_minusi, z) = rows[idUporabnik]    
        rows[idUporabnik] = (x_classify, y_plusi_minusi, z_ocena)
    
    # save csv points
    
    print_log(out_subfilename)
    
    izbor = "uporabniki.points(3d)"
    print_log(izbor)
    
    rows_points = []   
    for idUporabnik in rows:        
        (x_classify, y_plusi_minusi, z_ocena) = rows[idUporabnik]
        rows_points.append((x_classify, y_plusi_minusi, z_ocena))
    
    save_to_csv(rows_points, csvfilename % (out_subfilename, izbor), ("x_classify", "y_plusi_minusi", "z_ocena"))
    save_to_tab(rows_points, tabfilename % (out_subfilename, izbor), ("x_classify", "y_plusi_minusi", "z_ocena"), ("c", "c", "c"))
    
    izbor = "uporabniki.points(2d)xy"
    print_log(izbor)
    rows_points = []   
    for idUporabnik in rows:        
        (x_classify, y_plusi_minusi, z_ocena) = rows[idUporabnik]
        rows_points.append((x_classify, y_plusi_minusi))
    save_to_tab(rows_points, tabfilename % (out_subfilename, izbor), ("x_classify", "y_plusi_minusi"), ("c", "c"))

    izbor = "uporabniki.points(2d)xz"
    print_log(izbor)
    rows_points = []   
    for idUporabnik in rows:        
        (x_classify, y_plusi_minusi, z_ocena) = rows[idUporabnik]
        rows_points.append((x_classify, z_ocena))
    save_to_tab(rows_points, tabfilename % (out_subfilename, izbor), ("x_classify", "z_ocena"), ("c", "c"))

    izbor = "uporabniki.points(2d)yz"
    print_log(izbor)
    rows_points = []   
    for idUporabnik in rows:        
        (x_classify, y_plusi_minusi, z_ocena) = rows[idUporabnik]
        rows_points.append((y_plusi_minusi, z_ocena))
    save_to_tab(rows_points, tabfilename % (out_subfilename, izbor), ("y_plusi_minusi", "z_ocena"), ("c", "c"))

# -------------------------------------

def save_csv_normirano_to_csv_points(csvfilename, tabfilename):
    in_subfilename = "normirano"
    out_subfilename = "points"

    save_csv_normirano_to_csv_points_uporabniki(csvfilename, in_subfilename, out_subfilename, tabfilename)


# ===========================================================
# main
# ===========================================================    

if __name__ == "__main__":
    
    t0_total = time.time()

    db = MySQLdb.connect("localhost", "root", "rootadmin", "mydb", charset="utf8")
    
    # ================================================

    # ----------------------------------------
    # parametri
    # ----------------------------------------

    data_path  = "data\\"   
    csv_path = data_path + "csv\\" 
    csv_file = csv_path + "r_%s_%s.csv"
    tab_file = csv_path + "r_%s_%s.tab"

    result_path = data_path
    log_file = result_path + "rudarjenje_log(%s).txt"

    classify_plus = '+'
             
    opravilo = "points"  # "counts", "odstotek", "normirano", "points"
    
    idUporabniki_slabi = "'mirn-an','el-cartel','ssdrag','tenisac-rdn4','jernejt','pinkfranc'"  # classify    
    #idUporabniki_slabi += ",'senna-maze-28','scenic','daryankoff','ponudnik','tuditi'"         # plusi/minusi (v primeru da ni avg razlike)

    # ----------------------------------------
    # rudarjenje
    # ----------------------------------------
    
    print_log("rudarjenje")

    if opravilo == "counts": 
        load_from_db_to_csv_counts(db, csv_file, idUporabniki_slabi)
    elif opravilo == "odstotek": 
        save_csv_counts_to_csv_odstotek(csv_file)
    elif opravilo == "normirano": 
        save_csv_counts_to_csv_normirano(csv_file, classify_plus)
    elif opravilo == "points": 
        save_csv_normirano_to_csv_points(csv_file, tab_file)

        
    # ================================================

    db.close()
    
    print_log("skupni čas = %.3f s" % (time.time() - t0_total))
    print_log("OK")
    write_log(log_file % opravilo)
