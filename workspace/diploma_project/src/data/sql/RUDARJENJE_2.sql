
-- ------------------------------------------------------------------
-- 1. ali so pozitivni('+') komentarji ocenjeni bolj z plusi ali minusi, kako pa negativni
--    procent plusov na pozitivnih/negativnih (klasificiranih) komentarjih, procent minusov na pozitivnih/negativnih (klasificiranih) komentarjih 
-- ------------------------------------------------------------------
 
SELECT klasifikacije.classify, count(*) classify_count, 
 SUM(komentar.plusi) sum_plusi, AVG(komentar.plusi) avg_plusi, 
 SUM(komentar.minusi) sum_minusi, AVG(komentar.minusi) avg_minusi
FROM komentar, klasifikacije
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id 
GROUP BY klasifikacije.classify;

-- rudarjenje_2_1.csv

-- ------------------------------------------------------------------
-- 2. èasovni trend, ali se spreminja. - za najbolj komentirane športe (6) kako se spreminja pozitivnost komentarjev skozi èas v obdobju enega leta
--    v obdobju 1.10.2011 in 1.7.2013 - za vsak mesec procent pozitivnih in procent negativnih komentarjev (klasifikacije) za 6 najbolj komentiranih športov
-- ------------------------------------------------------------------

SELECT novica.podkategorija, COUNT(*) podkategorija_count 
FROM komentar, novica, klasifikacije 
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica 
GROUP BY novica.podkategorija
ORDER BY podkategorija_count DESC;

-- rudarjenje_2_2a.csv

SELECT novica.podkategorija, YEAR(novica.datum) leto,  MONTH(novica.datum) mesec, klasifikacije.classify, COUNT(*) classify_count 
FROM komentar, novica, klasifikacije 
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica AND
novica.datum < DATE('2013-07-01') AND novica.podkategorija IN ('podkategorija', 'nogomet', 'zimski-sporti', 'kosarka', 'hokej', 'tenis', 'rokomet')
GROUP BY novica.podkategorija, YEAR(novica.datum),  MONTH(novica.datum), klasifikacije.classify 
ORDER BY novica.podkategorija, leto, mesec, klasifikacije.classify;

-- rudarjenje_2_2b.csv

SELECT podkategorija, leto, mesec, COUNT(*) classify_count, 
 SUM(p) sum_p, AVG(p) avg_p, 
 SUM(m) sum_m, AVG(m) avg_m 
FROM (
 SELECT novica.podkategorija, YEAR(novica.datum) leto,  MONTH(novica.datum) mesec,
 IF(klasifikacije.classify = '+', 1, 0) p, 
 IF(klasifikacije.classify = '-', 1, 0) m
 FROM komentar, novica, klasifikacije 
 WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica AND
 novica.datum < DATE('2013-07-01') AND novica.podkategorija IN ('podkategorija', 'nogomet', 'zimski-sporti', 'kosarka', 'hokej', 'tenis', 'rokomet')
) V
GROUP BY podkategorija, leto, mesec
ORDER BY podkategorija, leto, mesec;

-- rudarjenje_2_2c.csv


-- ------------------------------------------------------------------
-- 3. kateri novinar ima najboljše ocenjene novice 
--    povpreèje ocen novic po novinarjih
-- ------------------------------------------------------------------

SELECT novica.Novinar_idNovinar, COUNT(*) novinar_count
FROM novica
GROUP BY novica.Novinar_idNovinar
ORDER BY novinar_count DESC;

-- rudarjenje_2_3a.csv

SELECT novica.Novinar_idNovinar, COUNT(*) ocena_count,
  ROUND(SUM(novica.ocena), 3) sum_ocena, ROUND(AVG(novica.ocena), 3) avg_ocena
FROM novica
GROUP BY novica.Novinar_idNovinar
ORDER BY ocena_count desc;

-- rudarjenje_2_3b.csv

-- ------------------------------------------------------------------
-- 4. ali imajo novice z boljšo oceno veè komentarjev
--    povpreèno število komentarjev pri novicah z ocenami med 4-5, 3-4 in ostale 0-3
-- ------------------------------------------------------------------

SELECT novica.ocena, COUNT(*) komentarji_count 
FROM novica, komentar
WHERE komentar.Novica_idNovica = novica.idNovica
GROUP BY novica.ocena
ORDER BY novica.ocena DESC;

-- rudarjenje_2_4a.csv

SELECT 
  IF(novica.ocena >= 4, '4 do 5', 
	IF(novica.ocena >= 3, '3 do 4', 
	IF(novica.ocena >= 2, '2 do 3', 
												'0 do 2'))) ocena_od_do,
  COUNT(*) komentarji_count 
FROM novica, komentar
WHERE komentar.Novica_idNovica = novica.idNovica
GROUP BY 
  IF(novica.ocena >= 4, '4 do 5', 
	IF(novica.ocena >= 3, '3 do 4', 
	IF(novica.ocena >= 2, '2 do 3', 
												'0 do 2')))
ORDER BY ocena_od_do DESC;

-- rudarjenje_2_4b.csv

-- ------------------------------------------------------------------
-- 5. ali imajo novice, ki so bolje ocenjene tudi bolj pozitivne komentarje
--    procent pozitivnih komentarjev na novice z oceno med 4 - 5, 3 - 4, in ostale
-- ------------------------------------------------------------------

SELECT ocena_od_do, COUNT(*) ocena_count, 
 SUM(p) sum_p, AVG(p) avg_p, 
 SUM(m) sum_m, AVG(m) avg_m 
FROM (
 SELECT 
  IF(novica.ocena >= 4, '4 do 5', 
	IF(novica.ocena >= 3, '3 do 4', 
	IF(novica.ocena >= 2, '2 do 3', 
												'0 do 2'))) ocena_od_do,
 IF(klasifikacije.classify = '+', 1, 0) p, 
 IF(klasifikacije.classify = '-', 1, 0) m
 FROM komentar, novica, klasifikacije 
 WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica
) V
GROUP BY ocena_od_do
ORDER BY ocena_od_do DESC;

-- rudarjenje_2_5.csv
