-- ==================================================================
-- PODKATEGORIJE
-- ==================================================================

SELECT novica.podkategorija, klasifikacije.classify, COUNT(*) FROM komentar, novica, klasifikacije 
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica 
GROUP BY novica.podkategorija, klasifikacije.classify;

-- ==================================================================
-- 70% vseh komentarjev da 900 uporabnikov 
-- ==================================================================

SELECT komentar.Uporabnik_idUporabnik, COUNT(*) FROM komentar, klasifikacije
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id 
GROUP BY komentar.Uporabnik_idUporabnik
HAVING COUNT(*) > 65 
ORDER BY COUNT(*) DESC;

-- uporabniki_cnt_komentarji.csv

SELECT komentar.Uporabnik_idUporabnik, COUNT(*) FROM komentar, klasifikacije
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND
      komentar.Uporabnik_idUporabnik NOT IN (
			 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'
			)
GROUP BY komentar.Uporabnik_idUporabnik
HAVING COUNT(*) > 65 
ORDER BY COUNT(*) DESC;

-- uporabniki_cnt_komentarji_899.csv


-- ==================================================================

SELECT count(*) from 
(
SELECT komentar.Uporabnik_idUporabnik, COUNT(*) FROM komentar, klasifikacije
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND
      komentar.Uporabnik_idUporabnik NOT IN (
			 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'
			)
GROUP BY komentar.Uporabnik_idUporabnik
HAVING COUNT(*) > 65 
) a;
-- 893

-- ==================================================================
-- UPORABNIKI
-- ==================================================================

-- Dimenzije:
-- 
-- 1. Procent pozitivnih komentarjev uporabnika : 
-- 		sum(+) / sum(all) * 100 = procent
-- 
-- 2. Priljubljenost komentarja od uporabnika :	 
-- 	 	Pomeni koliko imajo njegovi komentar plusov in koliko minusov
-- 	 	sum(komentar.plusi) sum(komenatar.minusi)
--
--  2.a  povpreèje avg(plusi - minusi) na komentar
--
--  2.b  povpreèje avg(plusi - minusi) na novico
--       delejeno z n komentarjev na novico 
--     
--       najprej za vsak novico pogledamo povpreèje razlik plusov in minusov komentarjev, 
--				in tako doloèimo nulto mejo in nato glede na to nulto mejo doloèimo doloèeno 
--				povpreèje za komentar od uporabnika èe je meja -, se mu ta minus prišteje, 
--  			èe je + se mu odšteje, in potem naprej vse poteka isto kot je bilo prej. 
-- 
-- 3. Kvaliteta komentiranih novic
-- 	 	sum(Novica.ocena) / count(komentarjev)

-- ------------------------------------------------------------------

-- ------------------------------------------------------------------
-- 1 -- 
-- ------------------------------------------------------------------

-- slabi
-- 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'

SELECT komentar.Uporabnik_idUporabnik, klasifikacije.classify, COUNT(*) classify_count FROM komentar, klasifikacije
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND
      komentar.Uporabnik_idUporabnik NOT IN (
			 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'
			)
GROUP BY komentar.Uporabnik_idUporabnik, klasifikacije.classify;

SELECT Uporabnik_idUporabnik, classify, classify_count 
FROM (
  SELECT komentar.Uporabnik_idUporabnik, klasifikacije.classify, COUNT(*) classify_count FROM komentar, klasifikacije
  WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND
        komentar.Uporabnik_idUporabnik NOT IN (
		  	 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'
			  ) and
        komentar.Uporabnik_idUporabnik IN (
					SELECT komentar.Uporabnik_idUporabnik FROM komentar, klasifikacije
					WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id 
					GROUP BY komentar.Uporabnik_idUporabnik
					HAVING COUNT(*) > 65 
				)
  GROUP BY komentar.Uporabnik_idUporabnik, klasifikacije.classify
) a 
--where
--	(SELECT COUNT(*) FROM komentar WHERE komentar.Uporabnik_idUporabnik = a.Uporabnik_idUporabnik and
--   exists(select * from klasifikacije where klasifikacije.classify != 'N' and klasifikacije.id = komentar.idKomentar)
--	) > 65

	-- (SELECT COUNT(*) FROM komentar WHERE Uporabnik_idUporabnik = a.Uporabnik_idUporabnik) > 65;
	
-- ------------------------------------------------------------------
-- 2 --
-- ------------------------------------------------------------------

-- slabi
-- 'senna-maze-28', 'scenic', 'daryankoff', 'ponudnik', 'tuditi'

SELECT komentar.Uporabnik_idUporabnik, sum(komentar.plusi), sum(komentar.minusi) FROM komentar, klasifikacije
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND
      komentar.Uporabnik_idUporabnik NOT IN (
			 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'
			)
GROUP BY komentar.Uporabnik_idUporabnik;

-- ------------------------------------------------------------------
-- 2.a --
-- ------------------------------------------------------------------

SELECT komentar.Uporabnik_idUporabnik, ROUND(AVG(komentar.plusi - komentar.minusi), 0) FROM komentar, klasifikacije
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND
      komentar.Uporabnik_idUporabnik NOT IN (
			 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'
			)
GROUP BY komentar.Uporabnik_idUporabnik;

-- ------------------------------------------------------------------
-- 2.b --
-- ------------------------------------------------------------------

 -- ???
 
-- ------------------------------------------------------------------
-- 3 --
-- ------------------------------------------------------------------

SELECT komentar.Uporabnik_idUporabnik, AVG(novica.ocena) FROM komentar, klasifikacije, novica
WHERE klasifikacije.classify != 'N' AND komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica AND 
      komentar.Uporabnik_idUporabnik NOT IN (
			 'mirn-an', 'el-cartel', 'ssdrag', 'tenisac-rdn4', 'jernejt', 'pinkfranc'
			)
GROUP BY komentar.Uporabnik_idUporabnik;


