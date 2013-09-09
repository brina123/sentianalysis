select 
	komentar.idKomentar, 
	komentar.plusi, 
	komentar.minusi, 
	novica.url, 
	novica.naslov, 
	komentar.besedilo 
from  
	komentar,
	novica
where 
	novica.idNovica = komentar.Novica_idNovica 
  and novica.kategorija = 'sport'
  and komentar.besedilo not like '@%'
  and trim(komentar.besedilo) != ''
  and idKomentar mod 400 = 0
limit 1000;

-- ==================================================================


select count(*) from komentar;
-- 1606226

select count(*) from novica;
-- 42637

select count(*) from novica where kategorija = 'sport';
-- 10401

select count(*) from komentar, novica where novica.kategorija = 'sport' and komentar.Novica_idNovica = novica.idNovica;
-- 425362

select count(*) from komentar, novica where novica.kategorija = 'sport' and komentar.Novica_idNovica = novica.idNovica 
and trim(komentar.besedilo) != '';
-- 424464

select count(*) from komentar, novica where novica.kategorija = 'sport' and komentar.Novica_idNovica = novica.idNovica 
and trim(komentar.besedilo) != '' and komentar.besedilo not like '@%';
-- 400781

-- ==================================================================

SELECT komentar.idKomentar, komentar.besedilo FROM komentar, novica 
WHERE novica.kategorija = 'sport' AND komentar.Novica_idNovica = novica.idNovica AND 
			TRIM(komentar.besedilo) != '' AND komentar.besedilo NOT LIKE '@%'; 
-- 400781
	
-- ==================================================================

SELECT count(*) FROM komentar, novica 
WHERE novica.kategorija = 'sport' AND komentar.Novica_idNovica = novica.idNovica AND 
      NOT EXISTS(SELECT id FROM klasifikacije where id = komentar.idKomentar) AND 
			TRIM(komentar.besedilo) != '' AND komentar.besedilo NOT LIKE '@%'; 
-- 468

-- ==================================================================

select count(distinct Uporabnik_idUporabnik) from komentar;
-- 19084

select count(distinct Uporabnik_idUporabnik) from klasifikacije, komentar where id = idKomentar;
-- 10886

select Uporabnik_idUporabnik,           count(*) from klasifikacije, komentar where id = idKomentar group by Uporabnik_idUporabnik;
select Uporabnik_idUporabnik, classify, count(*) from klasifikacije, komentar where id = idKomentar group by Uporabnik_idUporabnik, classify;


