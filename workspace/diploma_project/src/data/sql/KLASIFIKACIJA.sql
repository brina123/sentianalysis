
-- ==================================================================
-- Svm   +,-
-- ==================================================================

select count(*) from klasifikacije;
-- 400313	

select classify, count(*) from klasifikacije group by classify;
-- + 119616		29,88%		f=0,426
-- - 280697   70,12%			
--------------------   
--   400313	

-- ==================================================================
-- Svc    +,-,N  p=0.7 
-- ==================================================================

select count(*) from klasifikacije;
-- 400313

select classify, count(*) from klasifikacije group by classify;
-- +  95512		23,86%		
-- - 178566 	44,61%
-- N 126235		31,53%
--------------------
--   400313

-- +  95512		34,85%		f=0,535	
-- - 178566 	65,15%
--------------------
--   274078

-- ==================================================================

select count(distinct podkategorija) from novica;
-- 89

SELECT count(distinct novica.podkategorija) FROM komentar, novica, klasifikacije WHERE komentar.idKomentar = klasifikacije.id AND novica.idNovica = komentar.Novica_idNovica;
-- 21

