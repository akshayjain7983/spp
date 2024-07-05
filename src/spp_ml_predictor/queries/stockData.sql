<<loadSecurityExchangeCodes>>
SELECT s.id, s.exchange_code, e."name" exchange, es."name" segment, s.exchange_group, s.security_name, s.status, s.nsurl
FROM spp.securities s
INNER JOIN spp.exchange_segments es
ON s.exchange_segment_id = es.id
AND es.status = 'Active'
INNER JOIN spp.exchanges e
ON es.exchange_id = e.id
WHERE
e."name" = :exchange
AND
s.status = 'Active'
{}

<<loadSecurityPrices>>
SELECT sp.id, sp.security_id, s.exchange_code, es."name" segment, e."name" exchange, sp."date", sp."open", sp.high, sp.low, sp."close", sp.volume
FROM spp.security_prices sp
INNER JOIN spp.securities s
ON sp.security_id = s.id
AND s.status = 'Active'
INNER JOIN spp.exchange_segments es
ON s.exchange_segment_id = es.id
AND es.status = 'Active'
INNER JOIN spp.exchanges e
ON es.exchange_id = e.id
WHERE
e."name" = :exchange
AND sp."date" BETWEEN :trainingStartDate AND :trainingEndDate
AND exchange_code IN ({})

<<loadIndexLevels>>
SELECT il.id, il.index_id, i."index", e."name" exchange, il."date", il."open", il.high, il.low, il."close"
FROM spp.index_levels il
INNER JOIN spp.indices i
ON i.id = il.index_id
AND i.status = 'Active'
INNER JOIN spp.exchanges e
ON e.id = i.exchange_id
WHERE
e."name" = :exchange
AND i."index" = :index
AND il."date" BETWEEN :trainingStartDate AND :trainingEndDate

<<loadInterestRates>>
SELECT ir.institution, ir."date", ir.rate, ir.rate_type
FROM spp.interest_rates ir
WHERE institution = :institution
AND rate_type = :rateType

<<loadInflationRates>>
SELECT ir.institution, ir."date", ir.rate, ir.rate_type
FROM spp.inflation_rates ir
WHERE institution = :institution
AND rate_type = :rateType

<<saveForecastedIndexReturn>>
UPDATE spp.forecast_index_returns
SET is_active = FALSE,
last_updated_timestamp = now()
WHERE
index_id = :index_id
AND "date" = :date
AND forecast_model_name = :forecast_model_name
AND forecast_period = :forecast_period
AND is_active = TRUE;

INSERT INTO spp.forecast_index_returns
(index_id, "date", forecast_model_name, forecast_period, forecast_date, forecasted_level, forecasted_return)
VALUES(:index_id, :date, :forecast_model_name, :forecast_period, :forecast_date, :forecasted_level, :forecasted_return);

<<saveForecastedSecurityReturn>>
UPDATE spp.forecast_security_returns
SET is_active = FALSE,
last_updated_timestamp = now()
WHERE
security_id = :security_id
AND "date" = :date
AND forecast_model_name = :forecast_model_name
AND forecast_period = :forecast_period
AND is_active = TRUE;

INSERT INTO spp.forecast_security_returns
(security_id, "date", forecast_model_name, forecast_period, forecast_date, forecasted_price, forecasted_return)
VALUES(:security_id, :date, :forecast_model_name, :forecast_period, :forecast_date, :forecasted_price, :forecasted_return);


<<loadForecastedIndexReturns>>
SELECT fis.*
FROM spp.forecast_index_returns fis
INNER JOIN spp.indices i
ON fis.index_id = i.id
AND i.status = 'Active'
INNER JOIN spp.exchanges e
ON i.exchange_id = e.id
WHERE
e."name" = :exchange
AND i."index" = :index
AND fis."date" = :pScoreDate
AND fis.forecast_period = concat(:forecastDays, 'd')
AND fis.forecast_model_name = :forecastor
AND fis.is_active = TRUE


<<loadForecastedPScore>>
SELECT fps."date", fir.forecasted_return AS forecasted_index_return, fsr.forecasted_return AS forecasted_security_return, fps.forecasted_p_score
FROM spp.forecast_p_score fps
INNER JOIN spp.forecast_index_returns fir
ON fir.id = fps.forecast_index_returns_id
AND fir.is_active = TRUE
INNER JOIN spp.forecast_security_returns fsr
ON fsr.id = fps.forecast_security_returns_id
AND fsr.is_active = TRUE
INNER JOIN spp.indices i
ON i.id = fir.index_id
AND i.status = 'Active'
INNER JOIN spp.exchanges e
ON e.id = i.exchange_id
INNER JOIN spp.securities s
ON s.id = fsr.security_id
AND s.status = 'Active'
WHERE
e."name" = :exchange
AND i."index" = :index
AND fir."date" BETWEEN :startDate AND :endDate
AND fir.forecast_model_name = :forecastModel
AND fir.forecast_period = :forecastPeriod
AND s.exchange_code IN ({})

<<loadActualPScore>>
SELECT aps.*, ir."return" actual_index_return, sr."return" actual_security_return
FROM spp.actual_p_score aps
INNER JOIN spp.securities s
ON aps.security_id = s.id
AND s.status = 'Active'
INNER JOIN spp.indices i
ON aps.index_id = i.id
AND i.status = 'Active'
INNER JOIN spp.exchange_segments es
ON s.exchange_segment_id = es.id
AND es.status = 'Active'
INNER JOIN spp.exchanges e
ON es.exchange_id = e.id
AND i.exchange_id = e.id
INNER JOIN spp.index_returns ir
ON ir.id = aps.index_return_id
INNER JOIN spp.security_returns sr
ON sr.id = aps.security_return_id
WHERE
e."name" = :exchange
AND i."index" = :index
AND s.exchange_code IN ({})
AND aps."date" BETWEEN :startDate AND :endDate
AND aps.score_period = :scorePeriod

<<loadHolidays>>
WITH RECURSIVE t(d, n) AS (
    VALUES (COALESCE(:currentDate, current_date),1)
  UNION ALL
    SELECT d-1, n+1 FROM t
    WHERE n <= COALESCE(:days, (10*365))
)
SELECT d "date", spp.is_holiday(:exchange, :segment, d) FROM t
ORDER BY d ASC

<<saveForecastedPScore>>
INSERT INTO spp.forecast_p_score ("date", forecast_index_returns_id, forecast_security_returns_id, forecasted_p_score)
SELECT fir."date", fir.id AS forecast_index_returns_id, fsr.id AS forecast_security_returns_id, (fsr.forecasted_return - fir.forecasted_return) * 100 AS forecasted_p_score
FROM spp.forecast_index_returns fir
INNER JOIN spp.forecast_security_returns fsr
ON fir."date" = fsr."date"
AND fir.forecast_model_name = fsr.forecast_model_name
AND fir.forecast_period = fsr.forecast_period
AND fir.is_active = TRUE
AND fsr.is_active = TRUE
INNER JOIN spp.indices i
ON i.id = fir.index_id
AND i.status = 'Active'
INNER JOIN spp.exchanges e
ON e.id = i.exchange_id
INNER JOIN spp.securities s
ON s.id = fsr.security_id
AND s.status = 'Active'
INNER JOIN spp.exchange_segments es
ON es.id = s.exchange_segment_id
AND es.exchange_id = e.id
LEFT OUTER JOIN spp.forecast_p_score fps
ON fps.forecast_index_returns_id = fir.id
AND fps.forecast_security_returns_id = fsr.id
WHERE
e."name" = :exchange
AND i."index" = :index
AND es."name" = :segment
AND fir."date" = :pScoreDate
AND fir.forecast_model_name = :forecastor
AND fir.forecast_period = :forecastPeriod
AND (fps.forecast_index_returns_id IS NULL
		OR fps.forecast_security_returns_id IS NULL)
AND s.exchange_code IN ({})