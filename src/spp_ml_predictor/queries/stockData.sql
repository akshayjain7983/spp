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

<<saveForecastedPScoreTable>>
spp.forecast_p_score

<<saveForecastedPScore>>
UPDATE spp.forecast_p_score
SET is_active = FALSE
WHERE
security_id = :security_id
AND index_id = :index_id
AND "date" = :date
AND forecast_model_name = :forecast_model_name
AND forecast_period = :forecast_period
AND is_active = TRUE;

INSERT INTO spp.forecast_p_score
(security_id, index_id, "date", forecast_model_name, forecast_period, forecast_date, forecasted_index_return, forecasted_security_return, forecasted_p_score)
VALUES(:security_id, :index_id, :date, :forecast_model_name, :forecast_period, :forecast_date, :forecasted_index_return, :forecasted_security_return, :forecasted_p_score);
