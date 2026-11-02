/*
CLTV inference query.

Run this in AMC with a FULL-MONTH window only.
Example current scoring window (early Nov run): 2026-10-01 through 2026-10-31
to score users from October features for next-month CLTV change prediction.

Output intentionally excludes pct_change_cltv_oct_to_nov because infer.py predicts it.
*/

WITH
  monthly_purchases AS (
    SELECT
      user_id,
      DATE_TRUNC ('month', event_date_utc) AS purchase_month,
      SUM(CAST(total_product_sales AS DECIMAL(18, 2))) AS monthly_sales,
      SUM(total_units_sold) AS monthly_units,
      COUNT(DISTINCT conversion_id) AS monthly_orders,
      CASE
        WHEN COUNT(DISTINCT conversion_id) > 0 THEN SUM(CAST(total_product_sales AS DECIMAL(18, 2))) / COUNT(DISTINCT conversion_id)
        ELSE 0
      END AS monthly_avg_order_value
    FROM
      conversions_all_for_audiences
    WHERE
      user_id IS NOT NULL
      AND total_product_sales > 0
      AND event_category = 'purchase'
    GROUP BY
      1,
      2
  ),
  monthly_dsp_spend AS (
    SELECT
      user_id,
      DATE_TRUNC ('month', impression_date_utc) AS spend_month,
      SUM(total_cost) / 100000 AS monthly_dsp_spend,
      SUM(clicks) AS monthly_dsp_clicks,
      SUM(impressions) AS monthly_dsp_impressions,
      COUNT(DISTINCT campaign_id) AS monthly_dsp_campaigns
    FROM
      (
        SELECT
          user_id,
          impression_date_utc,
          campaign_id,
          SUM(total_cost) AS total_cost,
          0 AS clicks,
          SUM(impressions) AS impressions
        FROM
          dsp_impressions_for_audiences
        GROUP BY
          1,
          2,
          3
        UNION ALL
        SELECT
          user_id,
          impression_date_utc,
          campaign_id,
          0 AS total_cost,
          SUM(clicks) AS clicks,
          0 AS impressions
        FROM
          dsp_clicks_for_audiences
        GROUP BY
          1,
          2,
          3
      )
    WHERE
      user_id IS NOT NULL
    GROUP BY
      1,
      2
  ),
  monthly_sa_impressions AS (
    SELECT
      user_id,
      DATE_TRUNC ('month', event_date_utc) AS spend_month,
      SUM(impressions) AS monthly_sa_impressions,
      SUM(clicks) AS monthly_sa_clicks,
      COUNT(DISTINCT campaign_id) AS monthly_sa_campaigns
    FROM
      sponsored_ads_traffic_for_audiences
    WHERE
      user_id IS NOT NULL
    GROUP BY
      1,
      2
  ),
  monthly_sa_spend AS (
    SELECT
      user_id,
      DATE_TRUNC ('month', event_date_utc) AS spend_month,
      SUM(spend) / 1000000 AS monthly_sa_spend
    FROM
      sponsored_ads_traffic_for_audiences
    WHERE
      user_id IS NOT NULL
      AND spend > 0
    GROUP BY
      1,
      2
  ),
  monthly_sa_combined AS (
    SELECT
      COALESCE(i.user_id, s.user_id) AS user_id,
      COALESCE(i.spend_month, s.spend_month) AS spend_month,
      COALESCE(i.monthly_sa_impressions, 0) AS monthly_sa_impressions,
      COALESCE(i.monthly_sa_clicks, 0) AS monthly_sa_clicks,
      COALESCE(i.monthly_sa_campaigns, 0) AS monthly_sa_campaigns,
      COALESCE(s.monthly_sa_spend, 0) AS monthly_sa_spend
    FROM
      monthly_sa_impressions i
      FULL OUTER JOIN monthly_sa_spend s ON i.user_id = s.user_id
      AND i.spend_month = s.spend_month
  ),
  campaign_tracking AS (
    SELECT
      user_id,
      DATE_TRUNC ('month', impression_date_utc) AS activity_month,
      campaign,
      'DSP' AS source
    FROM
      dsp_impressions_for_audiences
    WHERE
      user_id IS NOT NULL
      AND campaign IS NOT NULL
    UNION ALL
    SELECT
      user_id,
      DATE_TRUNC ('month', event_date_utc) AS activity_month,
      campaign,
      'SA' AS source
    FROM
      sponsored_ads_traffic_for_audiences
    WHERE
      user_id IS NOT NULL
      AND campaign IS NOT NULL
  ),
  monthly_combined AS (
    SELECT
      COALESCE(mp.user_id, ds.user_id, sa.user_id) AS user_id,
      COALESCE(mp.purchase_month, ds.spend_month, sa.spend_month) AS activity_month,
      COALESCE(mp.monthly_sales, 0) AS monthly_sales,
      COALESCE(mp.monthly_units, 0) AS monthly_units,
      COALESCE(mp.monthly_orders, 0) AS monthly_orders,
      COALESCE(mp.monthly_avg_order_value, 0) AS monthly_avg_order_value,
      COALESCE(ds.monthly_dsp_spend, 0) + COALESCE(sa.monthly_sa_spend, 0) AS monthly_spend,
      COALESCE(ds.monthly_dsp_clicks, 0) + COALESCE(sa.monthly_sa_clicks, 0) AS monthly_clicks,
      COALESCE(ds.monthly_dsp_impressions, 0) + COALESCE(sa.monthly_sa_impressions, 0) AS monthly_impressions,
      COALESCE(ds.monthly_dsp_campaigns, 0) + COALESCE(sa.monthly_sa_campaigns, 0) AS monthly_campaigns
    FROM
      monthly_purchases mp
      FULL OUTER JOIN monthly_dsp_spend ds ON mp.user_id = ds.user_id
      AND mp.purchase_month = ds.spend_month
      FULL OUTER JOIN monthly_sa_combined sa ON COALESCE(mp.user_id, ds.user_id) = sa.user_id
      AND COALESCE(mp.purchase_month, ds.spend_month) = sa.spend_month
  ),
  cumulative_calculations AS (
    SELECT
      mc1.user_id,
      mc1.activity_month,
      mc1.monthly_sales,
      mc1.monthly_units,
      mc1.monthly_orders,
      mc1.monthly_spend,
      mc1.monthly_clicks,
      mc1.monthly_impressions,
      mc1.monthly_campaigns,
      mc1.monthly_avg_order_value,
      SUM(mc2.monthly_sales) AS cumulative_sales,
      SUM(mc2.monthly_spend) AS cumulative_spend,
      SUM(mc2.monthly_sales) - SUM(mc2.monthly_spend) AS cumulative_cltv,
      SUM(mc2.monthly_units) AS cumulative_units,
      SUM(mc2.monthly_orders) AS cumulative_orders,
      SUM(mc2.monthly_clicks) AS cumulative_clicks,
      SUM(mc2.monthly_impressions) AS cumulative_impressions,
      COUNT(DISTINCT mc2.activity_month) AS months_with_activity,
      CASE
        WHEN SUM(mc2.monthly_orders) > 0 THEN SUM(mc2.monthly_sales) / SUM(mc2.monthly_orders)
        ELSE 0
      END AS cumulative_avg_order_value
    FROM
      monthly_combined mc1
      INNER JOIN monthly_combined mc2 ON mc1.user_id = mc2.user_id
      AND mc2.activity_month <= mc1.activity_month
    GROUP BY
      mc1.user_id,
      mc1.activity_month,
      mc1.monthly_sales,
      mc1.monthly_units,
      mc1.monthly_orders,
      mc1.monthly_spend,
      mc1.monthly_clicks,
      mc1.monthly_impressions,
      mc1.monthly_campaigns,
      mc1.monthly_avg_order_value
  ),
  cumulative_campaigns AS (
    SELECT
      ct1.user_id,
      ct1.activity_month,
      COUNT(DISTINCT ct2.campaign) AS cumulative_campaigns
    FROM
      campaign_tracking ct1
      INNER JOIN campaign_tracking ct2 ON ct1.user_id = ct2.user_id
      AND ct2.activity_month <= ct1.activity_month
    GROUP BY
      ct1.user_id,
      ct1.activity_month
  ),
  customer_monthly_metrics AS (
    SELECT
      ccm.user_id,
      ccm.activity_month,
      ccm.monthly_sales,
      ccm.monthly_units,
      ccm.monthly_orders,
      ccm.monthly_spend,
      ccm.monthly_clicks,
      ccm.monthly_impressions,
      ccm.monthly_campaigns,
      ccm.monthly_avg_order_value,
      ccm.cumulative_sales,
      ccm.cumulative_spend,
      ccm.cumulative_cltv,
      ccm.cumulative_units,
      ccm.cumulative_orders,
      ccm.cumulative_clicks,
      ccm.cumulative_impressions,
      ccm.months_with_activity,
      ccm.cumulative_avg_order_value,
      COALESCE(cc.cumulative_campaigns, 0) AS cumulative_campaigns,
      ccm.monthly_impressions AS monthly_frequency,
      ROW_NUMBER() OVER (
        PARTITION BY
          ccm.user_id
        ORDER BY
          ccm.activity_month DESC
      ) AS recency_rank
    FROM
      cumulative_calculations ccm
      LEFT JOIN cumulative_campaigns cc ON ccm.user_id = cc.user_id
      AND ccm.activity_month = cc.activity_month
  ),
  most_recent_month AS (
    SELECT
      MAX(activity_month) AS max_month
    FROM
      customer_monthly_metrics
  ),
  most_recent_metrics AS (
    SELECT
      cmm.user_id,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_cltv
        END
      ) AS current_month_cltv,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_sales
        END
      ) AS sales_through_current_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_spend
        END
      ) AS spend_through_current_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_clicks
        END
      ) AS clicks_through_current_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_impressions
        END
      ) AS impressions_through_current_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_avg_order_value
        END
      ) AS aov_through_current_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_campaigns
        END
      ) AS campaigns_through_current_month,
      SUM(
        CASE
          WHEN cmm.activity_month = mrm.max_month THEN cmm.monthly_impressions
          ELSE 0
        END
      ) AS current_month_frequency,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month
          OR (
            seconds_between (mrm.max_month, cmm.activity_month) > 0
            AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016
          ) THEN cmm.cumulative_cltv
        END
      ) AS next_month_cltv,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month
          OR (
            seconds_between (mrm.max_month, cmm.activity_month) > 0
            AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016
          ) THEN cmm.cumulative_sales
        END
      ) AS sales_through_next_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month
          OR (
            seconds_between (mrm.max_month, cmm.activity_month) > 0
            AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016
          ) THEN cmm.cumulative_spend
        END
      ) AS spend_through_next_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month
          OR (
            seconds_between (mrm.max_month, cmm.activity_month) > 0
            AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016
          ) THEN cmm.cumulative_clicks
        END
      ) AS clicks_through_next_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month
          OR (
            seconds_between (mrm.max_month, cmm.activity_month) > 0
            AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016
          ) THEN cmm.cumulative_impressions
        END
      ) AS impressions_through_next_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month
          OR (
            seconds_between (mrm.max_month, cmm.activity_month) > 0
            AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016
          ) THEN cmm.cumulative_avg_order_value
        END
      ) AS aov_through_next_month,
      MAX(
        CASE
          WHEN cmm.activity_month <= mrm.max_month
          OR (
            seconds_between (mrm.max_month, cmm.activity_month) > 0
            AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016
          ) THEN cmm.cumulative_campaigns
        END
      ) AS campaigns_through_next_month,
      SUM(
        CASE
          WHEN seconds_between (mrm.max_month, cmm.activity_month) > 0
          AND seconds_between (mrm.max_month, cmm.activity_month) <= 2630016 THEN cmm.monthly_impressions
          ELSE 0
        END
      ) AS next_month_frequency,
      COUNT(
        DISTINCT CASE
          WHEN cmm.monthly_sales > 0 THEN cmm.activity_month
        END
      ) AS months_active_sales,
      COUNT(
        DISTINCT CASE
          WHEN cmm.monthly_spend > 0 THEN cmm.activity_month
        END
      ) AS months_active_spend
    FROM
      customer_monthly_metrics cmm
      CROSS JOIN most_recent_month mrm
    GROUP BY
      cmm.user_id
  )
SELECT
  onm.user_id,
  onm.months_active_sales,
  onm.months_active_spend,
  onm.sales_through_current_month,
  onm.spend_through_current_month,
  onm.clicks_through_current_month,
  onm.impressions_through_current_month,
  onm.aov_through_current_month,
  onm.campaigns_through_current_month,
  onm.current_month_frequency,
  onm.sales_through_next_month,
  onm.spend_through_next_month,
  onm.clicks_through_next_month,
  onm.impressions_through_next_month,
  onm.aov_through_next_month,
  onm.campaigns_through_next_month,
  onm.next_month_frequency
FROM
  most_recent_metrics onm
WHERE
  onm.user_id IS NOT NULL
  AND onm.current_month_cltv IS NOT NULL
