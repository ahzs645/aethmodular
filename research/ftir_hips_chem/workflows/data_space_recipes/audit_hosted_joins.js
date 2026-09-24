/**
 * Read-only cross-dataset checks against Zoer's hosted DuckDB artifacts.
 *
 * Run from a Zoer backend pod that has @duckdb/node-api and /data/datasets:
 *   kubectl -n zoer exec -i deploy/zoer-backend -- bun run - < audit_hosted_joins.js
 * Each artifact is copied to a temporary directory because the NFS-backed
 * originals have a DuckDB lock. The temporary copies are always removed.
 */
import { DuckDBInstance } from "@duckdb/node-api";
import { copyFile, mkdtemp, rm } from "node:fs/promises";

const datasetIds = {
  hips: "c5a0dd7b-1068-41d3-b023-309998a70823",
  ftir: "9f73e74d-31f3-44da-bd3f-8ba8513405a0",
  unified: "caa04b1f-7f86-4c34-b059-a5a8d04c7501",
  davis: "aceb7980-151a-43b5-9c74-239511d2cef2",
  chemFour: "f82f3a0f-8366-445a-9cfc-c4f60b1c77c6",
  chemPublic: "7955ed6c-3b53-46e1-8443-caa80ae64cdf",
  sites: "5576fd5b-f39b-41bb-919f-434b77bfc8f3",
  ftirArchive: "b7069975-a26d-4928-9bcb-21718c81fd5f",
  improvePortal: "768e3bb0-d2c8-45ea-8b7f-a50fad328cf9",
  improveTor: "bf0bc7b0-6a0b-4948-b6cc-9428f4572e87",
};

const root = process.env.ZOER_DATASETS_DIR ?? "/data/datasets";
const scratch = await mkdtemp("/tmp/aeth-hosted-joins-");
const instance = await DuckDBInstance.create(":memory:");
const connection = await instance.connect();

async function report(name, sql) {
  const reader = await connection.runAndReadAll(sql);
  console.log(JSON.stringify({ check: name, rows: reader.getRowObjectsJson() }));
}

try {
  for (const [alias, id] of Object.entries(datasetIds)) {
    const copy = `${scratch}/${alias}.duckdb`;
    await copyFile(`${root}/${id}/dataset.duckdb`, copy);
    await connection.run(`ATTACH '${copy}' AS ${alias} (READ_ONLY)`);
  }

  await report("hips_grain", `
    SELECT count(*) row_count, count(DISTINCT FilterId) filter_ids,
      count(DISTINCT regexp_replace(FilterId, '-[0-9]+$', '')) base_ids,
      count(*) FILTER (WHERE SampleDate IS NULL) missing_sample_date
    FROM hips.main.spartan_hips_batch1_51_v2`);

  await report("hips_revision", `
    SELECT count(*) matched_filter_ids,
      count(*) FILTER (WHERE try_cast(old.Fabs AS DOUBLE) IS DISTINCT FROM new.Fabs) changed_fabs,
      count(*) FILTER (WHERE try_cast(old.SampleDate AS DATE) IS DISTINCT FROM new.SampleDate) changed_sample_date
    FROM hips.main.spartan_hips_batch1_51 old
    JOIN hips.main.spartan_hips_batch1_51_v2 new USING (FilterId)`);

  await report("four_site_ftir_in_unified", `
    WITH f AS (SELECT FilterId, Parameter, SampleDate, Concentration_ug_m3 FROM ftir.main.four_sites_ftir_data_v2),
    u AS (SELECT FilterId, Parameter, try_cast(SampleDate AS DATE) SampleDate, Concentration FROM unified.main.unified_filter_dataset)
    SELECT (SELECT count(*) FROM f) source_rows,
      (SELECT count(*) FROM f JOIN u USING (FilterId, Parameter)) joined_rows,
      (SELECT count(*) FROM f JOIN u USING (FilterId, Parameter) WHERE f.SampleDate IS DISTINCT FROM u.SampleDate) changed_sample_dates,
      (SELECT count(*) FROM f JOIN u USING (FilterId, Parameter) WHERE f.Concentration_ug_m3 IS DISTINCT FROM u.Concentration) changed_concentrations,
      (SELECT count(*) FROM (SELECT FilterId, Parameter FROM u GROUP BY 1,2 HAVING count(*) > 1)) duplicate_unified_keys`);

  await report("hips_ftir_same_full_filter", `
    WITH f AS (SELECT FilterId, min(Site) Site, min(SampleDate) SampleDate,
      count(DISTINCT Site) sites, count(DISTINCT SampleDate) dates
      FROM ftir.main.four_sites_ftir_data_v2 GROUP BY FilterId),
    h AS (SELECT FilterId, Site, SampleDate FROM hips.main.spartan_hips_batch1_51_v2)
    SELECT (SELECT count(*) FROM f) ftir_filter_ids,
      (SELECT count(*) FROM f WHERE sites > 1 OR dates > 1) inconsistent_ftir_filter_metadata,
      (SELECT count(*) FROM f JOIN h USING (FilterId)) shared_full_filter_ids,
      (SELECT count(*) FROM f JOIN h USING (FilterId) WHERE f.Site IS DISTINCT FROM h.Site) site_disagreements,
      (SELECT count(*) FROM f JOIN h USING (FilterId) WHERE f.SampleDate IS NOT NULL AND h.SampleDate IS NOT NULL AND f.SampleDate IS DISTINCT FROM h.SampleDate) known_date_disagreements,
      (SELECT count(*) FROM f JOIN h USING (FilterId) WHERE h.SampleDate IS NULL AND f.SampleDate IS NOT NULL) ftir_dates_for_missing_hips`);

  await report("four_site_ftir_differences", `
    SELECT f.FilterId, f.Parameter, f.Concentration_ug_m3 source_value,
      u.Concentration unified_value, u.DataSource
    FROM ftir.main.four_sites_ftir_data_v2 f
    JOIN unified.main.unified_filter_dataset u USING (FilterId, Parameter)
    WHERE f.Concentration_ug_m3 IS DISTINCT FROM u.Concentration
    ORDER BY f.FilterId, f.Parameter LIMIT 15`);

  await report("davis_filter_intervals", `
    WITH metadata AS (
      SELECT 'ETAD' site, ExternalFilterId id, cast(SamplingStartDate AS DATE) start_date, cast(SamplingEndDate AS DATE) end_date FROM davis.main.etad_metadata_7aa7cf5adf
      UNION ALL SELECT 'CHTS', ExternalFilterId, cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.chts_filters_abc680ff97
      UNION ALL SELECT 'ETBI', ExternalFilterId, cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.etbi_filters_f8485e5e98
      UNION ALL SELECT 'INDH', ExternalFilterId, cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.indh_filters_a998fa6d87
      UNION ALL SELECT 'USPA', ExternalFilterId, cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.uspa_filters_9b7fa2e7d3
    ), unique_interval AS (SELECT id, min(start_date) start_date, min(end_date) end_date, count(*) n,
      count(DISTINCT struct_pack(start_date, end_date)) interval_count FROM metadata GROUP BY id)
    SELECT count(*) source_rows, count(DISTINCT id) filter_ids,
      count(*) FILTER (WHERE start_date IS NULL OR end_date IS NULL) missing_interval,
      count(*) FILTER (WHERE end_date < start_date) inverted_interval,
      (SELECT count(*) FROM unique_interval WHERE n > 1) repeated_filter_ids,
      (SELECT count(*) FROM unique_interval WHERE interval_count > 1) conflicting_intervals,
      (SELECT count(*) FROM unique_interval m JOIN hips.main.spartan_hips_batch1_51_v2 h ON m.id = h.FilterId) full_id_hips_matches,
      (SELECT count(*) FROM unique_interval m JOIN hips.main.spartan_hips_batch1_51_v2 h ON m.id = h.FilterId WHERE m.start_date IS DISTINCT FROM h.SampleDate) hips_sample_date_mismatches
    FROM metadata`);

  await report("davis_by_site", `
    WITH metadata AS (
      SELECT 'ETAD' site, ExternalFilterId id FROM davis.main.etad_metadata_7aa7cf5adf
      UNION ALL SELECT 'CHTS', ExternalFilterId FROM davis.main.chts_filters_abc680ff97
      UNION ALL SELECT 'ETBI', ExternalFilterId FROM davis.main.etbi_filters_f8485e5e98
      UNION ALL SELECT 'INDH', ExternalFilterId FROM davis.main.indh_filters_a998fa6d87
      UNION ALL SELECT 'USPA', ExternalFilterId FROM davis.main.uspa_filters_9b7fa2e7d3
    ) SELECT m.site, count(*) row_count, count(DISTINCT id) filter_ids,
      count(DISTINCT h.FilterId) hips_matches
    FROM metadata m LEFT JOIN hips.main.spartan_hips_batch1_51_v2 h ON m.id = h.FilterId GROUP BY m.site ORDER BY m.site`);

  await report("davis_interval_completeness_by_site", `
    WITH metadata AS (
      SELECT 'ETAD' site, cast(SamplingStartDate AS DATE) start_date, cast(SamplingEndDate AS DATE) end_date FROM davis.main.etad_metadata_7aa7cf5adf
      UNION ALL SELECT 'CHTS', cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.chts_filters_abc680ff97
      UNION ALL SELECT 'ETBI', cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.etbi_filters_f8485e5e98
      UNION ALL SELECT 'INDH', cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.indh_filters_a998fa6d87
      UNION ALL SELECT 'USPA', cast(SamplingStartDate AS DATE), cast(SamplingEndDate AS DATE) FROM davis.main.uspa_filters_9b7fa2e7d3
    ) SELECT site, count(*) row_count,
      count(*) FILTER (WHERE start_date IS NULL OR end_date IS NULL) missing_window
    FROM metadata GROUP BY site ORDER BY site`);

  await report("chem_four_site_grain", `
    WITH chem AS (
      SELECT * FROM chemFour.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_uspa
    ), keys AS (SELECT Site_Code, Filter_ID, Parameter_Code, Method_Code, count(*) n FROM chem GROUP BY 1,2,3,4)
    SELECT (SELECT count(*) FROM chem) row_count,
      (SELECT count(*) FROM keys) distinct_measurement_keys,
      (SELECT count(*) FROM keys WHERE n > 1) repeated_measurement_keys,
      (SELECT count(DISTINCT Filter_ID) FROM chem) base_filter_ids,
      (SELECT count(*) FROM chem WHERE Start_Year_local IS NULL OR End_Year_local IS NULL) missing_years`);

  await report("chem_four_site_vs_public", `
    WITH f AS (
      SELECT * FROM chemFour.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_uspa
    ), p AS (
      SELECT * FROM chemPublic.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT * FROM chemPublic.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT * FROM chemPublic.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT * FROM chemPublic.main.filterbased_chemspecpm25_uspa
    )
    SELECT (SELECT count(*) FROM f) four_site_rows,
      (SELECT count(*) FROM p) public_rows_at_four_sites,
      (SELECT count(*) FROM f JOIN p USING (Site_Code, Filter_ID, Parameter_Code, Method_Code)) key_matches,
      (SELECT count(*) FROM f JOIN p USING (Site_Code, Filter_ID, Parameter_Code, Method_Code) WHERE f.Value IS DISTINCT FROM p.Value) changed_values,
      (SELECT count(*) FROM f JOIN p USING (Site_Code, Filter_ID, Parameter_Code, Method_Code) WHERE f.Start_Year_local IS DISTINCT FROM p.Start_Year_local OR f.Start_Month_local IS DISTINCT FROM p.Start_Month_local OR f.Start_Day_local IS DISTINCT FROM p.Start_Day_local) changed_start_dates`);

  await report("chem_four_site_exact_overlap", `
    WITH f AS (
      SELECT * FROM chemFour.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_uspa
    ), p AS (
      SELECT * FROM chemPublic.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT * FROM chemPublic.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT * FROM chemPublic.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT * FROM chemPublic.main.filterbased_chemspecpm25_uspa
    )
    SELECT (SELECT count(*) FROM (SELECT * FROM f INTERSECT ALL SELECT * FROM p)) exact_shared_rows,
      (SELECT count(*) FROM (SELECT * FROM f EXCEPT ALL SELECT * FROM p)) four_site_only_rows,
      (SELECT count(*) FROM (SELECT * FROM p EXCEPT ALL SELECT * FROM f)) public_only_rows,
      (SELECT count(*) FROM f) four_site_rows, (SELECT count(*) FROM p) public_rows`);

  await report("chem_repeated_keys_example", `
    SELECT Site_Code, Filter_ID, Parameter_Code, Method_Code, count(*) n,
      count(DISTINCT Value) distinct_values, count(DISTINCT Flag) distinct_flags
    FROM chemFour.main.filterbased_chemspecpm25_etad
    GROUP BY 1,2,3,4 HAVING count(*) > 1 ORDER BY n DESC LIMIT 5`);

  await report("chemspec_method_media", `
    WITH f AS (
      SELECT * FROM chemFour.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT * FROM chemFour.main.filterbased_chemspecpm25_uspa
    ) SELECT Method_Code, Analysis_Description, Collection_Description,
      count(*) measurement_rows, count(DISTINCT struct_pack(Site_Code,Filter_ID)) sample_keys
    FROM f GROUP BY 1,2,3 ORDER BY measurement_rows DESC LIMIT 20`);

  await report("ftir_hips_chemspec_filter_family", `
    WITH f AS (
      SELECT DISTINCT FilterId, Site,
        regexp_replace(FilterId, '-[0-9]+$', '') base_id
      FROM ftir.main.four_sites_ftir_data_v2
    ), h AS (
      SELECT FilterId FROM hips.main.spartan_hips_batch1_51_v2
    ), c AS (
      SELECT DISTINCT Site_Code, Filter_ID, Method_Code, Collection_Description
      FROM chemFour.main.filterbased_chemspecpm25_chts
      UNION SELECT DISTINCT Site_Code, Filter_ID, Method_Code, Collection_Description
      FROM chemFour.main.filterbased_chemspecpm25_etad
      UNION SELECT DISTINCT Site_Code, Filter_ID, Method_Code, Collection_Description
      FROM chemFour.main.filterbased_chemspecpm25_indh
      UNION SELECT DISTINCT Site_Code, Filter_ID, Method_Code, Collection_Description
      FROM chemFour.main.filterbased_chemspecpm25_uspa
    ) SELECT count(DISTINCT f.FilterId) ftir_filters,
      count(DISTINCT f.FilterId) FILTER (WHERE h.FilterId IS NOT NULL) exact_hips_filters,
      count(DISTINCT f.FilterId) FILTER (WHERE c.Filter_ID IS NOT NULL) chemspec_base_matches,
      count(DISTINCT f.FilterId) FILTER (WHERE c.Method_Code IN ('217','218')) chemspec_ftir_method_matches,
      count(DISTINCT f.FilterId) FILTER (WHERE c.Method_Code='221') chemspec_hips_method_matches,
      count(DISTINCT f.FilterId) FILTER (WHERE c.Collection_Description ILIKE '%Nylon%') nylon_base_matches,
      count(DISTINCT f.FilterId) FILTER (WHERE c.Collection_Description ILIKE '%mesh Teflon%') mesh_teflon_base_matches
    FROM f LEFT JOIN h USING (FilterId)
    LEFT JOIN c ON f.Site=c.Site_Code AND f.base_id=c.Filter_ID`);

  await report("ftir_chemspec_optical_identity", `
    WITH f AS (
      SELECT FilterId, min(Site) Site, min(SampleDate) SampleDate,
        regexp_replace(FilterId, '-[0-9]+$', '') base_id
      FROM ftir.main.four_sites_ftir_data_v2 GROUP BY FilterId
    ), c AS (
      SELECT Site_Code, Filter_ID, Method_Code, Collection_Description,
        try_strptime(concat(Start_Year_local,'-',lpad(cast(Start_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE start_date
      FROM chemFour.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT Site_Code, Filter_ID, Method_Code, Collection_Description,
        try_strptime(concat(Start_Year_local,'-',lpad(cast(Start_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE
      FROM chemFour.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT Site_Code, Filter_ID, Method_Code, Collection_Description,
        try_strptime(concat(Start_Year_local,'-',lpad(cast(Start_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE
      FROM chemFour.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT Site_Code, Filter_ID, Method_Code, Collection_Description,
        try_strptime(concat(Start_Year_local,'-',lpad(cast(Start_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE
      FROM chemFour.main.filterbased_chemspecpm25_uspa
    ) SELECT Method_Code, Collection_Description,
      count(DISTINCT f.FilterId) matched_full_ftir_filters,
      count(DISTINCT f.FilterId) FILTER (WHERE f.SampleDate IS NOT NULL AND c.start_date IS NOT NULL AND f.SampleDate<>c.start_date) known_date_mismatch_filters,
      count(DISTINCT f.FilterId) FILTER (WHERE f.SampleDate IS NULL OR c.start_date IS NULL) missing_date_filters
    FROM f JOIN c ON f.Site=c.Site_Code AND f.base_id=c.Filter_ID
    WHERE c.Method_Code IN ('217','218','221','316','318')
    GROUP BY 1,2 ORDER BY Method_Code, matched_full_ftir_filters DESC`);

  await report("hips_to_public_sampling", `
    WITH public_filters AS (
      SELECT DISTINCT Site_Code, Filter_ID,
        try_strptime(concat(cast(Start_Year_local AS VARCHAR), '-', lpad(cast(Start_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE start_date,
        try_strptime(concat(cast(End_Year_local AS VARCHAR), '-', lpad(cast(End_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(End_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE end_date
      FROM chemPublic.main.filterbased_chemspecpm25_etad
      UNION ALL SELECT DISTINCT Site_Code, Filter_ID,
        try_strptime(concat(cast(Start_Year_local AS VARCHAR), '-', lpad(cast(Start_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE,
        try_strptime(concat(cast(End_Year_local AS VARCHAR), '-', lpad(cast(End_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(End_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE
      FROM chemPublic.main.filterbased_chemspecpm25_chts
      UNION ALL SELECT DISTINCT Site_Code, Filter_ID,
        try_strptime(concat(cast(Start_Year_local AS VARCHAR), '-', lpad(cast(Start_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE,
        try_strptime(concat(cast(End_Year_local AS VARCHAR), '-', lpad(cast(End_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(End_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE
      FROM chemPublic.main.filterbased_chemspecpm25_indh
      UNION ALL SELECT DISTINCT Site_Code, Filter_ID,
        try_strptime(concat(cast(Start_Year_local AS VARCHAR), '-', lpad(cast(Start_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE,
        try_strptime(concat(cast(End_Year_local AS VARCHAR), '-', lpad(cast(End_Month_local AS VARCHAR),2,'0'), '-', lpad(cast(End_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE
      FROM chemPublic.main.filterbased_chemspecpm25_uspa
    ), k AS (SELECT Site_Code, Filter_ID, min(start_date) start_date, min(end_date) end_date, count(*) versions,
      count(DISTINCT struct_pack(start_date, end_date)) intervals FROM public_filters GROUP BY 1,2),
    h AS (SELECT FilterId, Site, SampleDate, regexp_replace(FilterId, '-[0-9]+$', '') base_id FROM hips.main.spartan_hips_batch1_51_v2)
    SELECT (SELECT count(*) FROM k) public_filter_keys,
      (SELECT count(*) FROM k WHERE intervals > 1) conflicting_public_intervals,
      (SELECT count(*) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID) matched_hips_rows,
      (SELECT count(DISTINCT h.base_id) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID) matched_base_ids,
      (SELECT count(*) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID WHERE h.SampleDate IS DISTINCT FROM k.start_date) sample_date_disagreements,
      (SELECT count(*) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID WHERE h.SampleDate < k.start_date OR h.SampleDate > k.end_date) sample_dates_outside_window`);

  await report("site_lookup", `
    SELECT count(*) row_count, count(DISTINCT SiteCode) site_codes,
      count(*) FILTER (WHERE Latitude IS NULL OR Longitude IS NULL) missing_coordinates
    FROM sites.main.spartan_site_quick_lookup`);

  const publicTables = (await connection.runAndReadAll(`
    SELECT table_name FROM information_schema.tables
    WHERE table_catalog = 'chemPublic' AND table_schema = 'main'
      AND table_name LIKE 'filterbased_chemspecpm25_%'
    ORDER BY table_name`)).getRowObjectsJson().map((row) => row.table_name);
  if (publicTables.length !== 37) throw new Error(`Expected 37 public ChemSpec PM2.5 tables; found ${publicTables.length}`);
  const allPublic = publicTables.map((table) => `SELECT Site_Code, Filter_ID, Start_Year_local, Start_Month_local, Start_Day_local, Start_hour_local, End_Year_local, End_Month_local, End_Day_local, End_hour_local, Hours_sampled FROM chemPublic.main."${table}"`).join(" UNION ALL ");
  await report("hips_network_sampling_windows", `
    WITH raw AS (${allPublic}),
    p AS (SELECT DISTINCT Site_Code, Filter_ID,
      try_strptime(concat(Start_Year_local,'-',lpad(cast(Start_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE start_date,
      try_strptime(concat(End_Year_local,'-',lpad(cast(End_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(End_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE end_date,
      Start_hour_local start_hour, End_hour_local end_hour, Hours_sampled sampled_hours FROM raw),
    k AS (SELECT Site_Code, Filter_ID, min(start_date) start_date, min(end_date) end_date,
      min(start_hour) start_hour, min(end_hour) end_hour,
      count(*) variants, count(DISTINCT struct_pack(start_date,end_date,start_hour,end_hour)) distinct_windows
      FROM p GROUP BY 1,2),
    h AS (SELECT FilterId, Site, SampleDate, regexp_replace(FilterId, '-[0-9]+$', '') base_id FROM hips.main.spartan_hips_batch1_51_v2)
    SELECT (SELECT count(*) FROM k) public_filter_keys,
      (SELECT count(*) FROM k WHERE distinct_windows > 1) conflicting_windows,
      (SELECT count(*) FROM k WHERE start_date IS NULL OR end_date IS NULL OR start_hour IS NULL OR end_hour IS NULL) incomplete_windows,
      (SELECT count(*) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID) hips_matches,
      (SELECT count(*) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID WHERE h.SampleDate IS NULL) matched_hips_missing_sample_date,
      (SELECT count(*) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID WHERE h.SampleDate IS NOT NULL AND h.SampleDate IS DISTINCT FROM k.start_date) known_hips_date_disagreements,
      (SELECT count(*) FROM h JOIN k ON h.Site=k.Site_Code AND h.base_id=k.Filter_ID WHERE k.end_date < k.start_date) inverted_windows`);

  await report("hips_network_sampling_exceptions", `
    WITH raw AS (${allPublic}), p AS (SELECT DISTINCT Site_Code, Filter_ID,
      try_strptime(concat(Start_Year_local,'-',lpad(cast(Start_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(Start_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE start_date,
      try_strptime(concat(End_Year_local,'-',lpad(cast(End_Month_local AS VARCHAR),2,'0'),'-',lpad(cast(End_Day_local AS VARCHAR),2,'0')), '%Y-%m-%d')::DATE end_date FROM raw),
    h AS (SELECT FilterId, Site, SampleDate, regexp_replace(FilterId, '-[0-9]+$', '') base_id FROM hips.main.spartan_hips_batch1_51_v2)
    SELECT h.FilterId, h.Site, h.SampleDate, p.start_date, p.end_date
    FROM h JOIN p ON h.Site=p.Site_Code AND h.base_id=p.Filter_ID
    WHERE h.SampleDate IS DISTINCT FROM p.start_date OR p.end_date < p.start_date
    ORDER BY h.FilterId LIMIT 10`);

  await report("improve_tor_grain", `
    WITH t AS (SELECT * FROM improveTor.main.improve_tor_fractions_all_sites)
    SELECT count(*) row_count, count(DISTINCT struct_pack(validation,source_file,SiteCode,POC,SampleDate)) unique_keys,
      count(*) FILTER (WHERE validation='validated') validated_rows,
      count(*) FILTER (WHERE validation='preliminary') preliminary_rows,
      count(*) FILTER (WHERE fractions_complete) complete_rows,
      min(SampleDate) first_sample, max(SampleDate) last_sample,
      count(DISTINCT SiteCode) site_count
    FROM t`);

  await report("improve_tor_vs_portal", `
    WITH t AS (SELECT * FROM improveTor.main.improve_tor_fractions_all_sites),
    p AS (SELECT SiteCode, POC, sample_date, ECf_Val, OCf_Val, Dataset FROM improvePortal.main.improve_chem_data)
    SELECT count(*) joined_rows, count(DISTINCT struct_pack(t.SiteCode,t.POC,t.SampleDate)) joined_sample_keys,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS DISTINCT FROM p.ECf_Val) changed_ec,
      count(*) FILTER (WHERE t.OC_TOR_ugm3 IS DISTINCT FROM p.OCf_Val) changed_oc,
      count(*) FILTER (WHERE t.validation='preliminary') preliminary_joins
    FROM t JOIN p ON t.SiteCode=p.SiteCode AND t.POC=p.POC AND t.SampleDate=p.sample_date`);

  await report("improve_tor_portal_value_detail", `
    WITH t AS (SELECT * FROM improveTor.main.improve_tor_fractions_all_sites),
    p AS (SELECT SiteCode, POC, sample_date, ECf_Val, OCf_Val FROM improvePortal.main.improve_chem_data)
    SELECT count(*) joined_rows,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS NOT NULL AND p.ECf_Val IS NOT NULL AND abs(t.EC_TOR_ugm3-p.ECf_Val)>0.000001) ec_numeric_disagreements,
      count(*) FILTER (WHERE t.OC_TOR_ugm3 IS NOT NULL AND p.OCf_Val IS NOT NULL AND abs(t.OC_TOR_ugm3-p.OCf_Val)>0.000001) oc_numeric_disagreements,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS NULL AND p.ECf_Val=-999) ec_sentinel_vs_null,
      count(*) FILTER (WHERE t.OC_TOR_ugm3 IS NULL AND p.OCf_Val=-999) oc_sentinel_vs_null,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS NOT NULL AND p.ECf_Val=-999) ec_portal_sentinel_new_value,
      count(*) FILTER (WHERE t.OC_TOR_ugm3 IS NOT NULL AND p.OCf_Val=-999) oc_portal_sentinel_new_value,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS NOT NULL AND p.ECf_Val IS NOT NULL AND p.ECf_Val<>-999 AND abs(t.EC_TOR_ugm3-p.ECf_Val)>0.000001) ec_real_value_changes,
      count(*) FILTER (WHERE t.OC_TOR_ugm3 IS NOT NULL AND p.OCf_Val IS NOT NULL AND p.OCf_Val<>-999 AND abs(t.OC_TOR_ugm3-p.OCf_Val)>0.000001) oc_real_value_changes,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS NULL AND p.ECf_Val IS NOT NULL AND p.ECf_Val<>-999) ec_new_missing,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS NOT NULL AND p.ECf_Val IS NULL) ec_portal_missing
    FROM t JOIN p ON t.SiteCode=p.SiteCode AND t.POC=p.POC AND t.SampleDate=p.sample_date`);

  await report("improve_tor_portal_changed_by_source", `
    WITH t AS (SELECT * FROM improveTor.main.improve_tor_fractions_all_sites),
    p AS (SELECT SiteCode, POC, sample_date, ECf_Val, OCf_Val FROM improvePortal.main.improve_chem_data)
    SELECT t.source_file, count(*) matched_rows,
      count(*) FILTER (WHERE t.EC_TOR_ugm3 IS NOT NULL AND p.ECf_Val IS NOT NULL AND abs(t.EC_TOR_ugm3-p.ECf_Val)>0.000001) ec_different,
      max(abs(t.EC_TOR_ugm3-p.ECf_Val)) FILTER (WHERE t.EC_TOR_ugm3 IS NOT NULL AND p.ECf_Val IS NOT NULL AND p.ECf_Val<>-999) max_ec_abs_difference
    FROM t JOIN p ON t.SiteCode=p.SiteCode AND t.POC=p.POC AND t.SampleDate=p.sample_date
    GROUP BY t.source_file ORDER BY t.source_file`);

  await report("improve_tor_portal_changed_examples", `
    WITH t AS (SELECT * FROM improveTor.main.improve_tor_fractions_all_sites),
    p AS (SELECT SiteCode, POC, sample_date, ECf_Val, OCf_Val FROM improvePortal.main.improve_chem_data)
    SELECT t.SiteCode, t.POC, t.SampleDate, t.source_file, t.EC_TOR_ugm3, p.ECf_Val,
      t.OC_TOR_ugm3, p.OCf_Val
    FROM t JOIN p ON t.SiteCode=p.SiteCode AND t.POC=p.POC AND t.SampleDate=p.sample_date
    WHERE t.EC_TOR_ugm3 IS NOT NULL AND p.ECf_Val IS NOT NULL AND abs(t.EC_TOR_ugm3-p.ECf_Val)>0.000001
    ORDER BY abs(t.EC_TOR_ugm3-p.ECf_Val) DESC LIMIT 5`);

  await report("improve_tor_pool_matches", `
    WITH pool AS (SELECT * FROM improveTor.main.improve_tor_fractions_pool_matches),
    all_sites AS (SELECT * FROM improveTor.main.improve_tor_fractions_all_sites)
    SELECT pool.source_match, count(*) pool_rows,
      count(*) FILTER (WHERE all_sites.SiteCode IS NOT NULL) exact_sample_key_matches
    FROM pool LEFT JOIN all_sites ON pool.Site=all_sites.SiteCode AND pool.SampleDate=all_sites.SampleDate
      AND pool.POC=all_sites.POC
    GROUP BY pool.source_match ORDER BY pool_rows DESC`);

  await report("ftir_analysis_id_grain", `
    WITH c AS (SELECT AnalysisId, FilterId, SampleDate, Site FROM ftirArchive.main.ftir_catalog_5318a8efc3),
    m AS (SELECT AnalysisId, FilterId, SampleDate, Site FROM ftirArchive.main.ftir_metadata_472f06c701)
    SELECT (SELECT count(*) FROM c) catalog_rows, (SELECT count(DISTINCT AnalysisId) FROM c) catalog_analysis_ids,
      (SELECT count(*) FROM m) metadata_rows, (SELECT count(DISTINCT AnalysisId) FROM m) metadata_analysis_ids,
      (SELECT count(*) FROM c JOIN m USING (AnalysisId)) joined_rows,
      (SELECT count(*) FROM c JOIN m USING (AnalysisId) WHERE c.FilterId IS DISTINCT FROM m.FilterId OR c.SampleDate IS DISTINCT FROM m.SampleDate OR c.Site IS DISTINCT FROM m.Site) differing_identity_fields`);
} finally {
  connection.disconnectSync();
  instance.closeSync();
  await rm(scratch, { recursive: true, force: true });
}
