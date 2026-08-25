"""Carga de la configuracion versionada del snapshot MNQ.

Este modulo no contiene ninguna logica de analisis ni de verificacion de
datos. Su unica responsabilidad es leer ``configs/mnq_snapshot.yaml``,
resolver las rutas relativas que contiene contra la raiz del repositorio, y
devolver un objeto tipado (``SnapshotConfig``) para que el resto del
pipeline nunca tenga que leer el YAML directamente ni repetir la logica de
resolucion de rutas.

Por que existe este modulo (motivacion, no solo mecanica):
Antes de esta etapa no existia ningun lugar versionado donde declarar
decisiones como el tick size del contrato o la frontera del hold-out (ver
``reports/mnq/00_initial_repository_audit.md``, seccion 5, riesgo 2:
"``configs/`` vacio"). Sin este modulo, esas decisiones terminarian como
numeros sueltos dentro del codigo de analisis: invisibles, dificiles de
auditar y faciles de cambiar sin dejar rastro. Centralizar la carga aqui
asegura que, si alguien cambia el tick size o la frontera del hold-out, lo
hace en un solo sitio (el YAML) y todo el pipeline lo hereda.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuracion resuelta y tipada del snapshot MNQ.

    Todos los campos de tipo ``Path`` ya estan resueltos como rutas
    absolutas (relativas a la raiz del repositorio, no al directorio de
    trabajo actual desde el que se invoque el script). Esto evita el error
    comun de que un script funcione al ejecutarlo desde una carpeta y falle
    al ejecutarlo desde otra.
    """

    repo_root: Path
    raw_dir: Path
    separator: str
    columns: list[str]
    timestamp_format: str
    timestamp_raw_timezone: str
    tick_size: float
    tick_size_source: str
    holdout_boundary_utc: str
    holdout_boundary_source: str
    research_files: list[str]
    holdout_files: list[str]
    reports_dir: Path
    interim_dir: Path
    inventory_report_name: str
    violations_report_name: str
    per_file_summary_name: str
    bad_data_mask_name: str

    # TDA-02 -- opcionales: una configuracion sintetica de test (TDA-00/
    # TDA-01) no necesita declarar la seccion "tda02" del YAML. Todos los
    # defaults son ``None``/cadena vacia, para que crear un SnapshotConfig
    # sin seccion tda02 (como hacen los tests existentes) siga funcionando
    # sin cambios; el codigo de TDA-02 exige explicitamente que estos
    # campos esten presentes antes de usarlos.
    tda02_calendar_name: str = ""
    tda02_calendar_buffer_days: int = 10
    tda02_coverage_report_name: str = ""
    tda02_coverage_by_year_csv_name: str = ""
    tda02_coverage_by_month_csv_name: str = ""
    tda02_gaps_csv_name: str = ""
    tda02_incomplete_days_csv_name: str = ""
    tda02_out_of_grid_csv_name: str = ""
    tda02_dst_evidence_csv_name: str = ""
    tda02_heatmap_png_name: str = ""
    tda02_inactive_bar_mask_name: str = ""

    # TDA-03 -- misma logica de opcionalidad que TDA-02 (ver arriba).
    tda03_min_incoming_share_shared: float = 0.50
    tda03_confirmation_sessions_required: int = 1
    tda03_extreme_jump_top_n: int = 40
    tda03_report_name: str = ""
    tda03_transitions_csv_name: str = ""
    tda03_overlap_daily_evidence_csv_name: str = ""
    tda03_no_overlap_evidence_csv_name: str = ""
    tda03_invariance_table_csv_name: str = ""
    tda03_discarded_rows_csv_name: str = ""
    tda03_adjustment_factors_csv_name: str = ""
    tda03_extreme_jumps_csv_name: str = ""
    tda03_roll_mask_name: str = ""
    tda03_canonical_series_name: str = ""

    # TDA-04 -- misma logica de opcionalidad que TDA-02/03 (ver arriba).
    tda04_report_name: str = ""
    tda04_variables_parquet_name: str = ""
    tda04_validity_mask_parquet_name: str = ""
    tda04_losses_by_cause_csv_name: str = ""
    tda04_th07_r_vs_R_csv_name: str = ""

    # TDA-05 -- misma logica de opcionalidad que TDA-02/03/04 (ver arriba).
    tda05_report_name: str = ""
    tda05_global_csv_name: str = ""
    tda05_by_hour_csv_name: str = ""
    tda05_by_year_csv_name: str = ""
    tda05_by_year_hour_csv_name: str = ""
    tda05_histogram_png_name: str = ""
    tda05_tick_variables_parquet_name: str = ""

    # TDA-06 -- misma logica de opcionalidad que TDA-02/03/04/05 (ver arriba).
    tda06_n_boot: int = 300
    tda06_extreme_quantile: float = 0.99
    tda06_smoothing_window_minutes: int = 15
    tda06_max_breakpoints: int = 6
    tda06_min_spacing_minutes: int = 60
    tda06_year_stability_tolerance_minutes: int = 15
    tda06_year_stability_min_years: int = 3
    tda06_report_name: str = ""
    tda06_minute_global_csv_name: str = ""
    tda06_minute_by_year_csv_name: str = ""
    tda06_weekday_csv_name: str = ""
    tda06_segmentation_csv_name: str = ""
    tda06_calibration_csv_name: str = ""
    tda06_profile_png_name: str = ""
    tda06_s_m_parquet_name: str = ""
    tda06_r_tilde_parquet_name: str = ""

    # TDA-07 -- misma logica de opcionalidad que TDA-02/03/04/05/06 (ver arriba).
    tda07_n_boot: int = 300
    tda07_report_name: str = ""
    tda07_th08_global_csv_name: str = ""
    tda07_th08_by_cause_csv_name: str = ""
    tda07_distribution_tables_csv_name: str = ""
    tda07_qq_global_png_name: str = ""
    tda07_qq_by_segment_png_name: str = ""
    tda07_qq_th08_png_name: str = ""

    # Complemento TH10 (pre-TDA-08) -- misma logica de opcionalidad que
    # TDA-02/03/04/05/06/07 (ver arriba). ``th10_horizons`` vacio usa el
    # default de codigo (``HORIZONS`` en th10_horizon_scaling.py).
    th10_horizons: list[int] = None  # type: ignore[assignment]
    th10_n_boot: int = 300
    th10_report_name: str = ""
    th10_var_by_horizon_csv_name: str = ""
    th10_var_by_horizon_non_overlap_csv_name: str = ""
    th10_beta_by_year_csv_name: str = ""
    th10_var_h_plot_png_name: str = ""

    # TDA-08 -- misma logica de opcionalidad que TDA-02/03/04/05/06/07 (ver arriba).
    tda08_n_boot: int = 300
    tda08_n_perm: int = 200
    tda08_report_name: str = ""
    tda08_acf_global_csv_name: str = ""
    tda08_pacf_csv_name: str = ""
    tda08_bootstrap_ci_csv_name: str = ""
    tda08_portmanteau_csv_name: str = ""
    tda08_g2_calibration_null1_csv_name: str = ""
    tda08_g2_portmanteau_null1_csv_name: str = ""
    tda08_g2_calibration_csv_name: str = ""
    tda08_g2_portmanteau_null_csv_name: str = ""
    tda08_g2_moment_check_csv_name: str = ""
    tda08_ticks_translation_csv_name: str = ""
    tda08_multi_frequency_csv_name: str = ""
    tda08_acf_by_group_csv_name: str = ""
    tda08_volume_decile_sensitivity_csv_name: str = ""
    tda08_windows_csv_name: str = ""
    tda08_windows_by_year_csv_name: str = ""
    tda08_windows_adjacent_csv_name: str = ""
    tda08_acf_plot_png_name: str = ""

    # TDA08-H -- complemento ACOTADO de TDA-08 (memoria lineal a 30/60 min).
    # TDA-08 permanece CERRADA/CONGELADA; esta seccion es evidencia separada.
    tda08h_report_name: str = ""
    tda08h_multi_horizon_csv_name: str = ""
    tda08h_plot_png_name: str = ""

    # TDA-09 -- dependencia en magnitud / volatility clustering (TH19-TH21).
    # misma logica de opcionalidad que TDA-02..TDA-08 (ver arriba). Las
    # grillas de rezagos/ordenes (MAX_LAG_MAGNITUDE, BOOTSTRAP_LAGS_MAGNITUDE,
    # PORTMANTEAU_M_MAGNITUDE, STABILITY_LAGS, ARCH_LM_ORDERS, TH21_ENERGY_M,
    # umbrales de STOP-9/flatness/TH20) son constantes predeclaradas en
    # ``tda09_volatility_clustering.py`` -- no valores de configuracion,
    # mismo criterio que TDA-08 aplico a su propia grilla de rezagos.
    tda09_n_boot: int = 300
    tda09_n_perm: int = 200
    tda09_n_perm_arch_lm: int = 60
    tda09_report_name: str = ""
    tda09_acf_csv_name: str = ""
    tda09_bootstrap_ci_csv_name: str = ""
    tda09_clock_attribution_csv_name: str = ""
    tda09_persistence_by_year_csv_name: str = ""
    tda09_persistence_by_segment_csv_name: str = ""
    tda09_persistence_rolling_csv_name: str = ""
    tda09_portmanteau_csv_name: str = ""
    tda09_arch_lm_csv_name: str = ""
    tda09_g2_calibration_null1_csv_name: str = ""
    tda09_g2_calibration_secondary_csv_name: str = ""
    tda09_g2_synthetic_moment_check_csv_name: str = ""
    tda09_mean_removal_sensitivity_csv_name: str = ""
    tda09_clock_flatness_csv_name: str = ""
    tda09_acf_triple_png_name: str = ""
    tda09_acf_raw_vs_adjusted_png_name: str = ""
    tda09_decay_png_name: str = ""

    # TDA-10 -- Escala versus forma: origen de las colas. Misma logica de
    # opcionalidad que TDA-02..TDA-09. La grilla de ventanas/half-lives, los
    # umbrales del veredicto y la ventana de causalidad son constantes
    # predeclaradas en ``tda10_scale_vs_shape.py`` -- solo ``n_boot``
    # (bootstrap de bloques por jornada para la CI de curtosis del
    # estimador primario) es genuinamente configurable, mismo criterio que
    # TDA-07/08/09 aplicaron a su propio bootstrap.
    tda10_n_boot: int = 300
    tda10_report_name: str = ""
    tda10_kurtosis_csv_name: str = ""
    tda10_kurtosis_bootstrap_ci_csv_name: str = ""
    tda10_quantile_by_decile_csv_name: str = ""
    tda10_quantile_by_segment_csv_name: str = ""
    tda10_quantile_by_year_csv_name: str = ""
    tda10_sensitivity_csv_name: str = ""
    tda10_causality_check_csv_name: str = ""
    tda10_qq_points_csv_name: str = ""
    tda10_qq_primary_png_name: str = ""
    tda10_qq_sensitivity_png_name: str = ""
    tda10_quantile_profile_png_name: str = ""

    @property
    def inventory_report_path(self) -> Path:
        return self.reports_dir / self.inventory_report_name

    @property
    def violations_report_path(self) -> Path:
        return self.reports_dir / self.violations_report_name

    @property
    def per_file_summary_path(self) -> Path:
        return self.reports_dir / self.per_file_summary_name

    @property
    def bad_data_mask_path(self) -> Path:
        return self.interim_dir / self.bad_data_mask_name

    @property
    def tda02_coverage_report_path(self) -> Path:
        return self.reports_dir / self.tda02_coverage_report_name

    @property
    def tda02_coverage_by_year_csv_path(self) -> Path:
        return self.reports_dir / self.tda02_coverage_by_year_csv_name

    @property
    def tda02_coverage_by_month_csv_path(self) -> Path:
        return self.reports_dir / self.tda02_coverage_by_month_csv_name

    @property
    def tda02_gaps_csv_path(self) -> Path:
        return self.reports_dir / self.tda02_gaps_csv_name

    @property
    def tda02_incomplete_days_csv_path(self) -> Path:
        return self.reports_dir / self.tda02_incomplete_days_csv_name

    @property
    def tda02_out_of_grid_csv_path(self) -> Path:
        return self.reports_dir / self.tda02_out_of_grid_csv_name

    @property
    def tda02_dst_evidence_csv_path(self) -> Path:
        return self.reports_dir / self.tda02_dst_evidence_csv_name

    @property
    def tda02_heatmap_png_path(self) -> Path:
        return self.reports_dir / self.tda02_heatmap_png_name

    @property
    def tda02_inactive_bar_mask_path(self) -> Path:
        return self.interim_dir / self.tda02_inactive_bar_mask_name

    @property
    def tda03_report_path(self) -> Path:
        return self.reports_dir / self.tda03_report_name

    @property
    def tda03_transitions_csv_path(self) -> Path:
        return self.reports_dir / self.tda03_transitions_csv_name

    @property
    def tda03_overlap_daily_evidence_csv_path(self) -> Path:
        return self.reports_dir / self.tda03_overlap_daily_evidence_csv_name

    @property
    def tda03_no_overlap_evidence_csv_path(self) -> Path:
        return self.reports_dir / self.tda03_no_overlap_evidence_csv_name

    @property
    def tda03_invariance_table_csv_path(self) -> Path:
        return self.reports_dir / self.tda03_invariance_table_csv_name

    @property
    def tda03_discarded_rows_csv_path(self) -> Path:
        return self.reports_dir / self.tda03_discarded_rows_csv_name

    @property
    def tda03_adjustment_factors_csv_path(self) -> Path:
        return self.reports_dir / self.tda03_adjustment_factors_csv_name

    @property
    def tda03_extreme_jumps_csv_path(self) -> Path:
        return self.reports_dir / self.tda03_extreme_jumps_csv_name

    @property
    def tda03_roll_mask_path(self) -> Path:
        return self.interim_dir / self.tda03_roll_mask_name

    @property
    def tda03_canonical_series_path(self) -> Path:
        return self.interim_dir / self.tda03_canonical_series_name

    @property
    def tda04_report_path(self) -> Path:
        return self.reports_dir / self.tda04_report_name

    @property
    def tda04_variables_parquet_path(self) -> Path:
        return self.interim_dir / self.tda04_variables_parquet_name

    @property
    def tda04_validity_mask_parquet_path(self) -> Path:
        return self.interim_dir / self.tda04_validity_mask_parquet_name

    @property
    def tda04_losses_by_cause_csv_path(self) -> Path:
        return self.reports_dir / self.tda04_losses_by_cause_csv_name

    @property
    def tda04_th07_r_vs_R_csv_path(self) -> Path:
        return self.reports_dir / self.tda04_th07_r_vs_R_csv_name

    @property
    def tda05_report_path(self) -> Path:
        return self.reports_dir / self.tda05_report_name

    @property
    def tda05_global_csv_path(self) -> Path:
        return self.reports_dir / self.tda05_global_csv_name

    @property
    def tda05_by_hour_csv_path(self) -> Path:
        return self.reports_dir / self.tda05_by_hour_csv_name

    @property
    def tda05_by_year_csv_path(self) -> Path:
        return self.reports_dir / self.tda05_by_year_csv_name

    @property
    def tda05_by_year_hour_csv_path(self) -> Path:
        return self.reports_dir / self.tda05_by_year_hour_csv_name

    @property
    def tda05_histogram_png_path(self) -> Path:
        return self.reports_dir / self.tda05_histogram_png_name

    @property
    def tda05_tick_variables_parquet_path(self) -> Path:
        return self.interim_dir / self.tda05_tick_variables_parquet_name

    @property
    def tda06_report_path(self) -> Path:
        return self.reports_dir / self.tda06_report_name

    @property
    def tda06_minute_global_csv_path(self) -> Path:
        return self.reports_dir / self.tda06_minute_global_csv_name

    @property
    def tda06_minute_by_year_csv_path(self) -> Path:
        return self.reports_dir / self.tda06_minute_by_year_csv_name

    @property
    def tda06_weekday_csv_path(self) -> Path:
        return self.reports_dir / self.tda06_weekday_csv_name

    @property
    def tda06_segmentation_csv_path(self) -> Path:
        return self.reports_dir / self.tda06_segmentation_csv_name

    @property
    def tda06_calibration_csv_path(self) -> Path:
        return self.reports_dir / self.tda06_calibration_csv_name

    @property
    def tda06_profile_png_path(self) -> Path:
        return self.reports_dir / self.tda06_profile_png_name

    @property
    def tda06_s_m_parquet_path(self) -> Path:
        return self.interim_dir / self.tda06_s_m_parquet_name

    @property
    def tda06_r_tilde_parquet_path(self) -> Path:
        return self.interim_dir / self.tda06_r_tilde_parquet_name

    @property
    def tda07_report_path(self) -> Path:
        return self.reports_dir / self.tda07_report_name

    @property
    def tda07_th08_global_csv_path(self) -> Path:
        return self.reports_dir / self.tda07_th08_global_csv_name

    @property
    def tda07_th08_by_cause_csv_path(self) -> Path:
        return self.reports_dir / self.tda07_th08_by_cause_csv_name

    @property
    def tda07_distribution_tables_csv_path(self) -> Path:
        return self.reports_dir / self.tda07_distribution_tables_csv_name

    @property
    def tda07_qq_global_png_path(self) -> Path:
        return self.reports_dir / self.tda07_qq_global_png_name

    @property
    def tda07_qq_by_segment_png_path(self) -> Path:
        return self.reports_dir / self.tda07_qq_by_segment_png_name

    @property
    def tda07_qq_th08_png_path(self) -> Path:
        return self.reports_dir / self.tda07_qq_th08_png_name

    @property
    def th10_report_path(self) -> Path:
        return self.reports_dir / self.th10_report_name

    @property
    def th10_var_by_horizon_csv_path(self) -> Path:
        return self.reports_dir / self.th10_var_by_horizon_csv_name

    @property
    def th10_var_by_horizon_non_overlap_csv_path(self) -> Path:
        return self.reports_dir / self.th10_var_by_horizon_non_overlap_csv_name

    @property
    def th10_beta_by_year_csv_path(self) -> Path:
        return self.reports_dir / self.th10_beta_by_year_csv_name

    @property
    def th10_var_h_plot_png_path(self) -> Path:
        return self.reports_dir / self.th10_var_h_plot_png_name

    @property
    def tda08_report_path(self) -> Path:
        return self.reports_dir / self.tda08_report_name

    @property
    def tda08_acf_global_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_acf_global_csv_name

    @property
    def tda08_pacf_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_pacf_csv_name

    @property
    def tda08_bootstrap_ci_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_bootstrap_ci_csv_name

    @property
    def tda08_portmanteau_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_portmanteau_csv_name

    @property
    def tda08_g2_calibration_null1_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_g2_calibration_null1_csv_name

    @property
    def tda08_g2_portmanteau_null1_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_g2_portmanteau_null1_csv_name

    @property
    def tda08_g2_calibration_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_g2_calibration_csv_name

    @property
    def tda08_g2_portmanteau_null_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_g2_portmanteau_null_csv_name

    @property
    def tda08_g2_moment_check_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_g2_moment_check_csv_name

    @property
    def tda08_ticks_translation_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_ticks_translation_csv_name

    @property
    def tda08_multi_frequency_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_multi_frequency_csv_name

    @property
    def tda08_acf_by_group_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_acf_by_group_csv_name

    @property
    def tda08_volume_decile_sensitivity_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_volume_decile_sensitivity_csv_name

    @property
    def tda08_windows_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_windows_csv_name

    @property
    def tda08_windows_by_year_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_windows_by_year_csv_name

    @property
    def tda08_windows_adjacent_csv_path(self) -> Path:
        return self.reports_dir / self.tda08_windows_adjacent_csv_name

    @property
    def tda08_acf_plot_png_path(self) -> Path:
        return self.reports_dir / self.tda08_acf_plot_png_name

    @property
    def tda08h_report_path(self) -> Path:
        return self.reports_dir / self.tda08h_report_name

    @property
    def tda08h_multi_horizon_csv_path(self) -> Path:
        return self.reports_dir / self.tda08h_multi_horizon_csv_name

    @property
    def tda08h_plot_png_path(self) -> Path:
        return self.reports_dir / self.tda08h_plot_png_name

    @property
    def tda09_report_path(self) -> Path:
        return self.reports_dir / self.tda09_report_name

    @property
    def tda09_acf_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_acf_csv_name

    @property
    def tda09_bootstrap_ci_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_bootstrap_ci_csv_name

    @property
    def tda09_clock_attribution_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_clock_attribution_csv_name

    @property
    def tda09_persistence_by_year_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_persistence_by_year_csv_name

    @property
    def tda09_persistence_by_segment_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_persistence_by_segment_csv_name

    @property
    def tda09_persistence_rolling_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_persistence_rolling_csv_name

    @property
    def tda09_portmanteau_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_portmanteau_csv_name

    @property
    def tda09_arch_lm_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_arch_lm_csv_name

    @property
    def tda09_g2_calibration_null1_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_g2_calibration_null1_csv_name

    @property
    def tda09_g2_calibration_secondary_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_g2_calibration_secondary_csv_name

    @property
    def tda09_g2_synthetic_moment_check_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_g2_synthetic_moment_check_csv_name

    @property
    def tda09_mean_removal_sensitivity_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_mean_removal_sensitivity_csv_name

    @property
    def tda09_clock_flatness_csv_path(self) -> Path:
        return self.reports_dir / self.tda09_clock_flatness_csv_name

    @property
    def tda09_acf_triple_png_path(self) -> Path:
        return self.reports_dir / self.tda09_acf_triple_png_name

    @property
    def tda09_acf_raw_vs_adjusted_png_path(self) -> Path:
        return self.reports_dir / self.tda09_acf_raw_vs_adjusted_png_name

    @property
    def tda09_decay_png_path(self) -> Path:
        return self.reports_dir / self.tda09_decay_png_name

    @property
    def tda10_report_path(self) -> Path:
        return self.reports_dir / self.tda10_report_name

    @property
    def tda10_kurtosis_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_kurtosis_csv_name

    @property
    def tda10_kurtosis_bootstrap_ci_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_kurtosis_bootstrap_ci_csv_name

    @property
    def tda10_quantile_by_decile_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_quantile_by_decile_csv_name

    @property
    def tda10_quantile_by_segment_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_quantile_by_segment_csv_name

    @property
    def tda10_quantile_by_year_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_quantile_by_year_csv_name

    @property
    def tda10_sensitivity_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_sensitivity_csv_name

    @property
    def tda10_causality_check_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_causality_check_csv_name

    @property
    def tda10_qq_points_csv_path(self) -> Path:
        return self.reports_dir / self.tda10_qq_points_csv_name

    @property
    def tda10_qq_primary_png_path(self) -> Path:
        return self.reports_dir / self.tda10_qq_primary_png_name

    @property
    def tda10_qq_sensitivity_png_path(self) -> Path:
        return self.reports_dir / self.tda10_qq_sensitivity_png_name

    @property
    def tda10_quantile_profile_png_path(self) -> Path:
        return self.reports_dir / self.tda10_quantile_profile_png_name

    def research_file_paths(self) -> list[Path]:
        """Rutas absolutas de los archivos del conjunto de investigacion."""
        return [self.raw_dir / name for name in self.research_files]

    def holdout_file_paths(self) -> list[Path]:
        """Rutas absolutas de los archivos del hold-out (no se abren en TDA-00)."""
        return [self.raw_dir / name for name in self.holdout_files]


def load_config(config_path: Path | str) -> SnapshotConfig:
    """Lee ``configs/mnq_snapshot.yaml`` y devuelve un ``SnapshotConfig``.

    Entrada
    -------
    config_path : ruta al archivo YAML de configuracion (tipicamente
        ``configs/mnq_snapshot.yaml``).

    Transformacion
    --------------
    1. Se resuelve ``config_path`` a una ruta absoluta.
    2. Se calcula ``repo_root`` como el directorio DOS niveles por encima
       del archivo de configuracion (``configs/mnq_snapshot.yaml`` ->
       ``configs/`` -> raiz del repo). Esto es lo que permite que todas las
       rutas del YAML (``data/raw/mnq``, ``reports/mnq``, ...) esten
       escritas de forma relativa y legible, sin que el pipeline dependa
       del directorio de trabajo actual.
    3. Se leen las secciones del YAML y se ensamblan en el dataclass.

    Salida
    ------
    ``SnapshotConfig`` con todas las rutas ya resueltas de forma absoluta.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    repo_root = config_path.parent.parent

    raw_data = raw["raw_data"]
    instrument_spec = raw["instrument_spec"]
    holdout = raw["holdout"]
    tda00 = raw["tda00"]
    tda02 = raw.get("tda02", {})
    tda03 = raw.get("tda03", {})
    tda04 = raw.get("tda04", {})
    tda05 = raw.get("tda05", {})
    tda06 = raw.get("tda06", {})
    tda07 = raw.get("tda07", {})
    th10 = raw.get("th10", {})
    tda08 = raw.get("tda08", {})
    tda08h = raw.get("tda08h", {})
    tda09 = raw.get("tda09", {})
    tda10 = raw.get("tda10", {})

    return SnapshotConfig(
        repo_root=repo_root,
        raw_dir=repo_root / raw_data["dir"],
        separator=raw_data["separator"],
        columns=list(raw_data["columns"]),
        timestamp_format=raw_data["timestamp_format"],
        timestamp_raw_timezone=raw_data["timestamp_raw_timezone"],
        tick_size=float(instrument_spec["tick_size"]),
        tick_size_source=instrument_spec["tick_size_source"],
        holdout_boundary_utc=holdout["boundary_utc"],
        holdout_boundary_source=holdout["boundary_source"],
        research_files=list(holdout["research_files"]),
        holdout_files=list(holdout["holdout_files"]),
        reports_dir=repo_root / tda00["output_dir_reports"],
        interim_dir=repo_root / tda00["output_dir_interim"],
        inventory_report_name=tda00["inventory_report"],
        violations_report_name=tda00["violations_report"],
        per_file_summary_name=tda00["per_file_summary"],
        bad_data_mask_name=tda00["bad_data_mask"],
        tda02_calendar_name=tda02.get("calendar_name", ""),
        tda02_calendar_buffer_days=int(tda02.get("calendar_buffer_days", 10)),
        tda02_coverage_report_name=tda02.get("coverage_report", ""),
        tda02_coverage_by_year_csv_name=tda02.get("coverage_by_year_csv", ""),
        tda02_coverage_by_month_csv_name=tda02.get("coverage_by_month_csv", ""),
        tda02_gaps_csv_name=tda02.get("gaps_csv", ""),
        tda02_incomplete_days_csv_name=tda02.get("incomplete_days_csv", ""),
        tda02_out_of_grid_csv_name=tda02.get("out_of_grid_csv", ""),
        tda02_dst_evidence_csv_name=tda02.get("dst_evidence_csv", ""),
        tda02_heatmap_png_name=tda02.get("heatmap_png", ""),
        tda02_inactive_bar_mask_name=tda02.get("inactive_bar_mask", ""),
        tda03_min_incoming_share_shared=float(tda03.get("min_incoming_share_shared", 0.50)),
        tda03_confirmation_sessions_required=int(tda03.get("confirmation_sessions_required", 1)),
        tda03_extreme_jump_top_n=int(tda03.get("extreme_jump_top_n", 40)),
        tda03_report_name=tda03.get("report_name", ""),
        tda03_transitions_csv_name=tda03.get("transitions_csv", ""),
        tda03_overlap_daily_evidence_csv_name=tda03.get("overlap_daily_evidence_csv", ""),
        tda03_no_overlap_evidence_csv_name=tda03.get("no_overlap_evidence_csv", ""),
        tda03_invariance_table_csv_name=tda03.get("invariance_table_csv", ""),
        tda03_discarded_rows_csv_name=tda03.get("discarded_rows_csv", ""),
        tda03_adjustment_factors_csv_name=tda03.get("adjustment_factors_csv", ""),
        tda03_extreme_jumps_csv_name=tda03.get("extreme_jumps_csv", ""),
        tda03_roll_mask_name=tda03.get("roll_mask_name", ""),
        tda03_canonical_series_name=tda03.get("canonical_series_name", ""),
        tda04_report_name=tda04.get("report_name", ""),
        tda04_variables_parquet_name=tda04.get("variables_parquet_name", ""),
        tda04_validity_mask_parquet_name=tda04.get("validity_mask_parquet_name", ""),
        tda04_losses_by_cause_csv_name=tda04.get("losses_by_cause_csv", ""),
        tda04_th07_r_vs_R_csv_name=tda04.get("th07_r_vs_R_csv", ""),
        tda05_report_name=tda05.get("report_name", ""),
        tda05_global_csv_name=tda05.get("global_csv", ""),
        tda05_by_hour_csv_name=tda05.get("by_hour_csv", ""),
        tda05_by_year_csv_name=tda05.get("by_year_csv", ""),
        tda05_by_year_hour_csv_name=tda05.get("by_year_hour_csv", ""),
        tda05_histogram_png_name=tda05.get("histogram_png", ""),
        tda05_tick_variables_parquet_name=tda05.get("tick_variables_parquet_name", ""),
        tda06_n_boot=int(tda06.get("n_boot", 300)),
        tda06_extreme_quantile=float(tda06.get("extreme_quantile", 0.99)),
        tda06_smoothing_window_minutes=int(tda06.get("smoothing_window_minutes", 15)),
        tda06_max_breakpoints=int(tda06.get("max_breakpoints", 6)),
        tda06_min_spacing_minutes=int(tda06.get("min_spacing_minutes", 60)),
        tda06_year_stability_tolerance_minutes=int(tda06.get("year_stability_tolerance_minutes", 15)),
        tda06_year_stability_min_years=int(tda06.get("year_stability_min_years", 3)),
        tda06_report_name=tda06.get("report_name", ""),
        tda06_minute_global_csv_name=tda06.get("minute_global_csv", ""),
        tda06_minute_by_year_csv_name=tda06.get("minute_by_year_csv", ""),
        tda06_weekday_csv_name=tda06.get("weekday_csv", ""),
        tda06_segmentation_csv_name=tda06.get("segmentation_csv", ""),
        tda06_calibration_csv_name=tda06.get("calibration_csv", ""),
        tda06_profile_png_name=tda06.get("profile_png", ""),
        tda06_s_m_parquet_name=tda06.get("s_m_parquet_name", ""),
        tda06_r_tilde_parquet_name=tda06.get("r_tilde_parquet_name", ""),
        tda07_n_boot=int(tda07.get("n_boot", 300)),
        tda07_report_name=tda07.get("report_name", ""),
        tda07_th08_global_csv_name=tda07.get("th08_global_csv", ""),
        tda07_th08_by_cause_csv_name=tda07.get("th08_by_cause_csv", ""),
        tda07_distribution_tables_csv_name=tda07.get("distribution_tables_csv", ""),
        tda07_qq_global_png_name=tda07.get("qq_global_png", ""),
        tda07_qq_by_segment_png_name=tda07.get("qq_by_segment_png", ""),
        tda07_qq_th08_png_name=tda07.get("qq_th08_png", ""),
        th10_horizons=list(th10["horizons"]) if th10.get("horizons") else None,
        th10_n_boot=int(th10.get("n_boot", 300)),
        th10_report_name=th10.get("report_name", ""),
        th10_var_by_horizon_csv_name=th10.get("var_by_horizon_csv", ""),
        th10_var_by_horizon_non_overlap_csv_name=th10.get("var_by_horizon_non_overlap_csv", ""),
        th10_beta_by_year_csv_name=th10.get("beta_by_year_csv", ""),
        th10_var_h_plot_png_name=th10.get("var_h_plot_png", ""),
        tda08_n_boot=int(tda08.get("n_boot", 300)),
        tda08_n_perm=int(tda08.get("n_perm", 200)),
        tda08_report_name=tda08.get("report_name", ""),
        tda08_acf_global_csv_name=tda08.get("acf_global_csv", ""),
        tda08_pacf_csv_name=tda08.get("pacf_csv", ""),
        tda08_bootstrap_ci_csv_name=tda08.get("bootstrap_ci_csv", ""),
        tda08_portmanteau_csv_name=tda08.get("portmanteau_csv", ""),
        tda08_g2_calibration_null1_csv_name=tda08.get("g2_calibration_null1_csv", ""),
        tda08_g2_portmanteau_null1_csv_name=tda08.get("g2_portmanteau_null1_csv", ""),
        tda08_g2_calibration_csv_name=tda08.get("g2_calibration_csv", ""),
        tda08_g2_portmanteau_null_csv_name=tda08.get("g2_portmanteau_null_csv", ""),
        tda08_g2_moment_check_csv_name=tda08.get("g2_moment_check_csv", ""),
        tda08_ticks_translation_csv_name=tda08.get("ticks_translation_csv", ""),
        tda08_multi_frequency_csv_name=tda08.get("multi_frequency_csv", ""),
        tda08_acf_by_group_csv_name=tda08.get("acf_by_group_csv", ""),
        tda08_volume_decile_sensitivity_csv_name=tda08.get("volume_decile_sensitivity_csv", ""),
        tda08_windows_csv_name=tda08.get("windows_csv", ""),
        tda08_windows_by_year_csv_name=tda08.get("windows_by_year_csv", ""),
        tda08_windows_adjacent_csv_name=tda08.get("windows_adjacent_csv", ""),
        tda08_acf_plot_png_name=tda08.get("acf_plot_png", ""),
        tda08h_report_name=tda08h.get("report_name", ""),
        tda08h_multi_horizon_csv_name=tda08h.get("multi_horizon_csv", ""),
        tda08h_plot_png_name=tda08h.get("plot_png", ""),
        tda09_n_boot=int(tda09.get("n_boot", 300)),
        tda09_n_perm=int(tda09.get("n_perm", 200)),
        tda09_n_perm_arch_lm=int(tda09.get("n_perm_arch_lm", 60)),
        tda09_report_name=tda09.get("report_name", ""),
        tda09_acf_csv_name=tda09.get("acf_csv", ""),
        tda09_bootstrap_ci_csv_name=tda09.get("bootstrap_ci_csv", ""),
        tda09_clock_attribution_csv_name=tda09.get("clock_attribution_csv", ""),
        tda09_persistence_by_year_csv_name=tda09.get("persistence_by_year_csv", ""),
        tda09_persistence_by_segment_csv_name=tda09.get("persistence_by_segment_csv", ""),
        tda09_persistence_rolling_csv_name=tda09.get("persistence_rolling_csv", ""),
        tda09_portmanteau_csv_name=tda09.get("portmanteau_csv", ""),
        tda09_arch_lm_csv_name=tda09.get("arch_lm_csv", ""),
        tda09_g2_calibration_null1_csv_name=tda09.get("g2_calibration_null1_csv", ""),
        tda09_g2_calibration_secondary_csv_name=tda09.get("g2_calibration_secondary_csv", ""),
        tda09_g2_synthetic_moment_check_csv_name=tda09.get("g2_synthetic_moment_check_csv", ""),
        tda09_mean_removal_sensitivity_csv_name=tda09.get("mean_removal_sensitivity_csv", ""),
        tda09_clock_flatness_csv_name=tda09.get("clock_flatness_csv", ""),
        tda09_acf_triple_png_name=tda09.get("acf_triple_png", ""),
        tda09_acf_raw_vs_adjusted_png_name=tda09.get("acf_raw_vs_adjusted_png", ""),
        tda09_decay_png_name=tda09.get("decay_png", ""),
        tda10_n_boot=int(tda10.get("n_boot", 300)),
        tda10_report_name=tda10.get("report_name", ""),
        tda10_kurtosis_csv_name=tda10.get("kurtosis_csv", ""),
        tda10_kurtosis_bootstrap_ci_csv_name=tda10.get("kurtosis_bootstrap_ci_csv", ""),
        tda10_quantile_by_decile_csv_name=tda10.get("quantile_by_decile_csv", ""),
        tda10_quantile_by_segment_csv_name=tda10.get("quantile_by_segment_csv", ""),
        tda10_quantile_by_year_csv_name=tda10.get("quantile_by_year_csv", ""),
        tda10_sensitivity_csv_name=tda10.get("sensitivity_csv", ""),
        tda10_causality_check_csv_name=tda10.get("causality_check_csv", ""),
        tda10_qq_points_csv_name=tda10.get("qq_points_csv", ""),
        tda10_qq_primary_png_name=tda10.get("qq_primary_png", ""),
        tda10_qq_sensitivity_png_name=tda10.get("qq_sensitivity_png", ""),
        tda10_quantile_profile_png_name=tda10.get("quantile_profile_png", ""),
    )
