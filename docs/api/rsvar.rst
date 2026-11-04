RS-VAR
======

The single-trajectory fit writes a reusable pre-delta forecast. The member
generator uses the same fitted regime structure but follows an alternate
output path; see :doc:`../workflow/rsvar` for interpretation.
Source: `run_rsvar.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/pipeline/run_rsvar.py>`_,
`run_ensemble.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/pipeline/run_ensemble.py>`_,
`anomaly_generator.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/model/anomaly_generator.py>`_.

.. py:function:: samy_wgen.pipeline.run_rsvar.run_and_save_rsvar(isd_path, lat, output_path, city, verbose=True, longitude=None)

   Fit RS-VAR to a prepared hourly LST CSV and save the forecast workbook.

   :param isd_path: Merged ISD-plus-solar CSV path.
   :param lat: Site latitude in degrees.
   :param output_path: XLSX output path.
   :param city: Bundled ASCII city code.
   :param verbose: Print stage diagnostics.
   :param longitude: Site longitude; resolved from city config when omitted.
   :returns: ``(forecast_dataframe, fitted_pipeline)``; the workbook, CSV, and solar calibration file are also written.

.. py:function:: samy_wgen.utils.solar_latent.encode_solar(frame: pandas.DataFrame, latitude: float, longitude: float) -> tuple[pandas.DataFrame, dict[str, int]]

   Screen historical irradiance and encode bounded clearness and beam-share
   coordinates for the joint RS-VAR fit.

   :param frame: LST-indexed frame containing GHI, DNI, and DHI.
   :param latitude: Site latitude.
   :param longitude: Site longitude.
   :returns: Latent-coordinate frame and screening counts.

.. py:function:: samy_wgen.utils.solar_latent.decode_solar(latent: pandas.DataFrame, latitude: float, longitude: float) -> pandas.DataFrame

   Reconstruct nonnegative, bounded, component-closing GHI/DNI/DHI from
   fitted solar coordinates.

   :param latent: LST-indexed frame with both solar latent columns.
   :param latitude: Site latitude.
   :param longitude: Site longitude.
   :returns: Irradiance DataFrame with GHI, DNI, and DHI.

.. py:function:: samy_wgen.pipeline.run_ensemble.run_and_save_ensemble(isd_path: str | pathlib.Path, lat: float, output_dir: str | pathlib.Path, n_members: int = 50, start_date: str = '2025-01-01', end_date: str = '2025-12-31', regime_mode: str = 'ensemble', regime_strength: float = 1.0, chunk_years: int = 5, seed: int = 0, target_vars: list[str] | None = None, verbose: bool = True) -> pathlib.Path

   Fit from prepared observations and write regime-path Parquet members and
   a summary workbook.

   :param isd_path: Prepared hourly LST ISD-plus-solar CSV.
   :param lat: Site latitude.
   :param output_dir: Destination for member files.
   :param n_members: Number of members.
   :param start_date: Inclusive first forecast date.
   :param end_date: Inclusive last forecast date.
   :param regime_mode: ``soft``, ``hard``, or ``ensemble``.
   :param regime_strength: Regime sampling intensity for ensemble mode.
   :param chunk_years: Nominal years per anomaly-centering chunk.
   :param seed: Base random seed.
   :param target_vars: Optional variables to retain.
   :param verbose: Print diagnostics.
   :returns: Output directory path.

.. py:class:: samy_wgen.model.anomaly_generator.AnomalyGenerator(fitted_model, regime_mode='ensemble', regime_strength=1.0, chunk_years=5)

   Construct a generator from a fitted ``ForecastTo2100V5`` model.

   :param fitted_model: Trained model with regime fits and transition data.
   :param regime_mode: ``soft``, ``hard``, or ``ensemble``.
   :param regime_strength: Regime sampling intensity.
   :param chunk_years: Nominal years per centering chunk.

   .. py:method:: generate(n_members: int, start_date: str = '2025-01-01', end_date: str = '2025-12-31', seed: int | None = None) -> list[pandas.DataFrame]

      Generate hourly members without a climate delta.

      :param n_members: Number of members.
      :param start_date: Inclusive first date.
      :param end_date: Inclusive last date.
      :param seed: Base seed; member index is added for reproducibility.
      :returns: One hourly DataFrame per member.
