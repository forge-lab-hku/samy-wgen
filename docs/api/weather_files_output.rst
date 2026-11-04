Weather files output
====================

``samy-run`` orchestrates EPW and DDY production. The following reusable
functions in ``samy_wgen.utils.create_epw_ddy`` support direct DDY creation.
The standard pipeline's year-specific EPW writer is currently private, so
call :doc:`../cli/run` for supported end-to-end export.
Source: `create_epw_ddy.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/utils/create_epw_ddy.py>`_.

.. py:function:: samy_wgen.utils.create_epw_ddy.compute_ddy_params(df: pandas.DataFrame, elevation_m: float = 0.0) -> dict

   Derive design-day parameter values from an hourly weather frame.

   :param df: Frame containing at least ``temp`` and ``dewpoint``; pressure and wind fields are used when present.
   :param elevation_m: Site elevation used for pressure estimation.
   :returns: Dictionary of design-day parameters.
   :raises ValueError: Required temperature or dew-point data are absent.

.. py:function:: samy_wgen.utils.create_epw_ddy.write_ddy_from_dataframe(df: pandas.DataFrame, out_path: pathlib.Path, *, city: str, state: str = '', country: str = '', lat: float = 0.0, lon: float = 0.0, tz: float = 0.0, elevation_m: float = 0.0, label: str = 'future') -> pathlib.Path

   Compute design-day parameters, render IDF text, and write a DDY file.

   :param df: Hourly weather DataFrame.
   :param out_path: Destination ``.ddy`` path.
   :param city: Location name for file metadata.
   :param state: State or region label.
   :param country: Country label.
   :param lat: Site latitude.
   :param lon: Site longitude.
   :param tz: Fixed timezone offset in hours.
   :param elevation_m: Site elevation.
   :param label: Design-day label such as ``future``.
   :returns: Written DDY path.

.. py:function:: samy_wgen.utils.create_epw_ddy.generate_ddy_from_epw(epw_path, energyplus_path)

   Read an EPW and write a neighbouring DDY file. The
   ``energyplus_path`` argument is retained for compatibility and is unused.

   :param epw_path: Source EPW path.
   :param energyplus_path: Compatibility argument.
   :returns: ``True`` on success, ``False`` otherwise.
