# Superseded

`make_figSI03_SI06.py` produced Figs. S3 and S6 before `fig_si03_all_models.py`
and `fig_si06_leave_one_out.py` existed. `dump_pieces.py` wrote an intermediate
`symbtr_pieces.csv` that `fig_si_gaprate.py` no longer reads — it takes the
committed `results/09_symbtr_piece_scores.csv` instead.

Neither is on the path that reproduces the published figures. They are kept only
as a record of how the earlier versions were made.
