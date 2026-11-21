# vehicle_dynamic_info (retrain-v1 note)

- ggv.csv: unchanged. 0 to 10 m/s, ax and ay limits 10 m/s^2 (about mu = 1.02). Kept on purpose (Kaipo, 2026-11-20).
- ax_max_machines.csv: the original file was the full size race car table (0 to 72 m/s).
  It is now limited to 0 to 10 m/s (the NeoRacer max speed is 6 m/s).
  The value 5.3 m/s^2 is unchanged because the NeoRacer motor acceleration has not been measured.
  For speeds up to 6 m/s this gives the same raceline speed profile as before.
