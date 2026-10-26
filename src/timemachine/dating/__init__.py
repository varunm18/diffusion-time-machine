"""Estimating when a Commons image was made, and what kind of image it is.

Modules
-------
interval  ``DateInterval`` / ``DateCandidate`` and shared confidence levels.
sources   One parser per evidence source (hidden QS date, date text, EXIF,
          categories, subcategory/title names). Each returns candidates.
resolve   Combines candidates into one date, detecting known traps (upload
          date typed in as the date, scan dates in EXIF, conflicts).
medium    Image type (photo, painting, postcard, ...) and black-and-white hints.
pipeline  Runs dating + typing over a scene's metadata table.
"""
