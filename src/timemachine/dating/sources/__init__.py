"""One parser per date evidence source; each returns ``DateCandidate`` objects.

qs          hidden machine-readable "date QS" string (most reliable)
text        visible free-text date of the description page
exif        camera EXIF dates (+ scan/reproduction detection)
categories  dated category / subcategory names
title       dates in the file title (weak)
"""
