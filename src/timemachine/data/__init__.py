"""Training data: posed, dated views of a scene, served as SEVA-style samples.

views     the view table (posed + dated + photographic images of one COLMAP model)
images    download/resize view images into a local cache (+ pixel B&W detection)
cameras   pose/intrinsics conventions, SEVA camera normalization, Plücker rays
sampling  how input/target views are drawn (targets share one date)
dataset   the PyTorch Dataset (requires the ``torch`` extra)
"""
