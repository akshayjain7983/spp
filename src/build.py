import zipfile

with zipfile.PyZipFile("spp_ml_predictor.zip", mode="w") as zip_pkg:
    zip_pkg.writepy('spp_ml_predictor')