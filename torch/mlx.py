import torchmetrics, mlxtend
print(f"mlxtend version: {mlxtend.__version__}")
assert int(mlxtend.__version__.split(".")[1] >= 19, "mlxtend version should be 0.19.0 or high")
