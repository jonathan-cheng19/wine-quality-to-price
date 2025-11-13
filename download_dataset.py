import kagglehub

# Download latest version
path = kagglehub.dataset_download("budnyak/wine-rating-and-price")

print("Path to dataset files:", path)
