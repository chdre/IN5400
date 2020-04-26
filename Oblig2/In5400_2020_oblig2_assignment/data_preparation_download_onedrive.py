import utils_data_preparation_download_onedrive.download_from_onedrive as onedrive

download_dir = "data/coco/"

# download validation images
filename = "val2017.zip"
data_url = "http://images.cocodataset.org/zips/val2017.zip"
file_path = onedrive.download(url=data_url, download_dir=download_dir, filename=filename)
onedrive.extract(file_path=file_path, download_dir=download_dir)

# download vocabulary from Onedrive
filename = 'vocabulary.zip'
data_url = 'https://onedrive.live.com/download?cid=36039A0F53011CF6&resid=36039A0F53011CF6%21165452&authkey=ABuMhmfrJbE22WY'
file_path = onedrive.download(url=data_url, download_dir=download_dir, filename=filename)
onedrive.extract(file_path=file_path, download_dir=download_dir)


# download validation data cnn features from Onedrive
filename = 'Val2017_cnn_features.zip'
data_url = 'https://onedrive.live.com/download?cid=36039A0F53011CF6&resid=36039A0F53011CF6%21165451&authkey=AJsxllMIoDlXZr8'
file_path = onedrive.download(url=data_url, download_dir=download_dir, filename=filename)
onedrive.extract(file_path=file_path, download_dir=download_dir)

# download train data cnn features from Onedrive
filename = 'Train2017_cnn_features.zip'
data_url = 'https://onedrive.live.com/download?cid=36039A0F53011CF6&resid=36039A0F53011CF6%21165453&authkey=AJyHyUtwg0RJKw8'
file_path = onedrive.download(url=data_url, download_dir=download_dir, filename=filename)
onedrive.extract(file_path=file_path, download_dir=download_dir)
