Dataset tải ở: 

[ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)


# Setup nnU-Net
##### Bước 1
```powershell
pip install nnunet
```

##### Bước 2 Setup môi trường cho nnU-Net
```powershell
[System.Environment]::SetEnvironmentVariable(
    "nnUNet_raw_data_base",
    "C:\Users\T.Hung\nnunet_v1\raw",
    "User"
)

[System.Environment]::SetEnvironmentVariable(
    "nnUNet_preprocessed",
    "C:\Users\T.Hung\nnunet_v1\preprocessed",
    "User"
)

[System.Environment]::SetEnvironmentVariable(
    "RESULTS_FOLDER",
    "C:\Users\T.Hung\nnunet_v1\results",
    "User"
)

New-Item -ItemType Directory -Force -Path `
    "C:\Users\T.Hung\nnunet_v1\raw",
    "C:\Users\T.Hung\nnunet_v1\preprocessed",
    "C:\Users\T.Hung\nnunet_v1\results" | Out-Null

Write-Host "Done!"
```

##### Bước 3: Download pretrained model ACDC

```powershell
Invoke-WebRequest -Uri "https://zenodo.org/records/4003545/files/Task027_ACDC.zip?download=1" -OutFile "C:\Users\T.Hung\Task027_ACDC.zip"
```
