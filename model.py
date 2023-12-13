class monaimodel:
    def predict(input_path):
    # Import the necessary transformations from the MONAI library
        print("Start function")
        from monai.transforms import (
            AsDiscrete,                 # Transform to make label data discrete
            AsDiscreted,
            EnsureChannelFirstd,        # Ensure channel dimension is first in data
            Compose,                    # Compose a series of transformations
            CropForegroundd,            # Crop the foreground of an image using a mask
            LoadImaged,                 # Load image data from file paths
            Orientationd,               # Adjust image orientation based on specified codes
            RandAffined,                # Apply random affine transformations
            RandFlipd,                  # Randomly flip images and labels
            RandScaleIntensityd,        # Randomly scale intensity of the image
            RandShiftIntensityd,        # Randomly shift intensity of the image
            RandSpatialCropd,           # Randomly crop a spatial region from images and labels
            SaveImaged,                 # Save image data to file paths
            ScaleIntensityRanged,       # Scale intensity values within a specified range
            Spacingd,                   # Resample images to have a specified pixel spacing
            Invertd,                    # Invert intensity values of an image
        )
        print("TRansformations imported")

        from monai.data import (
            CacheDataset,                # A dataset that caches data in memory for faster access
            DataLoader,                 # DataLoader for iterating over batches of data
            Dataset,                    # Base dataset class for MONAI
            decollate_batch,             # Separate a batch of data into individual samples
        )
        from monai.networks.nets import UNet      # Import the UNet neural network architecture
        from monai.networks.layers import Norm 
        # import numpy as np
        import torch
        import nibabel as nib
        # from stl import mesh
        from skimage import measure
        from monai.transforms import Compose, EnsureChannelFirstd
        from monai.inferers import sliding_window_inference
        from monai.transforms import LoadImage  # LoadImage transform from MONAI
        from monai.handlers.utils import from_engine  # Utility for handling data from MONAI's engine
        import matplotlib.pyplot as plt  # Import the matplotlib library for visualization

        # Create a list of dictionaries where each dictionary contains image and label filenames
        test_files = [
            {"image": input_path}
        ]
        device = torch.device('cpu')

        print("fetched cpu device")
        # Define validation transforms for the image only
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),                        # Load image data
                EnsureChannelFirstd(keys=["image"]),               # Ensure channel dimension is first
                ScaleIntensityRanged(                            # Scale intensity values within a range
                    keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image",allow_smaller=True),  # Crop the foreground
                Orientationd(keys=["image"], axcodes="PLS"),       # Adjust image orientation
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),

            ]
        )
        # Create a dataset for testing using 'test_files' and apply validation transforms
        test_ds = Dataset(data=test_files, transform=test_transforms)

        # Create a data loader for the testing dataset with a batch size of 1
        test_loader = DataLoader(test_ds, batch_size=1)

        # Define a set of post-processing transforms for the model's predictions
        post_transforms = Compose([
            # Invert transformations applied during validation, to convert predictions back to original image space
            Invertd(
                keys="pred",                 # The key(s) to invert (in this case, "pred" is the output to be inverted)
                transform=test_transforms,     # The transform to be applied during inversion
                orig_keys="image",           # The key(s) of the original input data
                meta_keys="pred_meta_dict",   # The metadata key(s) associated with the output ("pred" metadata)
                orig_meta_keys="image_meta_dict",  # The metadata key(s) associated with the original input ("image" metadata)
                meta_key_postfix="meta_dict", # A postfix to add to keys to access the metadata
                nearest_interp=False,        # Interpolation method for resizing (False means no interpolation)
                to_tensor=True,              # Convert the result to a PyTorch tensor
            ),

            # Convert the model's predictions into a discrete format (e.g., one-hot encoding)
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            # Save the post-processed images to an output directory with specific postfix
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
        ])
        print("post transforms defined")
        # Create a LoadImage transform instance
        loader = LoadImage()
        model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            )

        model = model.to('cpu')

        # Load the model's best weights from a saved checkpoint
        model.load_state_dict(torch.load("The Fast API App\\best_metric_model.pth", map_location=torch.device('cpu')))

        print("model loaded")
        # Set the model to evaluation mode
        model.eval()

        # Iterate through the testing dataset
        with torch.no_grad():
            for test_data in test_loader:  # Loop through the test data
                test_inputs = test_data["image"].to(device)  # Move input data to the device (e.g., GPU)

                # Perform sliding window inference to generate predictions
                test_data["pred"] = sliding_window_inference(test_inputs, (96, 96, 96), 1, model, overlap=0.75)

                # Apply post-processing transforms to the predictions
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        
        import os
        import glob
        print("post transforms applied")
        folder = 'out'
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

       
        subfolder_path = subfolders[0]
        file_list = glob.glob(os.path.join(subfolder_path, '*'))

        if len(file_list) == 1 and os.path.isfile(file_list[0]):
            print(f'The file is: {file_list[0]}')
        else:
            print('No file or multiple files found in the subfolder.')
    

        print("file list fetched")
        import nibabel as nib
        import numpy as np
        from skimage import measure
        # Load NIfTI image
        nifti_img = nib.load(file_list[0])
        data = nifti_img.get_fdata()

        # Select a specific time frame (4th dimension)
        selected_data = data[..., 0]

        # Extract voxel spacing from NIfTI header
        voxel_spacing = nifti_img.header.get_zooms()[:3]

        # Extract mesh using skimage.measure
        verts, faces, _, _ = measure.marching_cubes(selected_data, level=0, spacing=voxel_spacing)

        # Save mesh as OBJ
        with open('obj_pred.obj', 'w') as obj_file:
            for v in verts:
                obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in faces:
                obj_file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
