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
        import numpy as np
        import torch
        import nibabel as nib
        from stl import mesh
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
        model.load_state_dict(torch.load("best_metric_model.pth", map_location=torch.device('cpu')))

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
        # Converint nii file to stl file


        import numpy as np
        import nibabel as nib
        from stl import mesh
        from skimage import measure


        # Load the 4D NIfTI file
        nii_file = nib.load(file_list[0])

        # Get the data as a NumPy array
        nii_data = nii_file.get_fdata()

        # Initialize empty lists to store vertices and faces
        all_verts = []
        all_faces = []

        # Iterate over each 3D volume in the 4D data
        for t in range(nii_data.shape[-1]):
            # Extract the 3D volume at time t
            volume = nii_data[..., t]

            # Generate a mesh for the isosurface at a threshold of 0
            verts, faces, normals, values = measure.marching_cubes(volume, 0)

            # Append the vertices and faces to the lists
            all_verts.append(verts)
            all_faces.append(faces)

        # Combine all vertices and faces into a single mesh
        combined_verts = np.concatenate(all_verts)
        combined_faces = np.concatenate(all_faces)

        # Create an STL mesh object for the combined mesh
        stl_mesh = mesh.Mesh(np.zeros(combined_faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(combined_faces):
            stl_mesh.vectors[i] = combined_verts[f]

        # Save the combined STL mesh to a single file
        stl_mesh.save('combined_output.stl')

        print("stl file saved")
        # Converting stl to obj file

        import trimesh

        # Load the STL file
        stl_file = r"combined_output.stl"
        mesh = trimesh.load_mesh(stl_file)

        # Save the mesh as an OBJ file in the Kaggle working directory
        obj_file = r"obj_pred.obj"
        mesh.export(obj_file, file_type='obj')
        print("obj file saved")

    def generateObject(fpath):
        import nibabel as nib
        import numpy as np
        from stl import mesh
        from skimage import measure
        

        nifti_file = nib.load(fpath)
        np_array = nifti_file.get_fdata()
        verts, faces, normals, values = measure.marching_cubes(np_array, 0)
        obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            obj_3d.vectors[i] = verts[f]
        obj_3d.save('mask.stl')
        # Save the combined STL mesh to a single file
        print("stl file saved")
        import trimesh
        # Converting STL to OBJ file
        stl_file = r"mask.stl"
        mesh = trimesh.load_mesh(stl_file)

        # Save the mesh as an OBJ file in the Kaggle working directory
        obj_file = r"mask.obj"
        mesh.export(obj_file, file_type='obj')
        print("obj file saved")


    
