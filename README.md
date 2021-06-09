Helper functions to convert bounding box data organized in dataframes or other formats to COCO format annotations. 

Assuming you have data in similar format to the files found [here](https://github.com/Lauler/newspaper_section_data), and assuming you have a column in your dataframe named `filename`, you can generate COCO annotations in the following way:

```python
df = pd.read_parquet("design5.parquet")

df["iscrowd"] = 0
df["ignore"] = 0

# df["filename"] # Assumed to exist
# df["image_width"] = 1000 # Assumed to exist
# df["image_height"] = 1600 # Assumed to exist

df = df.rename(columns={"label_xgboost_adjusted": "label"})

# Output annotations for the 3 labels nyheter_sverige, ledare and debatt
cocofy_from_df(df_test, 
               categories=["nyheter_sverige", "ledare", "debatt"], 
               image_folder="images")
```

## Documentation

A COCO conversion tool creating a minimally viable annotations.json for
object detection from a dataframe. Also copies image files from source
image folder to destination folder.


-   **df (pd.DataFrame):** Dataframe with the following mandatory columns:
    - filename (base name of file without folders)
    - x
    - y
    - width 
    - height
    - image_width (image file width)
    - image_height (image file_height
    - label
    - iscrowd (see COCO docs, set to 0 if unsure)
    - ignore (see COCO docs, set to 0 if unsure)

-   **categories (list):** List with category names to be included as annotations. Categories which are not listed will be ignored when creating bbox annotations. Category names need to match label names in dataframe. 

-   **image_folder (str):** Source folder where your images are located. The filenames present in `df` will be copied over to a folder named "cocofy/images" in your current working directory. 

-   **destination_folder (str):** Folder where your annotations and images will be stored. Default: "cocofy". I.e. annotations.json will be placed in folder "cocofy", and images will be under "cocofy/images". 

-   **info (None | dict):** Optional dictionary with information about your dataset. E.g. {"year": 2021, "version": "1.0", "contributor": "John Smith"}. 

-   **copy_images (bool):** If True, copies images from image_folder to {destination_folder}/images 