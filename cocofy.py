import os
from shutil import copy2

import json
import pandas as pd


df = pd.read_parquet("design5.parquet")

df["page_id"] = df["id"].str.extract("(.*#[0-9]-[0-9]{1,2})")
df["filename"] = df["page_id"]

df2 = df.drop_duplicates(["filename", "label_xgboost_adjusted"]).reset_index(drop=True)
df2["labels_per_page_count"] = df2.groupby("filename")["filename"].transform("count")

df3 = df2[
    ~((df2.labels_per_page_count > 1) & (df2.label_xgboost_adjusted == "other"))
].reset_index(drop=True)


df[["id", "filename"]][85:145]

df["filename"][0:38] = "dn_2021-04-20_partA_page1.jpg"
df["filename"][38:87] = "dn_2021-04-20_partA_page2.jpg"
df["filename"][87:143] = "dn_2021-04-20_partA_page3.jpg"
df["filename"][143:181] = "dn_2021-04-20_partA_page4.jpg"
df["filename"][181:193] = "dn_2021-04-20_partA_page5.jpg"
df["filename"][193:234] = "dn_2021-04-20_partA_page6.jpg"


def cocofy_from_list(filenames, categories, annotations, info=None):
    pass


def cocofy_from_df(
    df, categories, image_folder, destination_folder="cocofy", info=None, copy_images=True
):
    """
    A COCO conversion tool creating a minimally viable annotations.json for
    object detection from a dataframe. Also copies image files from source
    image folder to destination folder.
    
    df (pd.DataFrame): Dataframe with the following mandatory columns:
        - filename (base name of file without folders)
        - x
        - y 
        - width
        - height
        - image_width (image file width)
        - image_height (image file height)
        - label
        - iscrowd
        - ignore

    Read up on iscrowd and ignore in Coco docs. 
    Set them to 0 for all obs if you are unsure.

    categories (list): List with category names to be included as annotations. 
    Categories which are not listed will be ignored when creating bbox annotations. 
    Category names need to match label names in dataframe. 

    image_folder: Folder where your images are located. The filenames present in 
    df will be copied over to a folder named "cocofy/images" in your current
    working directory. 

    info (None | dict): Optional dictionary with information about your dataset.
    E.g. {"year": 2021, "version": "1.0", "contributor": "John Smith"}. 

    copy_images (bool): If True, copies images from image_folder to 
    {destination_folder}/images 

    """

    df = df[df["ignore"] == 0]

    # IMAGES
    df_images = df.drop_duplicates("filename")
    df_images = df_images[["filename", "image_width", "image_height"]].rename(
        columns={"filename": "file_name", "image_width": "width", "image_height": "height",}
    )
    images = df_images.to_dict("records")

    filenames_list = df["filename"].unique().tolist()
    filename_lookup = {filename: index for index, filename in enumerate(filenames_list)}

    for image in images:
        # Map filename to image_id
        image.update({"id": filename_lookup[image["file_name"]]})

    # CATEGORIES
    categories_lookup = {category: index for index, category in enumerate(categories)}
    categories_dicts = [{"name": category} for category in categories]

    for category in categories_dicts:
        # Map category to category_id
        category.update({"id": categories_lookup[category["name"]]})

    # ANNOTATIONS
    df_annotations = df[df["label"].isin(categories)].reset_index(drop=True)
    df_annotations["id"] = range(len(df_annotations))
    df_annotations["image_id"] = df_annotations["filename"].map(filename_lookup)
    df_annotations["category_id"] = df_annotations["label"].map(categories_lookup)
    df_annotations["bbox"] = df_annotations[["x", "y", "width", "height"]].values.tolist()
    df_annotations["area"] = df_annotations["width"] * df_annotations["height"]

    if "segmentation" not in df_annotations.columns:
        df_annotations["segmentation"] = [[] for _ in range(len(df_annotations))]

    annotations = df_annotations[
        ["id", "image_id", "category_id", "bbox", "area", "iscrowd", "ignore"]
    ].to_dict("records")

    # Output json
    annotation_json = {
        "images": images,
        "categories": categories_dicts,
        "annotations": annotations,
    }

    if info:
        annotation_json.append(info)

    os.makedirs(f"{destination_folder}/images", exist_ok=True)

    with open(f"{destination_folder}/annotations.json", "w") as f:
        json.dump(annotation_json, f, indent=4)

    # Copy image files
    if copy_images:
        for filename in filenames_list:
            copy2(
                src=f"{image_folder}/{os.path.basename(filename)}",
                dst=f"{destination_folder}/images",
            )


df["iscrowd"] = 0
df["ignore"] = 0

df_test = df[0:234]
df_test["image_width"] = 1000
df_test["image_height"] = 1600
df_test = df_test.rename(columns={"label_xgboost_adjusted": "label"})


cocofy_from_df(df_test, categories=["nyheter_sverige", "ledare", "debatt"], image_folder="images")

