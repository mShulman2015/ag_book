# Augmented Reality book

## To Run
Quick start: `python3 load_vid.py`

If one wants more fine grained control, such as over input or output file specifications rather than just reading from a file, one should go to `load_vid.py` at the top of which there is a list of variables that can be modified.

```
read_from_webcam - Tells the program if it should read from a webcam or a file

input_video_path and input_file - Tell the program where to find the input file if read_from_webcam == False
```

## To customize the run
If a custom run one should go into `vid_transform.py`, there in the `compute_final_frame` function, one can comment and uncomment the code in between the ##########s to see intermediate or simply lesser results than the original.

## To customize the run
To customize the contents of the book, one should go into `booklet_page_identifier.json` and modify the json there, as well as the files on the disk that are pointed to by the json.

```
original_dir - path to files for original photo
replacement_dir - path to files for replacement photo
popup_file_dir - path to files for popup_file obj

book_sheet_defenitions -- the contents of the booklet itself
  original - file name for original image (appended to path)
  replacement - file name for replacement image (appended to path)
  popup_file -- obj specifications
    file_name - file name for popup_file obj (appended to path)
    axies_index - index array to reorder xyz axis
    scale - scalar multiple per axis (can be negative)
    offset - 2D offset from the center of the image, which specifies were to place the object
    display_color - rgb touple to color to display the image
```

## QR code generation
While it is not necessary to generate the QR codes used in this project to test it, as all one needs to do to test it is to simply print the codes in `bookelet_defenition/original`, here is an explanation of how to do so, should one desire to.

Codes are generated from https://www.qr-code-generator.com/ using the `Text` tab, and `photo i` where `i` is the index of the photo (starting from 1)
